#include "renderers/ExternalLimiterRenderer.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <thread>

#include <juce_core/juce_core.h>
#include <nlohmann/json.hpp>

#include "ai/MasteringCompliance.h"
#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "engine/AudioFileIO.h"
#include "engine/OfflineRenderPipeline.h"
#include "renderers/BuiltInRenderer.h"
#include "util/WavWriter.h"

namespace automix::renderers {
namespace {

RenderResult fallbackToBuiltIn(const domain::Session& session,
                               const domain::RenderSettings& settings,
                               const IRenderer::ProgressCallback& onProgress,
                               std::atomic_bool* cancelFlag,
                               const std::string& reason) {
  BuiltInRenderer fallback;
  auto result = fallback.render(session, settings, onProgress, cancelFlag);
  result.rendererName = "ExternalLimiter (fallback BuiltIn)";
  result.logs.push_back("External limiter fallback reason: " + reason);
  return result;
}

std::string captureChildOutput(juce::ChildProcess& process, const size_t maxBytes) {
  std::string output;
  char buffer[2048];
  while (const int bytes = process.readProcessOutput(buffer, static_cast<int>(sizeof(buffer)))) {
    if (bytes <= 0) {
      break;
    }
    if (output.size() >= maxBytes) {
      continue;
    }
    const auto remaining = maxBytes - output.size();
    output.append(buffer, static_cast<size_t>(std::min<int>(bytes, static_cast<int>(remaining))));
  }
  return output;
}

} // namespace

bool ExternalLimiterRenderer::isAvailable() const { return true; }

RenderResult ExternalLimiterRenderer::render(const domain::Session& session,
                                             const domain::RenderSettings& settings,
                                             const ProgressCallback& onProgress,
                                             std::atomic_bool* cancelFlag) const {
  if (settings.externalRendererPath.empty()) {
    return fallbackToBuiltIn(session, settings, onProgress, cancelFlag, "No external binary path configured.");
  }

  const std::filesystem::path binaryPath(settings.externalRendererPath);
  std::error_code error;
  if (!std::filesystem::is_regular_file(binaryPath, error) || error) {
    return fallbackToBuiltIn(session, settings, onProgress, cancelFlag, "External binary path is invalid.");
  }

  try {
    engine::OfflineRenderPipeline pipeline;
    auto rawResult = pipeline.renderRawMix(
        session, settings,
        [&](const engine::RenderProgress& progress) {
          if (onProgress) {
            onProgress(progress.fraction * 0.4, progress.stage);
          }
        },
        cancelFlag);

    if (rawResult.cancelled) {
      return RenderResult{.cancelled = true, .rendererName = "ExternalLimiter", .logs = rawResult.logs};
    }

    const std::filesystem::path outputPath = settings.outputPath.empty() ? "export_master.wav" : settings.outputPath;
    const auto outputFormat = util::WavWriter::resolveFormat(outputPath, settings.outputFormat);
    if (outputPath.has_parent_path()) {
      std::filesystem::create_directories(outputPath.parent_path());
    }

    const auto nonce = std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const std::filesystem::path outputDir = outputPath.has_parent_path() ? outputPath.parent_path() : std::filesystem::current_path();
    const std::filesystem::path tempInputPath = outputDir / ("external_limiter_input_" + nonce + ".wav");
    const std::filesystem::path tempLimiterOutputPath = outputFormat == "wav" ? outputPath
                                                                               : (outputDir / ("external_limiter_output_" + nonce + ".wav"));
    const std::filesystem::path requestPath = outputDir / ("external_limiter_request_" + nonce + ".json");

    util::WavWriter writer;
    writer.write(tempInputPath, rawResult.mixBuffer, settings.outputBitDepth);

    const auto plan = session.masterPlan.has_value() ? session.masterPlan.value()
                                                      : automaster::HeuristicAutoMasterStrategy().buildPlan(
                                                            domain::MasterPreset::DefaultStreaming, rawResult.mixBuffer);

    nlohmann::json request = {
        {"inputPath", tempInputPath.string()},
        {"outputPath", tempLimiterOutputPath.string()},
        {"sampleRate", settings.outputSampleRate},
        {"bitDepth", settings.outputBitDepth},
        {"targetLufs", plan.targetLufs},
        {"ceilingDbtp", plan.truePeakDbtp},
        {"limiterCeilingDb", plan.limiterCeilingDb},
        {"limiterTruePeakEnabled", plan.limiterTruePeakEnabled},
        {"limiterLookaheadMs", plan.limiterLookaheadMs},
        {"limiterAttackMs", plan.limiterAttackMs},
        {"limiterReleaseMs", plan.limiterReleaseMs},
        {"preGainDb", plan.preGainDb},
        {"outputFormat", outputFormat},
        {"lossyBitrateKbps", settings.lossyBitrateKbps},
        {"lossyQuality", settings.lossyQuality},
    };
    {
      std::ofstream out(requestPath);
      out << request.dump(2);
    }

    juce::StringArray command;
    command.add(binaryPath.string());
    command.add("--request");
    command.add(requestPath.string());

    juce::ChildProcess process;
    if (!process.start(command)) {
      return fallbackToBuiltIn(session, settings, onProgress, cancelFlag, "Failed to start external process.");
    }

    if (onProgress) {
      onProgress(0.5, "External limiter processing");
    }

    const auto timeoutMs = std::max(1000, settings.externalRendererTimeoutMs);
    auto start = std::chrono::steady_clock::now();
    while (process.isRunning()) {
      if (cancelFlag != nullptr && cancelFlag->load()) {
        process.kill();
        return RenderResult{.cancelled = true, .rendererName = "ExternalLimiter"};
      }

      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
      if (elapsed > timeoutMs) {
        process.kill();
        return fallbackToBuiltIn(session, settings, onProgress, cancelFlag, "External limiter timed out.");
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    const int exitCode = process.getExitCode();
    const std::string processOutput = captureChildOutput(process, 65536);
    if (exitCode != 0 || !std::filesystem::exists(tempLimiterOutputPath)) {
      return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                               "External limiter failed with exit code " + std::to_string(exitCode));
    }

    if (onProgress) {
      onProgress(0.8, "Compliance post-check");
    }

    engine::AudioFileIO fileIo;
    auto mastered = fileIo.readAudioFile(tempLimiterOutputPath);
    automaster::HeuristicAutoMasterStrategy strategy;
    ai::MasteringCompliance compliance;
    automaster::MasteringReport complianceReport;

    const auto boundedPlan = compliance.enforcePlanBounds(plan);
    const auto checked = compliance.enforceOutput(mastered, boundedPlan, strategy, &complianceReport);
    mastered = checked;
    writer.write(outputPath,
                 mastered,
                 settings.outputBitDepth,
                 settings.outputFormat,
                 settings.lossyBitrateKbps,
                 settings.lossyQuality);

    analysis::StemAnalyzer analyzer;
    const auto spectrum = analyzer.analyzeBuffer(mastered);

    const std::filesystem::path reportPath = outputPath.string() + ".report.json";
    nlohmann::json report = {
        {"renderer", "ExternalLimiter"},
        {"binaryPath", binaryPath.string()},
        {"outputAudioPath", outputPath.string()},
        {"processExitCode", exitCode},
        {"integratedLufs", complianceReport.integratedLufs},
        {"shortTermLufs", complianceReport.shortTermLufs},
        {"loudnessRange", complianceReport.loudnessRange},
        {"truePeakDbtp", complianceReport.truePeakDbtp},
        {"samplePeakDbfs", complianceReport.samplePeakDbfs},
        {"monoCorrelation", complianceReport.monoCorrelation},
        {"spectrumLow", spectrum.lowEnergy},
        {"spectrumMid", spectrum.midEnergy},
        {"spectrumHigh", spectrum.highEnergy},
        {"stereoCorrelation", spectrum.stereoCorrelation},
        {"outputFormat", outputFormat},
        {"lossyBitrateKbps", settings.lossyBitrateKbps},
        {"lossyQuality", settings.lossyQuality},
        {"targetLufs", boundedPlan.targetLufs},
        {"targetTruePeakDbtp", boundedPlan.truePeakDbtp},
        {"limiterCeilingDb", boundedPlan.limiterCeilingDb},
        {"limiterLookaheadMs", boundedPlan.limiterLookaheadMs},
        {"limiterAttackMs", boundedPlan.limiterAttackMs},
        {"limiterReleaseMs", boundedPlan.limiterReleaseMs},
        {"limiterTruePeakEnabled", boundedPlan.limiterTruePeakEnabled},
        {"processOutput", processOutput},
    };
    {
      std::ofstream out(reportPath);
      out << report.dump(2);
    }

    std::filesystem::remove(tempInputPath, error);
    std::filesystem::remove(requestPath, error);
    if (tempLimiterOutputPath != outputPath) {
      std::filesystem::remove(tempLimiterOutputPath, error);
    }

    RenderResult result;
    result.success = true;
    result.rendererName = "ExternalLimiter";
    result.outputAudioPath = outputPath.string();
    result.reportPath = reportPath.string();
    result.logs = rawResult.logs;
    result.logs.push_back("External limiter binary: " + binaryPath.string());
    result.logs.push_back("External process exit code: " + std::to_string(exitCode));
    if (!processOutput.empty()) {
      result.logs.push_back("External process output captured.");
    }

    if (onProgress) {
      onProgress(1.0, "External limiter completed");
    }
    return result;
  } catch (const std::exception& errorException) {
    return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                             "External limiter exception: " + std::string(errorException.what()));
  }
}

} // namespace automix::renderers
