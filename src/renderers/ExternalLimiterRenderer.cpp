#include "renderers/ExternalLimiterRenderer.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <thread>
#include <vector>

#include <juce_core/juce_core.h>
#include <nlohmann/json.hpp>

#include "ai/MasteringCompliance.h"
#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automaster/OriginalMixReference.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/OfflineRenderPipeline.h"
#include "renderers/BuiltInRenderer.h"
#include "util/MetadataPolicy.h"
#include "util/WavWriter.h"

namespace automix::renderers {
namespace {

std::optional<std::filesystem::path> metadataSourcePath(const domain::Session& session) {
  if (session.originalMixPath.has_value()) {
    const std::filesystem::path originalPath(session.originalMixPath.value());
    std::error_code error;
    if (std::filesystem::is_regular_file(originalPath, error) && !error) {
      return originalPath;
    }
  }

  for (const auto& stem : session.stems) {
    if (!stem.enabled || stem.filePath.empty()) {
      continue;
    }
    const std::filesystem::path stemPath(stem.filePath);
    std::error_code error;
    if (std::filesystem::is_regular_file(stemPath, error) && !error) {
      return stemPath;
    }
  }

  for (const auto& stem : session.stems) {
    if (stem.filePath.empty()) {
      continue;
    }
    const std::filesystem::path stemPath(stem.filePath);
    std::error_code error;
    if (std::filesystem::is_regular_file(stemPath, error) && !error) {
      return stemPath;
    }
  }

  return std::nullopt;
}

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

std::string summarizeOutput(const std::string& text, const size_t limit = 220) {
  if (text.size() <= limit) {
    return text;
  }
  return text.substr(0, limit) + "...";
}

std::optional<nlohmann::json> parseJsonFromOutput(const std::string& output) {
  if (output.empty()) {
    return std::nullopt;
  }

  try {
    return nlohmann::json::parse(output);
  } catch (...) {
  }

  const auto begin = output.find('{');
  const auto end = output.rfind('}');
  if (begin == std::string::npos || end == std::string::npos || begin >= end) {
    return std::nullopt;
  }

  try {
    return nlohmann::json::parse(output.substr(begin, end - begin + 1));
  } catch (...) {
    return std::nullopt;
  }
}

bool isCompatibleSchema(const nlohmann::json& json) {
  if (!json.contains("schemaVersion")) {
    return false;
  }

  if (json.at("schemaVersion").is_string()) {
    const auto schema = json.at("schemaVersion").get<std::string>();
    return schema.rfind("1.", 0) == 0;
  }
  if (json.at("schemaVersion").is_number_integer()) {
    return json.at("schemaVersion").get<int>() == 1;
  }
  return false;
}

} // namespace

ExternalLimiterRenderer::ValidationResult ExternalLimiterRenderer::validateBinary(const std::filesystem::path& binaryPath,
                                                                                  const int timeoutMs) {
  ValidationResult result;
  result.errorCode = "unknown";

  std::error_code error;
  if (!std::filesystem::is_regular_file(binaryPath, error) || error) {
    result.errorCode = "binary_missing";
    result.diagnostics = "Binary path is missing or not a file.";
    return result;
  }

  const auto nonce = std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  const auto outputDir = binaryPath.parent_path().empty() ? std::filesystem::current_path() : binaryPath.parent_path();
  const auto requestPath = outputDir / ("external_limiter_validate_" + nonce + ".json");

  nlohmann::json request = {
      {"validate", true},
      {"schemaVersion", "1.0"},
      {"request", "capabilities"},
      {"inputPath", ""},
      {"outputPath", ""},
      {"sampleRate", 44100},
      {"bitDepth", 24},
  };

  {
    std::ofstream out(requestPath);
    out << request.dump(2);
  }

  juce::StringArray command;
  command.add(binaryPath.string());
  command.add("--validate");
  command.add("--request");
  command.add(requestPath.string());

  juce::ChildProcess process;
  if (!process.start(command)) {
    std::filesystem::remove(requestPath, error);
    result.errorCode = "launch_failed";
    result.diagnostics = "Failed to launch binary with --validate.";
    return result;
  }

  auto start = std::chrono::steady_clock::now();
  const int timeout = std::max(500, timeoutMs);
  while (process.isRunning()) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    if (elapsed > timeout) {
      process.kill();
      std::filesystem::remove(requestPath, error);
      result.errorCode = "timeout";
      result.diagnostics = "Validation timed out.";
      return result;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }

  const int exitCode = process.getExitCode();
  const std::string processOutput = captureChildOutput(process, 65536);
  std::filesystem::remove(requestPath, error);

  if (exitCode != 0) {
    result.errorCode = "exit_code";
    result.diagnostics = "Validation exited with code " + std::to_string(exitCode) + ". Output: " + summarizeOutput(processOutput);
    return result;
  }

  const auto json = parseJsonFromOutput(processOutput);
  if (!json.has_value() || !json->is_object()) {
    result.errorCode = "invalid_json";
    result.diagnostics = "Validation output is not valid JSON.";
    return result;
  }

  if (!json->contains("version") || !json->at("version").is_string()) {
    result.errorCode = "missing_version";
    result.diagnostics = "Validation response missing string field 'version'.";
    return result;
  }

  if (!json->contains("supportedFeatures") || !json->at("supportedFeatures").is_array()) {
    result.errorCode = "missing_supported_features";
    result.diagnostics = "Validation response missing array field 'supportedFeatures'.";
    return result;
  }

  if (!isCompatibleSchema(*json)) {
    result.errorCode = "schema_incompatible";
    result.diagnostics = "Validation response has incompatible or missing schemaVersion.";
    return result;
  }

  result.version = json->at("version").get<std::string>();
  for (const auto& value : json->at("supportedFeatures")) {
    if (value.is_string()) {
      result.supportedFeatures.push_back(value.get<std::string>());
    }
  }

  result.valid = true;
  result.errorCode = "ok";
  result.diagnostics = "Validation passed.";
  return result;
}

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

  const auto validation = validateBinary(binaryPath, std::min(5000, settings.externalRendererTimeoutMs));
  if (!validation.valid) {
    return fallbackToBuiltIn(session,
                             settings,
                             onProgress,
                             cancelFlag,
                             "External binary validation failed [" + validation.errorCode + "]: " + validation.diagnostics);
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
      RenderResult cancelledResult;
      cancelledResult.cancelled = true;
      cancelledResult.rendererName = "ExternalLimiter";
      cancelledResult.logs = rawResult.logs;
      return cancelledResult;
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

    automaster::HeuristicAutoMasterStrategy strategy;
    analysis::StemAnalyzer analyzer;
    const bool usedSessionMasterPlan = session.masterPlan.has_value();
    const bool usedSessionMixPlan = session.mixPlan.has_value();
    auto plan = session.masterPlan.has_value() ? session.masterPlan.value()
                                               : strategy.buildPlan(domain::MasterPreset::DefaultStreaming, rawResult.mixBuffer);
    if (!usedSessionMasterPlan && session.originalMixPath.has_value()) {
      try {
        engine::AudioFileIO fileIO;
        engine::AudioResampler resampler;
        auto originalMix = fileIO.readAudioFile(session.originalMixPath.value());
        if (originalMix.getSampleRate() != rawResult.mixBuffer.getSampleRate()) {
          originalMix = resampler.resampleLinear(originalMix, rawResult.mixBuffer.getSampleRate());
        }
        automaster::OriginalMixReference referenceTarget;
        plan = referenceTarget.applySoftTarget(plan, rawResult.mixBuffer, originalMix, strategy, analyzer);
      } catch (const std::exception& errorException) {
        rawResult.logs.push_back("Original mix reference skipped: " + std::string(errorException.what()));
      }
    }

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
        {"exportSpeedMode", settings.exportSpeedMode},
        {"lossyBitrateKbps", settings.lossyBitrateKbps},
        {"lossyQuality", settings.lossyQuality},
        {"gpuExecutionProvider", settings.gpuExecutionProvider},
        {"preferHardwareAcceleration", settings.preferHardwareAcceleration},
        {"metadataPolicy", settings.metadataPolicy},
        {"metadataTemplate", settings.metadataTemplate},
        {"masterPlanSource", usedSessionMasterPlan ? "session" : "heuristic"},
        {"mixPlanSource", usedSessionMixPlan ? "session" : "heuristic"},
        {"masterDecisionLog", plan.decisionLog},
        {"mixDecisionLog", session.mixPlan.has_value() ? session.mixPlan->decisionLog : std::vector<std::string>{}},
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
        RenderResult cancelledResult;
        cancelledResult.cancelled = true;
        cancelledResult.rendererName = "ExternalLimiter";
        return cancelledResult;
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
    ai::MasteringCompliance compliance;
    automaster::MasteringReport complianceReport;
    std::map<std::string, std::string> sourceMetadata;
    if (const auto sourcePath = metadataSourcePath(session); sourcePath.has_value()) {
      try {
        sourceMetadata = fileIo.readMetadata(sourcePath.value());
      } catch (const std::exception& errorException) {
        rawResult.logs.push_back("Metadata copy skipped: " + std::string(errorException.what()));
      }
    }
    std::vector<std::string> metadataPolicyNotes;
    const auto exportMetadata =
        util::applyMetadataPolicy(sourceMetadata, settings.metadataPolicy, settings.metadataTemplate, &metadataPolicyNotes);
    for (const auto& note : metadataPolicyNotes) {
      rawResult.logs.push_back(note);
    }

    const auto boundedPlan = compliance.enforcePlanBounds(plan);
    const auto checked = compliance.enforceOutput(mastered, boundedPlan, strategy, &complianceReport);
    mastered = checked;
    writer.write(outputPath,
                 mastered,
                 settings.outputBitDepth,
                 settings.outputFormat,
                 settings.lossyBitrateKbps,
                 settings.lossyQuality,
                 settings.mp3UseVbr,
                 settings.mp3VbrQuality,
                 exportMetadata);

    const auto spectrum = analyzer.analyzeBuffer(mastered);

    const std::filesystem::path reportPath = outputPath.string() + ".report.json";
    nlohmann::json report = {
        {"renderer", "ExternalLimiter"},
        {"binaryPath", binaryPath.string()},
        {"validatedVersion", validation.version},
        {"validatedFeatures", validation.supportedFeatures},
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
        {"masterPlanSource", usedSessionMasterPlan ? "session" : "heuristic"},
        {"mixPlanSource", usedSessionMixPlan ? "session" : "heuristic"},
        {"masterDecisionLog", boundedPlan.decisionLog},
        {"mixDecisionLog", session.mixPlan.has_value() ? session.mixPlan->decisionLog : std::vector<std::string>{}},
        {"exportSpeedMode", settings.exportSpeedMode},
        {"outputFormat", outputFormat},
        {"lossyBitrateKbps", settings.lossyBitrateKbps},
        {"lossyQuality", settings.lossyQuality},
        {"mp3Mode", settings.mp3UseVbr ? "vbr" : "cbr"},
        {"mp3VbrQuality", settings.mp3VbrQuality},
        {"metadataPolicy", settings.metadataPolicy},
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
    result.logs.push_back("External limiter validation version: " + validation.version);
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
