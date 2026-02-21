#include "renderers/SoxRenderer.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>
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
#include "renderers/SoxDiscovery.h"
#include "util/FileUtils.h"
#include "util/MetadataPolicy.h"
#include "util/MetadataSourceResolver.h"
#include "util/WavWriter.h"

namespace automix::renderers {
namespace {

using ::automix::util::isRegularFile;
using ::automix::util::metadataSourcePath;

constexpr size_t kMaxProcessOutputCaptureBytes = 32768;

std::string uniqueSuffix() {
  const auto value = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::to_string(value);
}

void drainProcessOutput(juce::ChildProcess& process, std::string& outputCapture) {
  char buffer[2048];
  for (;;) {
    const int bytesRead = process.readProcessOutput(buffer, static_cast<int>(sizeof(buffer)));
    if (bytesRead <= 0) {
      break;
    }

    if (outputCapture.size() < kMaxProcessOutputCaptureBytes) {
      const size_t remaining = kMaxProcessOutputCaptureBytes - outputCapture.size();
      const size_t toCopy = std::min(remaining, static_cast<size_t>(bytesRead));
      outputCapture.append(buffer, toCopy);
    }
  }
}

RenderResult unavailableResult(const std::string& reason) {
  RenderResult result;
  result.success = false;
  result.rendererName = "SoX";
  result.logs.push_back(reason);
  return result;
}

RenderResult cancelledResult(std::vector<std::string> logs = {}) {
  RenderResult result;
  result.success = false;
  result.cancelled = true;
  result.rendererName = "SoX";
  result.logs = std::move(logs);
  return result;
}

} // namespace

bool SoxRenderer::isAvailable() const {
  SoxDiscovery discovery;
  return discovery.find().has_value();
}

RenderResult SoxRenderer::render(const domain::Session& session,
                                 const domain::RenderSettings& settings,
                                 const ProgressCallback& onProgress,
                                 std::atomic_bool* cancelFlag) const {
  try {
    if (cancelFlag != nullptr && cancelFlag->load()) {
      return cancelledResult();
    }

    SoxDiscovery discovery;
    const auto binaryInfo = discovery.find();
    if (!binaryInfo.has_value()) {
      return unavailableResult("SoX binary not found. Bundle under assets/sox or set SOX_BIN.");
    }

    engine::OfflineRenderPipeline pipeline;
    auto renderState = pipeline.renderRawMix(
        session, settings,
        [&](const engine::RenderProgress& progress) {
          if (onProgress) {
            onProgress(progress.fraction * 0.50, progress.stage);
          }
        },
        cancelFlag);

    if (renderState.cancelled) {
      return cancelledResult(renderState.logs);
    }

    automaster::HeuristicAutoMasterStrategy strategy;
    analysis::StemAnalyzer analyzer;
    const bool usedSessionMasterPlan = session.masterPlan.has_value();
    const bool usedSessionMixPlan = session.mixPlan.has_value();
    auto plan = session.masterPlan.has_value()
                    ? session.masterPlan.value()
                    : strategy.buildPlan(domain::MasterPreset::DefaultStreaming, renderState.mixBuffer);
    if (!usedSessionMasterPlan && session.originalMixPath.has_value()) {
      try {
        engine::AudioFileIO fileIO;
        engine::AudioResampler resampler;
        auto originalMix = fileIO.readAudioFile(session.originalMixPath.value());
        if (originalMix.getSampleRate() != renderState.mixBuffer.getSampleRate()) {
          originalMix = resampler.resampleLinear(originalMix, renderState.mixBuffer.getSampleRate());
        }
        automaster::OriginalMixReference referenceTarget;
        plan = referenceTarget.applySoftTarget(plan, renderState.mixBuffer, originalMix, strategy, analyzer);
      } catch (const std::exception& error) {
        renderState.logs.push_back("Original mix reference skipped: " + std::string(error.what()));
      }
    }

    const std::filesystem::path outputPath = settings.outputPath.empty() ? "export_master.wav" : settings.outputPath;
    const auto outputFormat = util::WavWriter::resolveFormat(outputPath, settings.outputFormat);
    if (outputPath.has_parent_path()) {
      std::filesystem::create_directories(outputPath.parent_path());
    }

    const auto suffix = uniqueSuffix();
    const std::filesystem::path tempRoot = std::filesystem::temp_directory_path() / "automix_sox_renderer";
    std::filesystem::create_directories(tempRoot);

    const std::filesystem::path tempInputPath = tempRoot / ("sox_input_" + suffix + ".wav");
    const std::filesystem::path tempSoxOutputPath =
        outputFormat == "wav" ? outputPath : (tempRoot / ("sox_output_" + suffix + ".wav"));

    util::WavWriter writer;
    writer.write(tempInputPath, renderState.mixBuffer, std::clamp(settings.outputBitDepth, 16, 24));

    juce::StringArray command;
    command.add(binaryInfo->executablePath.generic_string());
    command.add(tempInputPath.generic_string());
    command.add(tempSoxOutputPath.generic_string());
    command.add("gain");
    command.add("-n");
    command.add(std::to_string(std::clamp(plan.limiterCeilingDb, -6.0, -0.1)));

    juce::ChildProcess process;
    if (!process.start(command)) {
      std::error_code ignore;
      std::filesystem::remove(tempInputPath, ignore);
      return unavailableResult("SoX renderer failed: unable to launch process.");
    }

    if (onProgress) {
      onProgress(0.65, "SoX processing");
    }

    std::string processOutput;
    while (process.isRunning()) {
      drainProcessOutput(process, processOutput);

      if (cancelFlag != nullptr && cancelFlag->load()) {
        process.kill();
        return cancelledResult();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    drainProcessOutput(process, processOutput);

    const int exitCode = process.getExitCode();
    if (exitCode != 0 || !isRegularFile(tempSoxOutputPath)) {
      std::error_code ignore;
      std::filesystem::remove(tempInputPath, ignore);
      if (tempSoxOutputPath != outputPath) {
        std::filesystem::remove(tempSoxOutputPath, ignore);
      }

      RenderResult failed = unavailableResult("SoX renderer failed with exit code " + std::to_string(exitCode) + ".");
      if (!processOutput.empty()) {
        failed.logs.push_back("SoX output: " + processOutput.substr(0, 240));
      }
      return failed;
    }

    if (onProgress) {
      onProgress(0.88, "SoX output validation");
    }

    engine::AudioFileIO fileIO;
    auto mastered = fileIO.readAudioFile(tempSoxOutputPath);
    std::map<std::string, std::string> sourceMetadata;
    if (const auto sourcePath = metadataSourcePath(session); sourcePath.has_value()) {
      try {
        sourceMetadata = fileIO.readMetadata(sourcePath.value());
      } catch (const std::exception& error) {
        renderState.logs.push_back("Metadata copy skipped: " + std::string(error.what()));
      }
    }
    std::vector<std::string> metadataPolicyNotes;
    const auto exportMetadata =
        util::applyMetadataPolicy(sourceMetadata, settings.metadataPolicy, settings.metadataTemplate, &metadataPolicyNotes);
    for (const auto& note : metadataPolicyNotes) {
      renderState.logs.push_back(note);
    }

    ai::MasteringCompliance compliance;
    const auto boundedPlan = compliance.enforcePlanBounds(plan);
    automaster::MasteringReport complianceReport;
    mastered = compliance.enforceOutput(mastered, boundedPlan, strategy, &complianceReport);
    writer.write(outputPath,
                 mastered,
                 settings.outputBitDepth,
                 settings.outputFormat,
                 settings.lossyBitrateKbps,
                 settings.lossyQuality,
                 settings.mp3UseVbr,
                 settings.mp3VbrQuality,
                 exportMetadata);

    const auto spectrumMetrics = analyzer.analyzeBuffer(mastered);

    std::filesystem::path reportPath;
    if (settings.writePerExportReportJson) {
      reportPath = outputPath.string() + ".report.json";
      nlohmann::json report = {
          {"renderer", "SoX"},
          {"soxBinary", binaryInfo->executablePath.string()},
          {"outputAudioPath", outputPath.string()},
          {"integratedLufs", complianceReport.integratedLufs},
          {"shortTermLufs", complianceReport.shortTermLufs},
          {"loudnessRange", complianceReport.loudnessRange},
          {"samplePeakDbfs", complianceReport.samplePeakDbfs},
          {"truePeakDbtp", complianceReport.truePeakDbtp},
          {"monoCorrelation", complianceReport.monoCorrelation},
          {"spectrumLow", spectrumMetrics.lowEnergy},
          {"spectrumMid", spectrumMetrics.midEnergy},
          {"spectrumHigh", spectrumMetrics.highEnergy},
          {"stereoCorrelation", spectrumMetrics.stereoCorrelation},
          {"masterPlanSource", usedSessionMasterPlan ? "session" : "heuristic"},
          {"mixPlanSource", usedSessionMixPlan ? "session" : "heuristic"},
          {"masterDecisionLog", boundedPlan.decisionLog},
          {"mixDecisionLog", session.mixPlan.has_value() ? session.mixPlan->decisionLog : std::vector<std::string>{}},
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
          {"renderLogs", renderState.logs},
      };

      std::ofstream out(reportPath);
      out << report.dump(2);
    }

    std::error_code ignore;
    std::filesystem::remove(tempInputPath, ignore);
    if (tempSoxOutputPath != outputPath) {
      std::filesystem::remove(tempSoxOutputPath, ignore);
    }

    RenderResult result;
    result.success = true;
    result.rendererName = "SoX";
    result.outputAudioPath = outputPath.string();
    result.reportPath = reportPath.empty() ? std::string {} : reportPath.string();
    result.logs.insert(result.logs.end(), renderState.logs.begin(), renderState.logs.end());
    result.logs.push_back("SoX executable: " + binaryInfo->executablePath.string());
    result.logs.push_back("SoX process exit code: " + std::to_string(exitCode));
    if (!processOutput.empty()) {
      result.logs.push_back("SoX output captured (truncated to 32KB).");
    }
    if (!settings.writePerExportReportJson) {
      result.logs.push_back("Report sidecar disabled (.report.json not written).");
    }
    result.logs.push_back("SoX completed.");

    if (onProgress) {
      onProgress(1.0, "SoX completed");
    }

    return result;
  } catch (const std::exception& error) {
    return unavailableResult("SoX renderer failed: " + std::string(error.what()));
  } catch (...) {
    return unavailableResult("SoX renderer failed: unknown error.");
  }
}

} // namespace automix::renderers
