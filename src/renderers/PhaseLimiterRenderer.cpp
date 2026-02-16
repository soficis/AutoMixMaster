#include "renderers/PhaseLimiterRenderer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
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
#include "renderers/PhaseLimiterDiscovery.h"
#include "util/MetadataPolicy.h"
#include "util/WavWriter.h"

namespace automix::renderers {
namespace {

constexpr int kPhaseLimiterSampleRate = 44100;
constexpr int kPhaseLimiterBitDepth = 16;
constexpr size_t kMaxProcessOutputCaptureBytes = 32768;

bool pathExists(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::exists(path, error);
}

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

class CurrentWorkingDirectoryGuard final {
 public:
  explicit CurrentWorkingDirectoryGuard(const std::filesystem::path& path)
      : previous_(std::filesystem::current_path()), changed_(false) {
    std::error_code error;
    std::filesystem::current_path(path, error);
    changed_ = !error;
  }

  ~CurrentWorkingDirectoryGuard() {
    if (!changed_) {
      return;
    }
    std::error_code error;
    std::filesystem::current_path(previous_, error);
  }

 private:
  std::filesystem::path previous_;
  bool changed_ = false;
};

std::string uniqueSuffix() {
  const auto value = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::to_string(value);
}

RenderResult fallbackToBuiltIn(const domain::Session& session,
                               const domain::RenderSettings& settings,
                               const IRenderer::ProgressCallback& onProgress,
                               std::atomic_bool* cancelFlag,
                               const std::string& reason) {
  try {
    BuiltInRenderer fallback;
    auto result = fallback.render(session, settings, onProgress, cancelFlag);
    result.rendererName = "PhaseLimiter (fallback BuiltIn)";
    result.logs.push_back("PhaseLimiter fallback reason: " + reason);
    return result;
  } catch (const std::exception& error) {
    RenderResult result;
    result.success = false;
    result.rendererName = "PhaseLimiter (fallback failed)";
    result.logs.push_back("PhaseLimiter fallback failed: " + std::string(error.what()));
    return result;
  } catch (...) {
    RenderResult result;
    result.success = false;
    result.rendererName = "PhaseLimiter (fallback failed)";
    result.logs.push_back("PhaseLimiter fallback failed: unknown error");
    return result;
  }
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

} // namespace

bool PhaseLimiterRenderer::isAvailable() const {
  PhaseLimiterDiscovery discovery;
  return discovery.find().has_value();
}

RenderResult PhaseLimiterRenderer::render(const domain::Session& session,
                                          const domain::RenderSettings& settings,
                                          const ProgressCallback& onProgress,
                                          std::atomic_bool* cancelFlag) const {
  try {
    if (cancelFlag != nullptr && cancelFlag->load()) {
      return RenderResult{.cancelled = true, .rendererName = "PhaseLimiter"};
    }

    PhaseLimiterDiscovery discovery;
    const auto binaryInfo = discovery.find();
    if (!binaryInfo.has_value()) {
      return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                               "PhaseLimiter binary not found in assets");
    }

    engine::OfflineRenderPipeline pipeline;
    auto renderState = pipeline.renderRawMix(
        session, settings,
        [&](const engine::RenderProgress& progress) {
          if (onProgress) {
            onProgress(progress.fraction * 0.5, progress.stage);
          }
        },
        cancelFlag);

    if (renderState.cancelled) {
      return RenderResult{.cancelled = true, .rendererName = "PhaseLimiter", .logs = renderState.logs};
    }

    engine::AudioBuffer rawMix = renderState.mixBuffer;
    if (rawMix.getSampleRate() != static_cast<double>(kPhaseLimiterSampleRate)) {
      engine::AudioResampler resampler;
      rawMix = resampler.resampleLinear(rawMix, static_cast<double>(kPhaseLimiterSampleRate));
    }

    automaster::HeuristicAutoMasterStrategy strategy;
    analysis::StemAnalyzer analyzer;
    const bool usedSessionMasterPlan = session.masterPlan.has_value();
    const bool usedSessionMixPlan = session.mixPlan.has_value();
    auto plan = session.masterPlan.has_value()
                    ? session.masterPlan.value()
                    : strategy.buildPlan(domain::MasterPreset::DefaultStreaming, rawMix);
    if (!usedSessionMasterPlan && session.originalMixPath.has_value()) {
      try {
        engine::AudioFileIO fileIO;
        engine::AudioResampler resampler;
        auto originalMix = fileIO.readAudioFile(session.originalMixPath.value());
        if (originalMix.getSampleRate() != rawMix.getSampleRate()) {
          originalMix = resampler.resampleLinear(originalMix, rawMix.getSampleRate());
        }
        automaster::OriginalMixReference referenceTarget;
        plan = referenceTarget.applySoftTarget(plan, rawMix, originalMix, strategy, analyzer);
      } catch (const std::exception& errorException) {
        renderState.logs.push_back("Original mix reference skipped: " + std::string(errorException.what()));
      }
    }

    const std::filesystem::path outputPath = settings.outputPath.empty() ? "export_master.wav" : settings.outputPath;
    const auto outputFormat = util::WavWriter::resolveFormat(outputPath, settings.outputFormat);
    if (outputPath.has_parent_path()) {
      std::filesystem::create_directories(outputPath.parent_path());
    }

    const auto suffix = uniqueSuffix();
    const std::filesystem::path tempRoot = binaryInfo->installRoot / "tmp";
    const std::filesystem::path tempWorkDir = tempRoot / ("work_" + suffix);
    const std::filesystem::path tempInputPath = tempRoot / ("input_" + suffix + ".wav");
    const std::filesystem::path tempPhaseOutputPath = outputFormat == "wav"
                                                          ? outputPath
                                                          : (tempRoot / ("phase_output_" + suffix + ".wav"));
    const std::filesystem::path relativeTempWorkDir = std::filesystem::path("tmp") / ("work_" + suffix);
    const std::filesystem::path relativeTempInputPath = std::filesystem::path("tmp") / ("input_" + suffix + ".wav");
    std::filesystem::create_directories(tempWorkDir);

    util::WavWriter writer;
    writer.write(tempInputPath, rawMix, kPhaseLimiterBitDepth);

    CurrentWorkingDirectoryGuard workingDirectory(binaryInfo->installRoot);

    juce::StringArray command;
    command.add(binaryInfo->executablePath.generic_string());
    command.add("-input=" + relativeTempInputPath.generic_string());
    command.add("-output=" + tempPhaseOutputPath.generic_string());
    command.add("-disable_input_encode=true");
    command.add("-output_format=wav");
    command.add("-sample_rate=44100");
    command.add("-bit_depth=" + std::to_string(std::clamp(settings.outputBitDepth, 16, 24)));
    command.add("-ceiling=" + std::to_string(plan.limiterCeilingDb));
    command.add("-mastering=true");
    command.add("-tmp=" + relativeTempWorkDir.generic_string());

    juce::ChildProcess process;
    if (!process.start(command)) {
      return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                               "failed to launch phase_limiter process");
    }

    if (onProgress) {
      onProgress(0.6, "PhaseLimiter processing");
    }

    std::string processOutput;
    while (process.isRunning()) {
      drainProcessOutput(process, processOutput);

      if (cancelFlag != nullptr && cancelFlag->load()) {
        process.kill();
        return RenderResult{.cancelled = true, .rendererName = "PhaseLimiter"};
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    drainProcessOutput(process, processOutput);

    const auto exitCode = process.getExitCode();
    if (exitCode != 0 || !pathExists(tempPhaseOutputPath)) {
      const std::string outputHint = processOutput.empty()
                                         ? ""
                                         : (" output=" + processOutput.substr(0, 240));
      return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                               "phase_limiter failed (exit=" + std::to_string(exitCode) + ")" + outputHint);
    }

    if (onProgress) {
      onProgress(0.9, "PhaseLimiter output validation");
    }

    engine::AudioFileIO fileIO;
    auto mastered = fileIO.readAudioFile(tempPhaseOutputPath);
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

    const std::filesystem::path reportPath = outputPath.string() + ".report.json";
    nlohmann::json report = {
        {"renderer", "PhaseLimiter"},
        {"phaseLimiterBinary", binaryInfo->executablePath.string()},
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
        {"exportSpeedMode", settings.exportSpeedMode},
        {"outputFormat", outputFormat},
        {"lossyBitrateKbps", settings.lossyBitrateKbps},
        {"lossyQuality", settings.lossyQuality},
        {"mp3Mode", settings.mp3UseVbr ? "vbr" : "cbr"},
        {"mp3VbrQuality", settings.mp3VbrQuality},
        {"metadataPolicy", settings.metadataPolicy},
        {"preGainDb", boundedPlan.preGainDb},
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

    std::error_code ignore;
    std::filesystem::remove(tempInputPath, ignore);
    std::filesystem::remove_all(tempWorkDir, ignore);
    if (tempPhaseOutputPath != outputPath) {
      std::filesystem::remove(tempPhaseOutputPath, ignore);
    }

    RenderResult result;
    result.success = true;
    result.rendererName = "PhaseLimiter";
    result.outputAudioPath = outputPath.string();
    result.reportPath = reportPath.string();
    result.logs.insert(result.logs.end(), renderState.logs.begin(), renderState.logs.end());
    result.logs.push_back("PhaseLimiter executable: " + binaryInfo->executablePath.string());
    result.logs.push_back("PhaseLimiter root: " + binaryInfo->installRoot.string());
    result.logs.push_back("PhaseLimiter process exit code: " + std::to_string(exitCode));
    if (!processOutput.empty()) {
      result.logs.push_back("PhaseLimiter output captured (truncated to 32KB).");
    }
    result.logs.push_back("PhaseLimiter completed.");

    if (onProgress) {
      onProgress(1.0, "PhaseLimiter completed");
    }

    return result;
  } catch (const std::exception& error) {
    return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                             "phase_limiter exception: " + std::string(error.what()));
  } catch (...) {
    return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                             "phase_limiter exception: unknown error");
  }
}

} // namespace automix::renderers
