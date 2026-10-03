#include "renderers/PhaseLimiterRenderer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
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
#include "util/FileUtils.h"
#include "util/MetadataPolicy.h"
#include "util/MetadataSourceResolver.h"
#include "util/WavWriter.h"

namespace automix::renderers {
namespace {

using ::automix::util::metadataSourcePath;
using ::automix::util::pathFromUtf8;
using ::automix::util::pathToUtf8;

constexpr int kPhaseLimiterSampleRate = 44100;
constexpr int kPhaseLimiterBitDepth = 16;
constexpr size_t kMaxProcessOutputCaptureBytes = 32768;

bool pathExists(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::exists(path, error);
}

// Per-render scratch space under the OS temp directory, removed on every way
// out of render(). It used to live in installRoot/tmp and was cleaned only on
// success, so each failed run leaked its input WAV and work directory (13 GB on
// one machine) - and installRoot is read-only in a Program Files install.
class ScratchDirectory final {
 public:
  explicit ScratchDirectory(std::filesystem::path path) : path_(std::move(path)) {
    std::error_code error;
    std::filesystem::create_directories(path_ / "work", error);
  }
  ~ScratchDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  ScratchDirectory(const ScratchDirectory&) = delete;
  ScratchDirectory& operator=(const ScratchDirectory&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::filesystem::path scratchBase() { return std::filesystem::temp_directory_path() / "automix_phaselimiter"; }

bool startsWith(const std::string& text, const std::string& prefix) { return text.rfind(prefix, 0) == 0; }

// Once per process: remove what earlier versions leaked into installRoot/tmp
// (only their own file patterns, never anything else there) and scratch runs
// of ours older than a day, which only a crash leaves behind.
void sweepLeftoverScratch(const std::filesystem::path& installRoot) {
  static std::once_flag once;
  std::call_once(once, [&installRoot] {
    std::error_code error;
    const auto legacy = installRoot / "tmp";
    if (std::filesystem::is_directory(legacy, error)) {
      for (const auto& entry : std::filesystem::directory_iterator(legacy, error)) {
        const auto name = entry.path().filename().string();
        const bool leaked = (startsWith(name, "input_") && entry.path().extension() == ".wav") ||
                            (startsWith(name, "phase_output_") && entry.path().extension() == ".wav") ||
                            (startsWith(name, "work_") && entry.is_directory(error));
        if (leaked) {
          std::filesystem::remove_all(entry.path(), error);
        }
      }
    }
    const auto cutoff = std::filesystem::file_time_type::clock::now() - std::chrono::hours(24);
    if (std::filesystem::is_directory(scratchBase(), error)) {
      for (const auto& entry : std::filesystem::directory_iterator(scratchBase(), error)) {
        if (entry.is_directory(error) && std::filesystem::last_write_time(entry.path(), error) < cutoff) {
          std::filesystem::remove_all(entry.path(), error);
        }
      }
    }
  });
}
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
  const auto found = PhaseLimiterDiscovery{}.find();
  return found.has_value() && isCompleteInstall(*found) && ffmpegForPhaseLimiter().has_value();
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
    if (!isCompleteInstall(*binaryInfo)) {
      return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                               "PhaseLimiter install is incomplete: missing " +
                                   pathToUtf8(masteringReferencePath(*binaryInfo)));
    }

    const auto ffmpegPath = ffmpegForPhaseLimiter();
    if (!ffmpegPath.has_value()) {
      RenderResult result;
      result.success = false;
      result.rendererName = "PhaseLimiter";
      result.logs.push_back(
          "PhaseLimiter needs ffmpeg, which was not found. Install ffmpeg or set FFMPEG_BIN.");
      return result;
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

    const std::filesystem::path outputPath =
        settings.outputPath.empty() ? std::filesystem::path("export_master.wav") : pathFromUtf8(settings.outputPath);
    const auto outputFormat = util::WavWriter::resolveFormat(outputPath, settings.outputFormat);
    if (outputPath.has_parent_path()) {
      std::filesystem::create_directories(outputPath.parent_path());
    }

    sweepLeftoverScratch(binaryInfo->installRoot);
    const ScratchDirectory scratch(scratchBase() / uniqueSuffix());
    const std::filesystem::path tempWorkDir = scratch.path() / "work";
    const std::filesystem::path tempInputPath = scratch.path() / "input.wav";
    const std::filesystem::path tempPhaseOutputPath = scratch.path() / "phase_output.wav";

    util::WavWriter writer;
    writer.write(tempInputPath, rawMix, kPhaseLimiterBitDepth);

    // Absolute paths throughout: no process-wide working-directory change,
    // which raced with parallel renders.
    juce::StringArray command;
    command.add(pathToUtf8(binaryInfo->executablePath));
    command.add("-input=" + pathToUtf8(tempInputPath));
    command.add("-output=" + pathToUtf8(tempPhaseOutputPath));
    command.add("-ffmpeg=" + pathToUtf8(*ffmpegPath));
    command.add("-mastering_reference_file=" + pathToUtf8(masteringReferencePath(*binaryInfo)));
    command.add("-sound_quality2_cache=" + pathToUtf8(soundQualityCachePath(*binaryInfo)));
    command.add("-disable_input_encode=true");
    command.add("-output_format=wav");
    command.add("-sample_rate=44100");
    command.add("-bit_depth=" + std::to_string(std::clamp(settings.outputBitDepth, 16, 24)));
    command.add("-ceiling=" + std::to_string(plan.limiterCeilingDb));
    command.add("-mastering=true");
    command.add("-tmp=" + pathToUtf8(tempWorkDir));

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
    const bool hasOutput = pathExists(tempPhaseOutputPath);
    if (!hasOutput) {
      std::string snippet;
      if (processOutput.size() <= 500) {
        snippet = processOutput;
      } else {
        snippet = processOutput.substr(0, 200) + "\n...[snip]...\n" +
                  processOutput.substr(processOutput.size() - 300);
      }
      const std::string outputHint = snippet.empty() ? "" : (" output=" + snippet);
      return fallbackToBuiltIn(session, settings, onProgress, cancelFlag,
                               "phase_limiter failed (exit=" + std::to_string(exitCode) + ")" + outputHint);
    }
    if (exitCode != 0) {
      renderState.logs.push_back(
          "PhaseLimiter returned non-zero exit code (" + std::to_string(exitCode) +
          ") but produced an output file; continuing with generated audio.");
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

    std::filesystem::path reportPath;
    if (settings.writePerExportReportJson) {
      reportPath = pathFromUtf8(pathToUtf8(outputPath) + ".report.json");
      nlohmann::json report = {
          {"renderer", "PhaseLimiter"},
          {"phaseLimiterBinary", pathToUtf8(binaryInfo->executablePath)},
          {"outputAudioPath", pathToUtf8(outputPath)},
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
    }


    RenderResult result;
    result.success = true;
    result.rendererName = "PhaseLimiter";
    result.outputAudioPath = pathToUtf8(outputPath);
    result.reportPath = reportPath.empty() ? std::string {} : pathToUtf8(reportPath);
    result.logs.insert(result.logs.end(), renderState.logs.begin(), renderState.logs.end());
    result.logs.push_back("PhaseLimiter executable: " + pathToUtf8(binaryInfo->executablePath));
    result.logs.push_back("PhaseLimiter root: " + pathToUtf8(binaryInfo->installRoot));
    result.logs.push_back("PhaseLimiter process exit code: " + std::to_string(exitCode));
    if (!processOutput.empty()) {
      result.logs.push_back("PhaseLimiter output captured (truncated to 32KB).");
    }
    if (!settings.writePerExportReportJson) {
      result.logs.push_back("Report sidecar disabled (.report.json not written).");
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
