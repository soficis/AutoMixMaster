#include "renderers/PhaseLimiterRenderer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <thread>
#include <vector>

#include <juce_core/juce_core.h>
#include <nlohmann/json.hpp>

#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/OfflineRenderPipeline.h"
#include "renderers/BuiltInRenderer.h"
#include "renderers/PhaseLimiterDiscovery.h"
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

    const std::filesystem::path outputPath = settings.outputPath.empty() ? "export_master.wav" : settings.outputPath;
    if (outputPath.has_parent_path()) {
      std::filesystem::create_directories(outputPath.parent_path());
    }

    const auto suffix = uniqueSuffix();
    const std::filesystem::path tempRoot = binaryInfo->installRoot / "tmp";
    const std::filesystem::path tempWorkDir = tempRoot / ("work_" + suffix);
    const std::filesystem::path tempInputPath = tempRoot / ("input_" + suffix + ".wav");
    const std::filesystem::path relativeTempWorkDir = std::filesystem::path("tmp") / ("work_" + suffix);
    const std::filesystem::path relativeTempInputPath = std::filesystem::path("tmp") / ("input_" + suffix + ".wav");
    std::filesystem::create_directories(tempWorkDir);

    util::WavWriter writer;
    writer.write(tempInputPath, rawMix, kPhaseLimiterBitDepth);

    CurrentWorkingDirectoryGuard workingDirectory(binaryInfo->installRoot);

    juce::StringArray command;
    command.add(binaryInfo->executablePath.generic_string());
    command.add("-input=" + relativeTempInputPath.generic_string());
    command.add("-output=" + outputPath.generic_string());
    command.add("-disable_input_encode=true");
    command.add("-output_format=wav");
    command.add("-sample_rate=44100");
    command.add("-bit_depth=16");
    command.add("-ceiling=-1");
    command.add("-mastering=false");
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
    if (exitCode != 0 || !pathExists(outputPath)) {
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
    const auto mastered = fileIO.readAudioFile(outputPath);

    automaster::HeuristicAutoMasterStrategy strategy;
    analysis::StemAnalyzer analyzer;
    const double integratedLufs = strategy.measureIntegratedLufs(mastered);
    const double truePeakDbtp = strategy.estimateTruePeakDbtp(mastered, 4);
    const auto spectrumMetrics = analyzer.analyzeBuffer(mastered);

    const std::filesystem::path reportPath = outputPath.string() + ".report.json";
    nlohmann::json report = {
        {"renderer", "PhaseLimiter"},
        {"phaseLimiterBinary", binaryInfo->executablePath.string()},
        {"outputAudioPath", outputPath.string()},
        {"integratedLufs", integratedLufs},
        {"truePeakDbtp", truePeakDbtp},
        {"spectrumLow", spectrumMetrics.lowEnergy},
        {"spectrumMid", spectrumMetrics.midEnergy},
        {"spectrumHigh", spectrumMetrics.highEnergy},
        {"stereoCorrelation", spectrumMetrics.stereoCorrelation},
        {"renderLogs", renderState.logs},
    };

    std::ofstream out(reportPath);
    out << report.dump(2);

    std::error_code ignore;
    std::filesystem::remove(tempInputPath, ignore);
    std::filesystem::remove_all(tempWorkDir, ignore);

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
