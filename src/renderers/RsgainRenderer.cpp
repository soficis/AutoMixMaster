#include "renderers/RsgainRenderer.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
#include <utility>

#include <juce_core/juce_core.h>
#include <nlohmann/json.hpp>

#include "renderers/BuiltInRenderer.h"
#include "renderers/RsgainDiscovery.h"
#include "util/FileUtils.h"
#include "util/StringUtils.h"

namespace automix::renderers {
namespace {

using ::automix::util::isRegularFile;
using ::automix::util::toLower;

constexpr size_t kMaxProcessOutputCaptureBytes = 32768;

bool supportsReplayGainTagging(const std::filesystem::path& outputPath) {
  static const std::set<std::string> supportedExtensions = {
      ".flac",
      ".mp3",
      ".ogg",
      ".opus",
      ".m4a",
      ".mp4",
      ".wv",
      ".wma",
  };
  return supportedExtensions.contains(toLower(outputPath.extension().string()));
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
  result.rendererName = "rsgain";
  result.logs.push_back(reason);
  return result;
}

RenderResult cancelledResult(std::vector<std::string> logs = {}) {
  RenderResult result;
  result.success = false;
  result.cancelled = true;
  result.rendererName = "rsgain";
  result.logs = std::move(logs);
  return result;
}

} // namespace

bool RsgainRenderer::isAvailable() const {
  RsgainDiscovery discovery;
  return discovery.find().has_value();
}

RenderResult RsgainRenderer::render(const domain::Session& session,
                                    const domain::RenderSettings& settings,
                                    const ProgressCallback& onProgress,
                                    std::atomic_bool* cancelFlag) const {
  try {
    if (cancelFlag != nullptr && cancelFlag->load()) {
      return cancelledResult();
    }

    RsgainDiscovery discovery;
    const auto binaryInfo = discovery.find();
    if (!binaryInfo.has_value()) {
      return unavailableResult("rsgain binary not found. Bundle under assets/rsgain or set RSGAIN_BIN.");
    }

    BuiltInRenderer builtInRenderer;
    const auto builtInResult = builtInRenderer.render(
        session, settings,
        [&](const double fraction, const std::string& stage) {
          if (onProgress) {
            onProgress(fraction * 0.85, stage);
          }
        },
        cancelFlag);

    if (builtInResult.cancelled) {
      return cancelledResult(builtInResult.logs);
    }
    if (!builtInResult.success || builtInResult.outputAudioPath.empty()) {
      RenderResult failed = unavailableResult("rsgain renderer failed: built-in pre-render step failed.");
      failed.logs.insert(failed.logs.end(), builtInResult.logs.begin(), builtInResult.logs.end());
      return failed;
    }

    const std::filesystem::path outputPath = builtInResult.outputAudioPath;
    if (!isRegularFile(outputPath)) {
      RenderResult failed = unavailableResult("rsgain renderer failed: output file missing after pre-render.");
      failed.logs.insert(failed.logs.end(), builtInResult.logs.begin(), builtInResult.logs.end());
      return failed;
    }

    RenderResult result;
    result.rendererName = "rsgain";
    result.outputAudioPath = builtInResult.outputAudioPath;
    result.reportPath = builtInResult.reportPath;
    result.logs = builtInResult.logs;
    result.logs.push_back("rsgain executable: " + binaryInfo->executablePath.string());

    if (!supportsReplayGainTagging(outputPath)) {
      result.success = true;
      result.logs.push_back(
          "rsgain skipped: output format does not support ReplayGain tags (.flac/.mp3/.ogg/.opus/.m4a/.mp4/.wv/.wma).");
      if (onProgress) {
        onProgress(1.0, "rsgain completed (tagging skipped)");
      }
      return result;
    }

    juce::StringArray command;
    command.add(binaryInfo->executablePath.generic_string());
    command.add("easy");
    command.add(outputPath.generic_string());

    juce::ChildProcess process;
    if (!process.start(command)) {
      result.success = false;
      result.logs.push_back("rsgain failed: unable to launch process.");
      return result;
    }

    if (onProgress) {
      onProgress(0.92, "rsgain loudness tagging");
    }

    std::string processOutput;
    while (process.isRunning()) {
      drainProcessOutput(process, processOutput);
      if (cancelFlag != nullptr && cancelFlag->load()) {
        process.kill();
        return cancelledResult(result.logs);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    drainProcessOutput(process, processOutput);

    const int exitCode = process.getExitCode();
    result.success = exitCode == 0;
    result.logs.push_back("rsgain process exit code: " + std::to_string(exitCode));
    if (!processOutput.empty()) {
      result.logs.push_back("rsgain output: " + processOutput.substr(0, 240));
    }

    nlohmann::json report;
    bool reportLoaded = false;
    if (!result.reportPath.empty()) {
      try {
        std::ifstream in(result.reportPath);
        if (in.is_open()) {
          in >> report;
          reportLoaded = report.is_object();
        }
      } catch (...) {
        reportLoaded = false;
      }
    }
    if (!reportLoaded) {
      report = nlohmann::json::object();
    }

    report["renderer"] = "rsgain";
    report["rsgainBinary"] = binaryInfo->executablePath.string();
    report["rsgainApplied"] = result.success;
    report["rsgainOutput"] = processOutput.substr(0, 240);
    report["outputAudioPath"] = result.outputAudioPath;

    if (!result.reportPath.empty()) {
      std::ofstream out(result.reportPath);
      out << report.dump(2);
    }

    if (onProgress) {
      onProgress(1.0, result.success ? "rsgain completed" : "rsgain failed");
    }

    return result;
  } catch (const std::exception& error) {
    return unavailableResult("rsgain renderer failed: " + std::string(error.what()));
  } catch (...) {
    return unavailableResult("rsgain renderer failed: unknown error.");
  }
}

} // namespace automix::renderers
