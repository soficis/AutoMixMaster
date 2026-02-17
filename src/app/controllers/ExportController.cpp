#include "app/controllers/ExportController.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <optional>
#include <sstream>

#include "analysis/StemHealthAssistant.h"
#include "renderers/ExternalLimiterRenderer.h"
#include "renderers/RendererFactory.h"
#include "util/BackgroundJob.h"
#include "util/CallbackDispatch.h"
#include "util/LameDownloader.h"
#include "util/StringUtils.h"

namespace automix::app {
namespace {

using ::automix::util::toLower;
using ::automix::util::extensionForFormat;

constexpr const char* kExportSpeedModeFinal = "final";
constexpr const char* kExportSpeedModeBalanced = "balanced";
constexpr const char* kExportSpeedModeQuick = "quick";

struct ExportHealthCacheEntry {
  std::string key;
  juce::String text;
  bool hasCriticalIssues = false;
  size_t issueCount = 0;
};

std::mutex& exportHealthCacheMutex() {
  static std::mutex mutex;
  return mutex;
}

std::optional<ExportHealthCacheEntry>& exportHealthCache() {
  static std::optional<ExportHealthCacheEntry> cache;
  return cache;
}

std::string buildHealthCacheKey(const domain::Session& session,
                                const std::vector<analysis::StemAnalysisEntry>& analysisEntries) {
  std::ostringstream key;
  key << "stems=" << session.stems.size() << "|entries=" << analysisEntries.size() << '|';

  for (const auto& stem : session.stems) {
    key << stem.id << ':' << stem.filePath << ':' << stem.enabled;
    if (stem.busId.has_value()) {
      key << ':' << stem.busId.value();
    }

    std::error_code error;
    const auto writeTime = std::filesystem::last_write_time(stem.filePath, error);
    if (!error) {
      const auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(writeTime.time_since_epoch()).count();
      key << ':' << ticks;
    }
    const auto size = std::filesystem::file_size(stem.filePath, error);
    if (!error) {
      key << ':' << size;
    }
    key << '|';
  }

  if (session.mixPlan.has_value()) {
    key << "mixPlan:" << session.mixPlan->dryWet
        << ':' << session.mixPlan->mixBusHeadroomDb
        << ':' << session.mixPlan->stemDecisions.size();
  } else {
    key << "mixPlan:none";
  }

  return key.str();
}

} // namespace

void ExportController::clearHealthCache() {
  std::scoped_lock lock(exportHealthCacheMutex());
  exportHealthCache().reset();
}

ExportController::ExportController(juce::ThreadPool& threadPool, Callbacks callbacks)
    : threadPool_(threadPool), callbacks_(std::move(callbacks)) {}

std::vector<util::WavWriter::FormatAvailability> ExportController::listCodecAvailability() const {
  return util::WavWriter::getAvailableFormats();
}

std::string ExportController::selectedExportSpeedMode(const int selectedId,
                                                      const std::map<int, std::string>& modeByComboId) const {
  const auto it = modeByComboId.find(selectedId);
  if (it == modeByComboId.end()) {
    return kExportSpeedModeFinal;
  }
  return it->second;
}

bool ExportController::isQuickExportMode(const std::string& exportSpeedMode) const {
  return toLower(exportSpeedMode) == kExportSpeedModeQuick;
}

QuickExportDefaults ExportController::quickExportDefaults(
    const std::map<int, std::string>& codecFormatByComboId) const {
  QuickExportDefaults defaults;
  const auto availability = util::WavWriter::getAvailableFormats();

  const auto isAvailable = [&availability](const std::string& formatName) {
    const auto entry = std::find_if(availability.begin(), availability.end(), [&](const auto& candidate) {
      return toLower(candidate.format) == toLower(formatName);
    });
    return entry != availability.end() && entry->available;
  };

  if (isAvailable("mp3")) {
    defaults.outputFormat = "mp3";
    return defaults;
  }

  for (const auto& [comboId, formatName] : codecFormatByComboId) {
    juce::ignoreUnused(comboId);
    if (isAvailable(formatName)) {
      defaults.outputFormat = formatName;
      defaults.usedFallbackCodec = true;
      return defaults;
    }
  }

  defaults.usedFallbackCodec = true;
  return defaults;
}

ExportCodecControls ExportController::codecControlsFor(const std::string& selectedFormat,
                                                       const bool mp3UseVbr,
                                                       const std::string& exportSpeedMode) const {
  const bool lossy = util::WavWriter::isLossyFormat(selectedFormat);
  const bool mp3 = toLower(selectedFormat) == "mp3";
  const bool vbr = mp3 && mp3UseVbr;
  const bool quickMode = isQuickExportMode(exportSpeedMode);

  ExportCodecControls controls;
  controls.formatEnabled = !quickMode;
  controls.bitrateEnabled = !quickMode && lossy && !vbr;
  controls.mp3ModeEnabled = !quickMode && mp3;
  controls.mp3VbrEnabled = !quickMode && vbr;
  return controls;
}

domain::RenderSettings ExportController::buildRenderSettings(const BuildRenderSettingsRequest& request) const {
  domain::RenderSettings settings;
  settings.exportSpeedMode = request.exportSpeedMode.empty() ? kExportSpeedModeFinal : request.exportSpeedMode;
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;
  settings.outputBitDepth = 24;
  if (settings.exportSpeedMode == kExportSpeedModeBalanced) {
    settings.blockSize = 2048;
  } else if (settings.exportSpeedMode == kExportSpeedModeQuick) {
    settings.blockSize = 4096;
    settings.outputBitDepth = 16;
  }
  settings.processingThreads = 0;
  settings.preferHardwareAcceleration = true;
  settings.metadataPolicy = request.metadataPolicy.empty() ? "copy_all" : request.metadataPolicy;
  settings.metadataTemplate = request.metadataTemplate;

  settings.outputFormat = request.outputFormat.empty() ? "wav" : request.outputFormat;
  settings.lossyBitrateKbps = std::clamp(request.lossyBitrateKbps, 64, 320);
  settings.lossyQuality =
      std::clamp(static_cast<int>(std::lround((static_cast<double>(settings.lossyBitrateKbps) - 64.0) / 25.6)), 0, 10);
  settings.mp3UseVbr = toLower(settings.outputFormat) == "mp3" && request.mp3UseVbr;
  settings.mp3VbrQuality = std::clamp(request.mp3VbrQuality, 0, 9);

  if (settings.exportSpeedMode == kExportSpeedModeQuick) {
    const auto availability = util::WavWriter::getAvailableFormats();
    const auto hasAvailableMp3 = std::find_if(availability.begin(), availability.end(), [](const auto& entry) {
      return toLower(entry.format) == "mp3" && entry.available;
    }) != availability.end();

    if (hasAvailableMp3) {
      settings.outputFormat = "mp3";
      settings.lossyBitrateKbps = 320;
      settings.lossyQuality = 8;
      settings.mp3UseVbr = true;
      settings.mp3VbrQuality = 0;
    }
  }

  switch (request.gpuProviderSelectionId) {
    case 2:
      settings.gpuExecutionProvider = "cpu";
      break;
    case 3:
      settings.gpuExecutionProvider = "directml";
      break;
    case 4:
      settings.gpuExecutionProvider = "coreml";
      break;
    case 5:
      settings.gpuExecutionProvider = "cuda";
      break;
    default:
      settings.gpuExecutionProvider = "auto";
      break;
  }

  if (!request.outputPath.empty()) {
    std::filesystem::path normalizedPath(request.outputPath);
    const auto requiredExtension = extensionForFormat(settings.outputFormat);
    if (toLower(normalizedPath.extension().string()) != requiredExtension) {
      normalizedPath.replace_extension(requiredExtension);
    }
    settings.outputPath = normalizedPath.string();
  }

  settings.rendererName = request.selectedRendererId.empty() ? "BuiltIn" : request.selectedRendererId;
  for (const auto& info : request.rendererInfos) {
    if (info.id == settings.rendererName) {
      settings.externalRendererPath = info.binaryPath.string();
      break;
    }
  }

  return settings;
}

ExportPreflightResult ExportController::preflight(const ExportPreflightRequest& request) const {
  ExportPreflightResult result;
  if (const auto profile = domain::findProjectProfile(request.projectProfiles, request.projectProfileId);
      profile.has_value()) {
    if (!profile->pinnedRendererIds.empty()) {
      const bool pinned = std::find(profile->pinnedRendererIds.begin(),
                                    profile->pinnedRendererIds.end(),
                                    request.selectedRendererId) != profile->pinnedRendererIds.end();
      if (!pinned) {
        result.taskHistoryText = "Renderer " + juce::String(request.selectedRendererId) +
                                 " not pinned for profile " + profile->id;
        if (toLower(request.safetyPolicyId) == "strict") {
          result.allowed = false;
          result.statusText = "Export blocked by strict safety policy: renderer not pinned for selected profile";
        }
      }
    }
  }
  return result;
}

void ExportController::runExport(const domain::Session& session,
                                 const domain::RenderSettings& settings,
                                 const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
                                 std::atomic_bool& cancelFlag) {
  const bool quickExportMode = toLower(settings.exportSpeedMode) == "quick";

  struct RenderJob final : juce::ThreadPoolJob {
    domain::Session session;
    domain::RenderSettings settings;
    std::vector<analysis::StemAnalysisEntry> analysisEntries;
    std::atomic_bool* cancelFlag;
    bool quickExportMode;
    Callbacks callbacks;

    RenderJob(domain::Session sess, domain::RenderSettings sett,
              std::vector<analysis::StemAnalysisEntry> entries,
              std::atomic_bool* cancel, bool quick, Callbacks cb)
        : juce::ThreadPoolJob("ExportJob"),
          session(std::move(sess)),
          settings(std::move(sett)),
          analysisEntries(std::move(entries)),
          cancelFlag(cancel),
          quickExportMode(quick),
          callbacks(std::move(cb)) {}

    bool isCancellationRequested() const {
      return shouldExit() || (cancelFlag != nullptr && cancelFlag->load());
    }

    void requestCancellation() const {
      if (cancelFlag != nullptr) {
        cancelFlag->store(true);
      }
    }

    JobStatus runJob() override {
      renderers::RenderResult renderResult;
      juce::String crashMessage;
      juce::String healthText;
      bool healthHasCriticalIssues = false;
      size_t healthIssueCount = 0;

      try {
        if (isCancellationRequested()) {
          requestCancellation();
          renderResult.cancelled = true;
          renderResult.rendererName = settings.rendererName;
        }

        if (quickExportMode) {
          healthText = "Quick export mode: stem-health preflight skipped for faster turnaround.";
        } else {
          if (isCancellationRequested()) {
            requestCancellation();
            renderResult.cancelled = true;
            renderResult.rendererName = settings.rendererName;
          }

          if (analysisEntries.empty()) {
            if (callbacks.onStatus) {
              callbacks.onStatus("Export: analyzing stems");
            }

            analysis::StemAnalyzer analyzer;
            analysisEntries = analyzer.analyzeSession(session);
          }
          if (isCancellationRequested()) {
            requestCancellation();
            renderResult.cancelled = true;
            renderResult.rendererName = settings.rendererName;
          }
          const auto healthCacheKey = buildHealthCacheKey(session, analysisEntries);
          bool healthCacheHit = false;
          {
            std::scoped_lock lock(exportHealthCacheMutex());
            const auto& cached = exportHealthCache();
            if (cached.has_value() && cached->key == healthCacheKey) {
              healthText = cached->text;
              healthHasCriticalIssues = cached->hasCriticalIssues;
              healthIssueCount = cached->issueCount;
              healthCacheHit = true;
            }
          }

          if (!healthCacheHit) {
            analysis::StemHealthAssistant healthAssistant;
            const auto healthReport = healthAssistant.analyze(session, analysisEntries);
            healthText = juce::String(healthAssistant.toText(healthReport));
            healthHasCriticalIssues = healthReport.hasCriticalIssues;
            healthIssueCount = healthReport.issues.size();

            std::scoped_lock lock(exportHealthCacheMutex());
            exportHealthCache() = ExportHealthCacheEntry{
                .key = healthCacheKey,
                .text = healthText,
                .hasCriticalIssues = healthHasCriticalIssues,
                .issueCount = healthIssueCount,
            };
          }
        }

        if (!renderResult.cancelled) {
          auto renderer = renderers::createRenderer(settings.rendererName);

          std::mutex progressMutex;
          auto lastProgressEmit = std::chrono::steady_clock::time_point {};
          double lastProgressFraction = -1.0;
          std::string lastProgressStage;
          auto capturedCallbacks = callbacks;

          renderResult = renderer->render(
              session,
              settings,
              [this, capturedCallbacks, &progressMutex, &lastProgressEmit, &lastProgressFraction, &lastProgressStage](
                  const double progress, const std::string& stage) {
                if (shouldExit()) {
                  requestCancellation();
                }
                bool emit = false;
                {
                  std::scoped_lock lock(progressMutex);
                  const auto now = std::chrono::steady_clock::now();
                  const bool stageChanged = stage != lastProgressStage;
                  const bool finalProgress = progress >= 0.999;
                  const bool timeGateOpen =
                      lastProgressEmit.time_since_epoch().count() == 0 ||
                      now - lastProgressEmit >= std::chrono::milliseconds(180);
                  const bool deltaGateOpen = std::abs(progress - lastProgressFraction) >= 0.02;
                  emit = stageChanged || finalProgress || (timeGateOpen && deltaGateOpen);
                  if (emit) {
                    lastProgressEmit = now;
                    lastProgressFraction = progress;
                    lastProgressStage = stage;
                  }
                }
                if (!emit) {
                  return;
                }

                if (stage == "Mix render cache hit") {
                  if (capturedCallbacks.onStatus) {
                    capturedCallbacks.onStatus("Export: Using cached mix render (fast path)");
                  }
                  if (capturedCallbacks.onTaskHistory) {
                    capturedCallbacks.onTaskHistory("Export using cached mix render");
                  }
                  return;
                }
                if (capturedCallbacks.onStatus) {
                  capturedCallbacks.onStatus("Export: " + stage + " (" +
                                             std::to_string(static_cast<int>(progress * 100.0)) + "%)");
                }
                if (progress >= 0.999 || stage != "Summing stem buses") {
                  if (capturedCallbacks.onTaskHistory) {
                    capturedCallbacks.onTaskHistory("Export " + stage + " " +
                                                     std::to_string(static_cast<int>(progress * 100.0)) + "%");
                  }
                }
              },
              cancelFlag);
        }
      } catch (const std::exception& error) {
        crashMessage = "Export exception:\n" + juce::String(error.what());
      } catch (...) {
        crashMessage = "Export exception:\nUnknown error";
      }

      ExportResult result;
      result.success = renderResult.success;
      result.cancelled = renderResult.cancelled || isCancellationRequested();
      result.rendererName = renderResult.rendererName;
      result.outputAudioPath = renderResult.outputAudioPath;
      result.reportPath = renderResult.reportPath;
      result.exportSpeedMode = settings.exportSpeedMode;
      result.logs = std::move(renderResult.logs);
      result.analysisEntries = std::move(analysisEntries);
      result.healthText = healthText;
      result.healthHasCriticalIssues = healthHasCriticalIssues;
      result.healthIssueCount = healthIssueCount;
      result.crashMessage = crashMessage;

      auto capturedCallbacks = callbacks;
      util::dispatchCallback([capturedCallbacks, result = std::move(result)]() mutable {
        if (capturedCallbacks.onExportComplete) {
          capturedCallbacks.onExportComplete(std::move(result));
        }
      });

      return jobHasFinished;
    }
  };

  threadPool_.addJob(
      new RenderJob(session, settings, analysisEntries, &cancelFlag, quickExportMode, callbacks_),
      true);
}

void ExportController::validateExternalRenderer(std::string selectedPath, std::string selectedName) {
  if (selectedPath.empty()) {
    return;
  }
  if (callbacks_.onStatus) {
    callbacks_.onStatus("Validating external renderer...");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("External renderer validation started: " + selectedName);
  }

  threadPool_.addJob(new util::BackgroundJob(
                         [callbacks = callbacks_,
                          selectedPath = std::move(selectedPath),
                          selectedName = std::move(selectedName)]() mutable {
                           const auto validation = renderers::ExternalLimiterRenderer::validateBinary(selectedPath);
                           ExternalRendererValidationResult result;
                           result.selectedPath = selectedPath;
                           result.selectedName = selectedName;
                           result.valid = validation.valid;
                           result.diagnostics = validation.diagnostics;

                           util::dispatchCallback([callbacks, result = std::move(result)]() mutable {
                             if (callbacks.onExternalRendererValidated) {
                               callbacks.onExternalRendererValidated(std::move(result));
                             }
                           });
                         }),
                     true);
}

void ExportController::prefetchLame() {
  if (callbacks_.onStatus) {
    callbacks_.onStatus("Prefetching LAME...");
  }

  threadPool_.addJob(new util::BackgroundJob([callbacks = callbacks_]() mutable {
    const auto download = util::LameDownloader::ensureAvailable();
    LamePrefetchResult result;
    result.success = download.success;
    result.executablePath = download.executablePath.string();
    result.detail = download.detail;

    util::dispatchCallback([callbacks, result = std::move(result)]() mutable {
      if (callbacks.onLamePrefetchComplete) {
        callbacks.onLamePrefetchComplete(std::move(result));
      }
    });
  }), true);
}

} // namespace automix::app
