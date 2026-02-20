#include "app/controllers/ExportController.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <sstream>

#include "analysis/StemHealthAssistant.h"
#include "renderers/RendererFactory.h"
#include "util/StringUtils.h"

namespace automix::app {
namespace {

using ::automix::util::toLower;

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

double clampProgress(const double progress) {
  return std::clamp(progress, 0.0, 1.0);
}

void emitProgress(const ExportController::Callbacks& callbacks, const double progress) {
  if (callbacks.onProgress) {
    callbacks.onProgress(clampProgress(progress));
  }
}

} // namespace

void ExportController::clearHealthCache() {
  std::scoped_lock lock(exportHealthCacheMutex());
  exportHealthCache().reset();
}

ExportController::ExportController(juce::ThreadPool& threadPool, Callbacks callbacks)
    : threadPool_(threadPool), callbacks_(std::move(callbacks)) {}

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

    JobStatus runJob() override {
      renderers::RenderResult renderResult;
      juce::String crashMessage;
      juce::String healthText;
      bool healthHasCriticalIssues = false;
      size_t healthIssueCount = 0;

      try {
        emitProgress(callbacks, 0.02);
        if (quickExportMode) {
          healthText = "Quick export mode: stem-health preflight skipped for faster turnaround.";
          emitProgress(callbacks, 0.28);
        } else {
          if (analysisEntries.empty()) {
            if (callbacks.onStatus) {
              callbacks.onStatus("Export: analyzing stems");
            }
            emitProgress(callbacks, 0.08);

            analysis::StemAnalyzer analyzer;
            analysisEntries = analyzer.analyzeSession(session);
            emitProgress(callbacks, 0.22);
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
          emitProgress(callbacks, 0.32);
        }

        auto renderer = renderers::createRenderer(settings.rendererName);

        std::mutex progressMutex;
        auto lastProgressEmit = std::chrono::steady_clock::time_point {};
        double lastProgressFraction = -1.0;
        std::string lastProgressStage;
        auto capturedCallbacks = callbacks;

        renderResult = renderer->render(
            session,
            settings,
            [capturedCallbacks, &progressMutex, &lastProgressEmit, &lastProgressFraction, &lastProgressStage](
                const double progress, const std::string& stage) {
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
              emitProgress(capturedCallbacks, 0.35 + (0.63 * progress));
              if (progress >= 0.999 || stage != "Summing stem buses") {
                if (capturedCallbacks.onTaskHistory) {
                  capturedCallbacks.onTaskHistory("Export " + stage + " " +
                                                   std::to_string(static_cast<int>(progress * 100.0)) + "%");
                }
              }
            },
            cancelFlag);
      } catch (const std::exception& error) {
        crashMessage = "Export exception:\n" + juce::String(error.what());
      } catch (...) {
        crashMessage = "Export exception:\nUnknown error";
      }

      ExportResult result;
      result.success = renderResult.success;
      result.cancelled = renderResult.cancelled;
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
      juce::MessageManager::callAsync([capturedCallbacks, result = std::move(result)]() mutable {
        emitProgress(capturedCallbacks, 1.0);
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

} // namespace automix::app
