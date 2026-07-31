#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include "ai/ModelManager.h"
#include "domain/RenderSettings.h"
#include "domain/Session.h"

namespace automix::app {

class TaskOrchestrator;

/// Standalone verification jobs for export and batch output.
/// Extracted from MainLayout to reduce its size; used via thin delegation.
class VerificationEngine {
 public:
  using HistorySink = std::function<void(const juce::String&)>;

  struct ExportContext {
    domain::Session session;
    domain::RenderSettings settings;
    std::string outputAudioPath;
    std::optional<ai::ModelPack> analysisPack;
  };

  struct BatchContext {
    std::filesystem::path inputFolder;
    std::filesystem::path outputFolder;
    domain::RenderSettings settings;
    std::optional<ai::ModelPack> analysisPack;
    bool recursiveScan = false;
  };

  /// Run asynchronous export verification on a background thread.
  static void runExportVerification(ExportContext context,
                                    juce::ThreadPool& backgroundPool,
                                    HistorySink onHistory);

  /// Run asynchronous batch verification on a background thread.
  static void runBatchVerification(BatchContext context,
                                   juce::ThreadPool& backgroundPool,
                                   HistorySink onHistory);
};

} // namespace automix::app
