#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include "analysis/StemAnalyzer.h"
#include "domain/Session.h"
#include "renderers/RendererRegistry.h"

namespace automix::app {

struct ExportResult {
  bool success = false;
  bool cancelled = false;
  std::string rendererName;
  std::string outputAudioPath;
  std::string reportPath;
  std::string exportSpeedMode;
  std::vector<std::string> logs;
  std::vector<analysis::StemAnalysisEntry> analysisEntries;
  juce::String healthText;
  bool healthHasCriticalIssues = false;
  size_t healthIssueCount = 0;
  juce::String crashMessage;
};

class ExportController {
 public:
  struct Callbacks {
    std::function<void(const std::string&)> onStatus;
    std::function<void(const std::string&)> onTaskHistory;
    std::function<void(double)> onProgress;
    std::function<void(ExportResult)> onExportComplete;
  };

  ExportController(juce::ThreadPool& threadPool, Callbacks callbacks);

  void runExport(const domain::Session& session,
                 const domain::RenderSettings& settings,
                 const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
                 std::atomic_bool& cancelFlag);

  static void clearHealthCache();

 private:
  juce::ThreadPool& threadPool_;
  Callbacks callbacks_;
};

} // namespace automix::app
