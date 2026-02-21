#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include "domain/Session.h"

namespace automix::app {

struct ImportResult {
  bool cancelled = false;
  std::vector<domain::Stem> stems;
  std::vector<std::string> logLines;
  std::optional<std::string> originalMixPath;
};

class ImportController {
 public:
  struct Callbacks {
    std::function<void(const std::string&)> onStatus;
    std::function<void(const std::string&)> onTaskHistory;
    std::function<void(double)> onProgress;
    std::function<void(ImportResult)> onImportComplete;
  };

  ImportController(juce::ThreadPool& threadPool, Callbacks callbacks);

  void importFiles(std::vector<juce::File> files,
                   bool useSeparation,
                   int preferredStemCount,
                   std::atomic_bool& cancelFlag,
                   std::optional<std::filesystem::path> separationModelRoot = std::nullopt);

 private:
  juce::ThreadPool& threadPool_;
  Callbacks callbacks_;
};

} // namespace automix::app
