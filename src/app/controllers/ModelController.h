#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include "ai/HuggingFaceModelHub.h"
#include "ai/ModelManager.h"

namespace automix::app {

class ModelController {
 public:
  struct Callbacks {
    std::function<void(const std::string&)> onStatus;
    std::function<void(const std::string&)> onTaskHistory;
    std::function<void(const std::string&)> onReport;
    std::function<void()> onModelPacksChanged;
    std::function<void()> onCatalogReady;
  };

  ModelController(ai::ModelManager& modelManager,
                  juce::ThreadPool& threadPool,
                  Callbacks callbacks);

  void fetchCatalog();
  void installModel(const std::string& repoId);
  void showInstalled();
  void checkUpdates();
  void verifyIntegrity();

  const std::vector<ai::HubModelInfo>& discoveredModels() const;
  void setModelHubRoot(const std::filesystem::path& root);

 private:
  ai::ModelManager& modelManager_;
  juce::ThreadPool& threadPool_;
  Callbacks callbacks_;
  std::filesystem::path modelHubRoot_;
  std::vector<ai::HubModelInfo> discoveredModels_;
};

} // namespace automix::app
