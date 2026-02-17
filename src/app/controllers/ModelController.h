#pragma once

#include <atomic>
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

enum class ModelMenuAction {
  BrowseAndDownload = 1,
  ShowInstalled = 2,
  CheckUpdates = 3,
  VerifyIntegrity = 4,
  OpenHubFolder = 5,
};

class ModelController {
 public:
  struct Callbacks {
    std::function<void(const std::string&)> onStatus;
    std::function<void(const std::string&)> onTaskHistory;
    std::function<void(const std::string&)> onReport;
    std::function<void()> onModelPacksChanged;
    std::function<void()> onCatalogReady;
    std::function<void(const std::filesystem::path&)> onRevealModelHubFolder;
    std::function<void()> onAsyncTaskComplete;
  };

  ModelController(ai::ModelManager& modelManager,
                  juce::ThreadPool& threadPool,
                  Callbacks callbacks);

  void fetchCatalog(std::atomic_bool& cancelFlag);
  void installModel(const std::string& repoId, std::atomic_bool& cancelFlag);
  void showInstalled();
  void checkUpdates(std::atomic_bool& cancelFlag);
  void verifyIntegrity();
  void dispatchMenuAction(ModelMenuAction action, std::atomic_bool* cancelFlag = nullptr);

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
