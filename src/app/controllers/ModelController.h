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

class ModelController {
 public:
  struct ModelHubOps {
    std::function<std::vector<ai::HubModelInfo>(const ai::HubModelQueryOptions&)> discoverRecommended;
    std::function<ai::HubInstallResult(const std::string&, const ai::HubInstallOptions&)> installModel;
    std::function<std::optional<ai::HubModelInfo>(const std::string&)> modelInfo;
  };

  struct Callbacks {
    std::function<void(const std::string&)> onStatus;
    std::function<void(const std::string&)> onTaskHistory;
    std::function<void(double)> onProgress;
    std::function<void(const std::string&)> onReport;
    std::function<void()> onModelPacksChanged;
    std::function<void(bool)> onCatalogReady;
    std::function<void(bool)> onInstallComplete;
    std::function<void(bool)> onUpdateCheckComplete;
  };

  static ModelHubOps createDefaultHubOps();

  ModelController(ai::ModelManager& modelManager,
                  juce::ThreadPool& threadPool,
                  Callbacks callbacks,
                  ModelHubOps hubOps);

  void fetchCatalog(std::atomic_bool& cancelFlag);
  void installModel(const std::string& repoId, std::atomic_bool& cancelFlag);
  void showInstalled();
  void checkUpdates(std::atomic_bool& cancelFlag);
  void verifyIntegrity();

  const std::vector<ai::HubModelInfo>& discoveredModels() const;
  void setModelHubRoot(const std::filesystem::path& root);

 private:
  ai::ModelManager& modelManager_;
  juce::ThreadPool& threadPool_;
  Callbacks callbacks_;
  ModelHubOps hubOps_;
  std::filesystem::path modelHubRoot_;
  std::vector<ai::HubModelInfo> discoveredModels_;
};

} // namespace automix::app
