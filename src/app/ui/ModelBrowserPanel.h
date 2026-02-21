#pragma once

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "ai/HuggingFaceModelHub.h"
#include "ai/ModelPackLoader.h"
#include "app/style/Theme.h"

namespace automix::app {

/// Panel for browsing, installing, and managing AI model packs.
/// Shown in a DialogWindow from MainLayout.
class ModelBrowserPanel final : public juce::Component {
 public:
  ModelBrowserPanel();

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setDiscoveredModels(const std::vector<ai::HubModelInfo>& models);
  void setStatus(const juce::String& status);
  void setActionsEnabled(bool enabled);
  void setActivePackDisplay(std::map<std::string, std::string> activePackByTask);
  void setInstalledPacks(std::vector<ai::ModelPack> installedPacks);
  void setInstalledModelIds(std::set<std::string> installedModelIds);
  bool isCuratedMode() const;
  bool isCuratedLockEnabled() const;
  std::string rawSearchQuery() const;
  std::string selectedTaskScope() const;

  // Callbacks
  std::function<void()> onFetchCatalog;
  std::function<void(const std::string& modelId)> onInstallModel;
  std::function<void(const std::string& modelId)> onUninstallModel;
  std::function<void(const std::string& modelId)> onUseInstalledModel;
  std::function<void(const std::string& taskScope, const std::string& packId)> onSetActivePack;
  std::function<void()> onCheckUpdates;
  std::function<void()> onVerifyIntegrity;

 private:
  class ModelRow final : public juce::Component {
   public:
    ModelRow();
    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseUp(const juce::MouseEvent& event) override;
    void setModel(const ai::HubModelInfo& model, bool installed);

    std::function<void(const std::string& modelId)> onInstall;
    std::function<void(const std::string& modelId)> onUninstall;
    std::function<void(const std::string& modelId)> onInspect;

   private:
    ai::HubModelInfo model_;
    bool installed_ = false;
    juce::Label repoLabel_;
    juce::Label detailLabel_;
    juce::TextButton installButton_{"Install"};
  };

  void rebuildModelRows();
  void updateTaskScopeUiState();
  void updateActivePackLabel();
  void updateInstalledPackChoices();
  void updateSetActivePackButtonState();
  void updateUseSelectedButtonState();
  void requestConfirmation(juce::MessageBoxIconType iconType,
                           juce::String title,
                           juce::String message,
                           juce::String confirmButtonText,
                           std::function<void()> onConfirm);
  void updateCatalogWarning();
  void updateCapabilityReport(const ai::HubModelInfo* model);
  std::vector<ai::HubModelInfo> visibleModels() const;
  static std::string normalizeTaskScope(std::string scope);

  juce::Label statusLabel_;
  juce::Label catalogModeLabel_{"", "Catalog"};
  juce::ComboBox catalogModeBox_;
  juce::ToggleButton curatedLockToggle_{"Lock curated set (stable)"};
  juce::Label taskScopeLabel_{"", "Task"};
  juce::ComboBox taskScopeBox_;
  juce::Label activePackTitleLabel_{"", "Active"};
  juce::Label activePackValueLabel_;
  juce::Label installedPackLabel_{"", "Installed"};
  juce::ComboBox installedPackBox_;
  juce::TextButton setActivePackButton_{"Set Active"};
  juce::Label catalogWarningLabel_;
  juce::TextEditor searchEditor_;
  juce::TextButton fetchButton_{"Fetch Catalog"};
  juce::TextButton installRecommendedButton_{"Install Recommended"};
  juce::TextButton useSelectedButton_{"Use Selected for Task"};
  juce::TextButton updatesButton_{"Check Updates"};
  juce::TextButton verifyButton_{"Verify"};

  juce::Viewport viewport_;
  juce::Component rowContainer_;
  juce::TextEditor capabilityReport_;
  std::vector<std::unique_ptr<ModelRow>> modelRows_;
  std::vector<ai::HubModelInfo> models_;
  std::vector<ai::ModelPack> installedPacks_;
  std::map<int, std::string> installedPackIdByComboId_;
  std::set<std::string> installedModelIds_;
  std::map<std::string, std::string> activePackByTask_;
  std::string selectedModelId_;
  bool actionsEnabled_ = true;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModelBrowserPanel)
};

} // namespace automix::app
