#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "ai/HuggingFaceModelHub.h"
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

  // Callbacks
  std::function<void()> onFetchCatalog;
  std::function<void(const std::string& repoId)> onInstallModel;
  std::function<void()> onCheckUpdates;
  std::function<void()> onVerifyIntegrity;

 private:
  class ModelRow final : public juce::Component {
   public:
    ModelRow();
    void paint(juce::Graphics& g) override;
    void resized() override;
    void setModel(const ai::HubModelInfo& model);

    std::function<void(const std::string& repoId)> onInstall;

   private:
    ai::HubModelInfo model_;
    juce::Label repoLabel_;
    juce::Label detailLabel_;
    juce::TextButton installButton_{"Install"};
  };

  void rebuildModelRows();

  juce::Label statusLabel_;
  juce::TextButton fetchButton_{"Fetch Catalog"};
  juce::TextButton updatesButton_{"Check Updates"};
  juce::TextButton verifyButton_{"Verify"};

  juce::Viewport viewport_;
  juce::Component rowContainer_;
  std::vector<std::unique_ptr<ModelRow>> modelRows_;
  std::vector<ai::HubModelInfo> models_;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModelBrowserPanel)
};

} // namespace automix::app
