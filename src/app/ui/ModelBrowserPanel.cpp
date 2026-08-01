#include "app/ui/ModelBrowserPanel.h"
#include "app/controllers/ModelController.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

namespace automix::app {
namespace {

std::string toLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](const char value) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
  });
  return text;
}

std::string normalizedTaskScope(std::string scope) {
  scope = toLower(std::move(scope));
  if (scope == "mix" || scope == "master" || scope == "analysis" || scope == "separation") {
    return scope;
  }
  return "analysis";
}

std::vector<std::string> expectedOutputKeysForTask(const std::string& taskScope) {
  if (taskScope == "mix") {
    return {"confidence", "global_gain_db", "global_pan_bias"};
  }
  if (taskScope == "master") {
    return {"confidence", "target_lufs", "pre_gain_db", "limiter_ceiling_db", "glue_ratio"};
  }
  if (taskScope == "separation") {
    return {"confidence", "separation_quality"};
  }
  return {"confidence"};
}

std::string inferEngine(const ai::HubModelInfo& model) {
  const auto ext = toLower(std::filesystem::path(model.primaryFile).extension().string());
  if (ext == ".onnx") {
    return "onnxruntime";
  }
  return "unknown";
}

std::string modelKey(const ai::HubModelInfo& model) {
  return model.modelId.empty() ? model.repoId : model.modelId;
}

} // namespace

using namespace theme;

// ── ModelRow ───────────────────────────────────────────────────────

ModelBrowserPanel::ModelRow::ModelRow() {
  repoLabel_.setFont(typography::body());
  repoLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  repoLabel_.setJustificationType(juce::Justification::centredLeft);

  detailLabel_.setFont(typography::caption());
  detailLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  detailLabel_.setJustificationType(juce::Justification::centredLeft);

  installButton_.onClick = [this] {
    const auto modelId = model_.modelId.empty() ? model_.repoId : model_.modelId;
    if (installed_) {
      if (onUninstall) {
        onUninstall(modelId);
      }
      return;
    }
    if (onInstall) {
      onInstall(modelId);
    }
  };

  addAndMakeVisible(repoLabel_);
  addAndMakeVisible(detailLabel_);
  addAndMakeVisible(installButton_);
}

void ModelBrowserPanel::ModelRow::paint(juce::Graphics& g) {
  g.setColour(colour(colours::surfaceBorder));
  g.drawHorizontalLine(getHeight() - 1, 0.0f, static_cast<float>(getWidth()));
}

void ModelBrowserPanel::ModelRow::resized() {
  auto area = getLocalBounds().reduced(4);
  installButton_.setBounds(area.removeFromRight(80).reduced(2));
  auto top = area.removeFromTop(area.getHeight() / 2);
  repoLabel_.setBounds(top);
  detailLabel_.setBounds(area);
}

void ModelBrowserPanel::ModelRow::mouseUp(const juce::MouseEvent& event) {
  juce::ignoreUnused(event);
  if (onInspect) {
    onInspect(model_.modelId.empty() ? model_.repoId : model_.modelId);
  }
}

void ModelBrowserPanel::ModelRow::setModel(const ai::HubModelInfo& model, const bool installed) {
  model_ = model;
  installed_ = installed;
  const auto title = model.displayName.empty() ? model.repoId : model.displayName;
  repoLabel_.setText(juce::String(title), juce::dontSendNotification);

  juce::String detail = "source: " + juce::String(model.source);
  detail += "  scope: " + juce::String(model.taskScope.empty() ? "analysis" : model.taskScope);
  if (!model.license.empty())
    detail += "  license: " + juce::String(model.license);
  if (!model.revision.empty())
    detail += "  rev: " + juce::String(model.revision);
  if (model.curated)
    detail += "  curated";
  if (installed_) {
    detail += "  installed";
  }
  if (!model.compatible)
    detail += "  incompatible";
  if (!model.compatible && !model.compatibilityReport.empty())
    detail += " (" + juce::String(model.compatibilityReport) + ")";
  detailLabel_.setText(detail, juce::dontSendNotification);

  installButton_.setEnabled(model.compatible);
  if (!model.compatible) {
    installButton_.setButtonText("Blocked");
  } else if (installed_) {
    installButton_.setButtonText("Uninstall");
  } else {
    installButton_.setButtonText("Install");
  }
}

// ── ModelBrowserPanel ──────────────────────────────────────────────

ModelBrowserPanel::ModelBrowserPanel() {
  statusLabel_.setText("Click 'Fetch Catalog' to browse available models", juce::dontSendNotification);
  statusLabel_.setFont(typography::body());
  statusLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  statusLabel_.setJustificationType(juce::Justification::centredLeft);

  catalogModeLabel_.setFont(typography::caption());
  catalogModeLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  catalogModeLabel_.setJustificationType(juce::Justification::centredLeft);
  activePackTitleLabel_.setFont(typography::caption());
  activePackTitleLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  activePackTitleLabel_.setJustificationType(juce::Justification::centredLeft);
  activePackValueLabel_.setFont(typography::caption());
  activePackValueLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  activePackValueLabel_.setJustificationType(juce::Justification::centredLeft);
  taskScopeLabel_.setFont(typography::caption());
  taskScopeLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  taskScopeLabel_.setJustificationType(juce::Justification::centredLeft);
  installedPackLabel_.setFont(typography::caption());
  installedPackLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  installedPackLabel_.setJustificationType(juce::Justification::centredLeft);
  catalogWarningLabel_.setFont(typography::caption());
  catalogWarningLabel_.setColour(juce::Label::textColourId, colour(colours::warning));
  catalogWarningLabel_.setJustificationType(juce::Justification::centredLeft);
  catalogWarningLabel_.setVisible(false);

  catalogModeBox_.addItem("Curated (Recommended)", 1);
  catalogModeBox_.addItem("Raw Search", 2);
  catalogModeBox_.setSelectedId(1, juce::dontSendNotification);
  catalogModeBox_.onChange = [this] {
    if (catalogModeBox_.getSelectedId() == 2 && curatedLockToggle_.getToggleState()) {
      curatedLockToggle_.setToggleState(false, juce::dontSendNotification);
    }
    updateTaskScopeUiState();
    rebuildModelRows();
  };
  curatedLockToggle_.setToggleState(true, juce::dontSendNotification);
  curatedLockToggle_.onClick = [this] {
    if (curatedLockToggle_.getToggleState()) {
      catalogModeBox_.setSelectedId(1, juce::dontSendNotification);
    }
    updateTaskScopeUiState();
    rebuildModelRows();
  };

  taskScopeBox_.addItem("Mix", 1);
  taskScopeBox_.addItem("Master", 2);
  taskScopeBox_.addItem("Analysis", 3);
  taskScopeBox_.addItem("Separation", 4);
  taskScopeBox_.setSelectedId(2, juce::dontSendNotification);
  taskScopeBox_.onChange = [this] {
    updateActivePackLabel();
    updateInstalledPackChoices();
    rebuildModelRows();
  };
  installedPackBox_.onChange = [this] {
    updateSetActivePackButtonState();
  };
  setActivePackButton_.onClick = [this] {
    const auto selectedId = installedPackBox_.getSelectedId();
    const auto selectedIt = installedPackIdByComboId_.find(selectedId);
    if (selectedIt == installedPackIdByComboId_.end()) {
      setStatus("No installed pack selected");
      return;
    }
    if (onSetActivePack) {
      onSetActivePack(selectedTaskScope(), selectedIt->second);
    }
  };

  searchEditor_.setTextToShowWhenEmpty("Search text for raw mode (e.g., demucs onnx)", colour(colours::textMuted));
  searchEditor_.setMultiLine(false);
  updateTaskScopeUiState();

  fetchButton_.onClick = [this] {
    if (onFetchCatalog)
      onFetchCatalog();
  };
  installRecommendedButton_.onClick = [this] {
    const auto visible = visibleModels();
    if (visible.empty()) {
      setStatus("No compatible models available for selected task");
      return;
    }

    const auto pickModel = [&]() -> std::optional<ai::HubModelInfo> {
      for (const auto& model : visible) {
        const auto id = modelKey(model);
        const bool installed = installedModelIds_.contains(id);
        if (model.recommended && model.compatible && !installed && (!isCuratedLockEnabled() || model.curated)) {
          return model;
        }
      }
      for (const auto& model : visible) {
        const auto id = modelKey(model);
        const bool installed = installedModelIds_.contains(id);
        if (model.compatible && !installed && (!isCuratedLockEnabled() || model.curated)) {
          return model;
        }
      }
      return std::nullopt;
    };

    const auto selected = pickModel();
    if (!selected.has_value()) {
      setStatus("No installable model found for selected task (already installed or incompatible)");
      return;
    }

    if (onInstallModel) {
      const auto modelId = modelKey(selected.value());
      const auto title = selected->displayName.empty() ? selected->repoId : selected->displayName;
      const bool requiresConsent = ModelController::modelRequiresLicenseConsent(selected->repoId);

      const juce::String dialogTitle = requiresConsent ? "License Consent Required (CC BY-NC 4.0)" : "Install Model";
      const juce::String dialogMsg = requiresConsent
          ? "Model '" + juce::String(title) + "' is subject to CC BY-NC 4.0 (Non-Commercial Use Only).\n\n"
            "By installing, you acknowledge and agree that this model will be used for non-commercial purposes only."
          : "Install model '" + juce::String(title) + "'?";
      const juce::String btnText = requiresConsent ? "I Agree & Install" : "Install";

      requestConfirmation(
          juce::AlertWindow::QuestionIcon,
          dialogTitle,
          dialogMsg,
          btnText,
          [this, modelId]() {
            if (onInstallModel) {
              onInstallModel(modelId);
            }
          });
    }
  };
  updatesButton_.onClick = [this] {
    if (onCheckUpdates)
      onCheckUpdates();
  };
  useSelectedButton_.onClick = [this] {
    if (selectedModelId_.empty()) {
      setStatus("Select an installed model first");
      return;
    }
    if (!installedModelIds_.contains(selectedModelId_)) {
      setStatus("Selected model is not installed");
      return;
    }
    if (onUseInstalledModel) {
      const auto selectedIt = std::find_if(models_.begin(), models_.end(), [&](const ai::HubModelInfo& model) {
        return modelKey(model) == selectedModelId_;
      });
      const auto title = selectedIt != models_.end() && !selectedIt->displayName.empty() ? selectedIt->displayName
                                                                                           : selectedModelId_;
      const auto scope = selectedTaskScope();
      const auto selectedModelId = selectedModelId_;
      requestConfirmation(
          juce::AlertWindow::QuestionIcon,
          "Use Model",
          "Use '" + juce::String(title) + "' for task '" + juce::String(scope) + "'?",
          "Use Model",
          [this, selectedModelId]() {
            if (onUseInstalledModel) {
              onUseInstalledModel(selectedModelId);
            }
          });
    }
  };
  verifyButton_.onClick = [this] {
    if (onVerifyIntegrity)
      onVerifyIntegrity();
  };

  viewport_.setViewedComponent(&rowContainer_, false);
  viewport_.setScrollBarsShown(true, false);
  capabilityReport_.setReadOnly(true);
  capabilityReport_.setMultiLine(true);
  capabilityReport_.setScrollbarsShown(true);
  capabilityReport_.setText("Select a model to view capability details.", juce::dontSendNotification);

  addAndMakeVisible(statusLabel_);
  addAndMakeVisible(catalogModeLabel_);
  addAndMakeVisible(catalogModeBox_);
  addAndMakeVisible(curatedLockToggle_);
  addAndMakeVisible(taskScopeLabel_);
  addAndMakeVisible(taskScopeBox_);
  addAndMakeVisible(activePackTitleLabel_);
  addAndMakeVisible(activePackValueLabel_);
  addAndMakeVisible(installedPackLabel_);
  addAndMakeVisible(installedPackBox_);
  addAndMakeVisible(setActivePackButton_);
  addAndMakeVisible(catalogWarningLabel_);
  addAndMakeVisible(searchEditor_);
  addAndMakeVisible(fetchButton_);
  addAndMakeVisible(installRecommendedButton_);
  addAndMakeVisible(useSelectedButton_);
  addAndMakeVisible(updatesButton_);
  addAndMakeVisible(verifyButton_);
  addAndMakeVisible(viewport_);
  addAndMakeVisible(capabilityReport_);
  updateActivePackLabel();
  updateInstalledPackChoices();
  updateUseSelectedButtonState();
}

void ModelBrowserPanel::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));
}

void ModelBrowserPanel::resized() {
  auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingMedium));

  auto statusRow = area.removeFromTop(28);
  statusLabel_.setBounds(statusRow);

  area.removeFromTop(4);
  auto modeRow = area.removeFromTop(30);
  catalogModeLabel_.setBounds(modeRow.removeFromLeft(60).reduced(1));
  catalogModeBox_.setBounds(modeRow.removeFromLeft(200).reduced(1));
  curatedLockToggle_.setBounds(modeRow.removeFromLeft(220).reduced(1));
  modeRow.removeFromLeft(4);
  searchEditor_.setBounds(modeRow.reduced(1));

  area.removeFromTop(4);
  auto taskRow = area.removeFromTop(28);
  taskScopeLabel_.setBounds(taskRow.removeFromLeft(60).reduced(1));
  taskScopeBox_.setBounds(taskRow.removeFromLeft(160).reduced(1));
  activePackTitleLabel_.setBounds(taskRow.removeFromLeft(50).reduced(1));
  activePackValueLabel_.setBounds(taskRow.reduced(1));

  area.removeFromTop(4);
  auto installedRow = area.removeFromTop(28);
  installedPackLabel_.setBounds(installedRow.removeFromLeft(60).reduced(1));
  installedPackBox_.setBounds(installedRow.removeFromLeft(320).reduced(1));
  setActivePackButton_.setBounds(installedRow.removeFromLeft(120).reduced(1));

  auto warningRow = area.removeFromTop(20);
  catalogWarningLabel_.setBounds(warningRow.reduced(1, 0));

  area.removeFromTop(4);
  auto buttonRow = area.removeFromTop(32);
  fetchButton_.setBounds(buttonRow.removeFromLeft(120).reduced(2));
  installRecommendedButton_.setBounds(buttonRow.removeFromLeft(170).reduced(2));
  useSelectedButton_.setBounds(buttonRow.removeFromLeft(160).reduced(2));
  updatesButton_.setBounds(buttonRow.removeFromLeft(120).reduced(2));
  verifyButton_.setBounds(buttonRow.removeFromLeft(80).reduced(2));

  area.removeFromTop(4);
  auto reportArea = area.removeFromBottom(116);
  capabilityReport_.setBounds(reportArea);
  area.removeFromBottom(4);
  viewport_.setBounds(area);

  // Resize existing rows to match viewport width
  const int availWidth = viewport_.getWidth() > 0 ? (viewport_.getWidth() - viewport_.getScrollBarThickness()) : 560;
  rowContainer_.setSize(availWidth, rowContainer_.getHeight());
  for (auto& row : modelRows_) {
    auto bounds = row->getBounds();
    row->setBounds(0, bounds.getY(), availWidth, bounds.getHeight());
  }
}

void ModelBrowserPanel::setDiscoveredModels(const std::vector<ai::HubModelInfo>& models) {
  models_ = models;
  rebuildModelRows();
}

void ModelBrowserPanel::setStatus(const juce::String& status) {
  statusLabel_.setText(status, juce::dontSendNotification);
}

void ModelBrowserPanel::setActionsEnabled(bool enabled) {
  actionsEnabled_ = enabled;
  catalogModeBox_.setEnabled(enabled);
  taskScopeBox_.setEnabled(enabled);
  installedPackBox_.setEnabled(enabled);
  curatedLockToggle_.setEnabled(enabled);
  searchEditor_.setEnabled(enabled && !isCuratedMode() && !isCuratedLockEnabled());
  fetchButton_.setEnabled(enabled);
  installRecommendedButton_.setEnabled(enabled);
  updateSetActivePackButtonState();
  useSelectedButton_.setEnabled(enabled && !selectedModelId_.empty() && installedModelIds_.contains(selectedModelId_));
  updatesButton_.setEnabled(enabled);
  verifyButton_.setEnabled(enabled);
  for (auto& row : modelRows_)
    row->setEnabled(enabled);
}

void ModelBrowserPanel::setActivePackDisplay(std::map<std::string, std::string> activePackByTask) {
  activePackByTask_ = std::move(activePackByTask);
  updateActivePackLabel();
  updateInstalledPackChoices();
  updateUseSelectedButtonState();
}

void ModelBrowserPanel::setInstalledPacks(std::vector<ai::ModelPack> installedPacks) {
  installedPacks_ = std::move(installedPacks);
  updateInstalledPackChoices();
}

void ModelBrowserPanel::setInstalledModelIds(std::set<std::string> installedModelIds) {
  installedModelIds_ = std::move(installedModelIds);
  rebuildModelRows();
  updateUseSelectedButtonState();
}

bool ModelBrowserPanel::isCuratedMode() const {
  return catalogModeBox_.getSelectedId() != 2;
}

bool ModelBrowserPanel::isCuratedLockEnabled() const {
  return curatedLockToggle_.getToggleState();
}

std::string ModelBrowserPanel::rawSearchQuery() const {
  return searchEditor_.getText().trim().toStdString();
}

std::string ModelBrowserPanel::selectedTaskScope() const {
  switch (taskScopeBox_.getSelectedId()) {
    case 1: return "mix";
    case 2: return "master";
    case 3: return "analysis";
    case 4: return "separation";
    default: return "analysis";
  }
}

void ModelBrowserPanel::rebuildModelRows() {
  modelRows_.clear();
  rowContainer_.removeAllChildren();

  constexpr int rowHeight = 52;
  const int availWidth = viewport_.getWidth() > 0 ? (viewport_.getWidth() - viewport_.getScrollBarThickness()) : 560;
  const int rowWidth = std::max(availWidth, 300);
  int y = 0;

  const auto visible = visibleModels();
  for (const auto& model : visible) {
    auto row = std::make_unique<ModelRow>();
    const auto id = modelKey(model);
    const bool installed = installedModelIds_.contains(id);
    row->setModel(model, installed);
    if (isCuratedLockEnabled() && !model.curated) {
      row->setEnabled(false);
    }
    row->onInstall = [this](const std::string& modelId) {
      if (onInstallModel) {
        auto selectedIt = std::find_if(models_.begin(), models_.end(), [&](const ai::HubModelInfo& model) {
          return modelKey(model) == modelId;
        });
        const auto title = selectedIt != models_.end() && !selectedIt->displayName.empty() ? selectedIt->displayName
                                                                                             : modelId;
        const auto repoId = selectedIt != models_.end() ? selectedIt->repoId : modelId;
        const bool requiresConsent = ModelController::modelRequiresLicenseConsent(repoId);

        const juce::String dialogTitle = requiresConsent ? "License Consent Required (CC BY-NC 4.0)" : "Install Model";
        const juce::String dialogMsg = requiresConsent
            ? "Model '" + juce::String(title) + "' is subject to CC BY-NC 4.0 (Non-Commercial Use Only).\n\n"
              "By installing, you acknowledge and agree that this model will be used for non-commercial purposes only."
            : "Install model '" + juce::String(title) + "'?";
        const juce::String btnText = requiresConsent ? "I Agree & Install" : "Install";

        requestConfirmation(
            juce::AlertWindow::QuestionIcon,
            dialogTitle,
            dialogMsg,
            btnText,
            [this, modelId]() {
              if (onInstallModel) {
                onInstallModel(modelId);
              }
            });
      }
    };
    row->onUninstall = [this](const std::string& modelId) {
      if (onUninstallModel) {
        auto selectedIt = std::find_if(models_.begin(), models_.end(), [&](const ai::HubModelInfo& model) {
          return modelKey(model) == modelId;
        });
        const auto title = selectedIt != models_.end() && !selectedIt->displayName.empty() ? selectedIt->displayName
                                                                                             : modelId;
        requestConfirmation(
            juce::AlertWindow::WarningIcon,
            "Uninstall Model",
            "Uninstall model '" + juce::String(title) + "'?",
            "Uninstall",
            [this, modelId]() {
              if (onUninstallModel) {
                onUninstallModel(modelId);
              }
            });
      }
    };
    row->onInspect = [this](const std::string& modelId) {
      selectedModelId_ = modelId;
      auto it = std::find_if(models_.begin(), models_.end(), [&](const ai::HubModelInfo& model) {
        return modelKey(model) == modelId;
      });
      updateCapabilityReport(it != models_.end() ? &(*it) : nullptr);
      updateUseSelectedButtonState();
    };
    row->setBounds(0, y, std::max(rowWidth, 300), rowHeight);
    rowContainer_.addAndMakeVisible(row.get());
    modelRows_.push_back(std::move(row));
    y += rowHeight;
  }

  rowContainer_.setSize(std::max(rowWidth, 300), y);

  const auto selectedIt = std::find_if(visible.begin(), visible.end(), [&](const ai::HubModelInfo& model) {
    return modelKey(model) == selectedModelId_;
  });
  if (selectedIt != visible.end()) {
    updateCapabilityReport(&(*selectedIt));
  } else if (!visible.empty()) {
    selectedModelId_ = modelKey(visible.front());
    updateCapabilityReport(&visible.front());
  } else {
    selectedModelId_.clear();
    updateCapabilityReport(nullptr);
  }

  const auto task = selectedTaskScope();
  if (visible.empty()) {
    if (models_.empty()) {
      statusLabel_.setText("No models found. Click 'Fetch Catalog'.", juce::dontSendNotification);
    } else {
      statusLabel_.setText("0 models for task '" + juce::String(task) + "' (catalog total: " +
                               juce::String(static_cast<int>(models_.size())) + ")",
                           juce::dontSendNotification);
    }
  } else {
    statusLabel_.setText(juce::String(static_cast<int>(visible.size())) + " models found for task '" +
                             juce::String(task) + "'",
                         juce::dontSendNotification);
  }

  updateUseSelectedButtonState();
  updateCatalogWarning();
}

void ModelBrowserPanel::updateTaskScopeUiState() {
  if (isCuratedLockEnabled()) {
    catalogModeBox_.setSelectedId(1, juce::dontSendNotification);
  }
  searchEditor_.setEnabled(!isCuratedLockEnabled() && !isCuratedMode());
  updateCatalogWarning();
}

void ModelBrowserPanel::updateActivePackLabel() {
  const auto task = selectedTaskScope();
  const auto it = activePackByTask_.find(task);
  juce::String text = "(none)";
  if (it != activePackByTask_.end() && !it->second.empty()) {
    text = juce::String(it->second);
  }
  activePackValueLabel_.setText(text, juce::dontSendNotification);
}

void ModelBrowserPanel::updateInstalledPackChoices() {
  installedPackIdByComboId_.clear();
  installedPackBox_.clear(juce::dontSendNotification);

  const auto task = selectedTaskScope();
  const auto activeIt = activePackByTask_.find(task);
  const auto activePackId = activeIt != activePackByTask_.end() ? activeIt->second : std::string {};

  int comboId = 1;
  int selectedComboId = 0;
  for (const auto& pack : installedPacks_) {
    if (normalizeTaskScope(pack.taskScope) != task) {
      continue;
    }
    const auto displayName = pack.name.empty() ? pack.id : (pack.name + " (" + pack.id + ")");
    installedPackBox_.addItem(juce::String(displayName), comboId);
    installedPackIdByComboId_[comboId] = pack.id;
    if (!activePackId.empty() && pack.id == activePackId) {
      selectedComboId = comboId;
    }
    ++comboId;
  }

  if (selectedComboId != 0) {
    installedPackBox_.setSelectedId(selectedComboId, juce::dontSendNotification);
  } else if (!installedPackIdByComboId_.empty()) {
    installedPackBox_.setSelectedItemIndex(0, juce::dontSendNotification);
  }
  updateSetActivePackButtonState();
}

void ModelBrowserPanel::updateSetActivePackButtonState() {
  const auto selectedId = installedPackBox_.getSelectedId();
  const auto selectedIt = installedPackIdByComboId_.find(selectedId);
  if (selectedIt == installedPackIdByComboId_.end()) {
    setActivePackButton_.setEnabled(false);
    return;
  }
  const auto task = selectedTaskScope();
  const auto activeIt = activePackByTask_.find(task);
  const bool alreadyActive = activeIt != activePackByTask_.end() && activeIt->second == selectedIt->second;
  setActivePackButton_.setEnabled(actionsEnabled_ && !alreadyActive);
}

void ModelBrowserPanel::updateUseSelectedButtonState() {
  if (selectedModelId_.empty()) {
    useSelectedButton_.setEnabled(false);
    useSelectedButton_.setButtonText("Use Selected for Task");
    return;
  }
  const auto installed = installedModelIds_.contains(selectedModelId_);
  useSelectedButton_.setEnabled(installed);
  if (!installed) {
    useSelectedButton_.setButtonText("Use Selected for Task");
    return;
  }
  useSelectedButton_.setButtonText("Use Selected for Task");
}

void ModelBrowserPanel::requestConfirmation(juce::MessageBoxIconType iconType,
                                            juce::String title,
                                            juce::String message,
                                            juce::String confirmButtonText,
                                            std::function<void()> onConfirm) {
  auto safeThis = juce::Component::SafePointer<ModelBrowserPanel>(this);
  juce::AlertWindow::showOkCancelBox(
      iconType,
      std::move(title),
      std::move(message),
      std::move(confirmButtonText),
      "Cancel",
      this,
      juce::ModalCallbackFunction::create([safeThis, onConfirm = std::move(onConfirm)](int result) mutable {
        if (result != 1 || safeThis == nullptr) {
          return;
        }
        if (onConfirm) {
          onConfirm();
        }
      }));
}

void ModelBrowserPanel::updateCapabilityReport(const ai::HubModelInfo* model) {
  if (model == nullptr) {
    capabilityReport_.setText("No model selected.", juce::dontSendNotification);
    return;
  }

  const auto task = normalizeTaskScope(model->taskScope);
  const auto outputs = expectedOutputKeysForTask(task);

  std::ostringstream report;
  report << "Name: " << (model->displayName.empty() ? model->repoId : model->displayName) << "\n";
  report << "Model ID: " << modelKey(*model) << "\n";
  report << "Task: " << task << "\n";
  {
    const auto activeIt = activePackByTask_.find(task);
    report << "Active task pack: "
           << ((activeIt != activePackByTask_.end() && !activeIt->second.empty()) ? activeIt->second : "(none)")
           << "\n";
  }
  report << "Engine: " << inferEngine(*model) << "\n";
  report << "Outputs: ";
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i != 0) {
      report << ", ";
    }
    report << outputs[i];
  }
  report << "\n";
  report << "License: " << (model->license.empty() ? "unknown" : model->license) << "\n";
  report << "Source: " << (model->sourceUrl.empty() ? model->source : model->sourceUrl) << "\n";
  report << "Primary file: " << model->primaryFile << "\n";
  report << "Revision: " << model->revision << "\n";
  report << "Installed: " << (installedModelIds_.contains(modelKey(*model)) ? "yes" : "no") << "\n";
  report << "Compatible: " << (model->compatible ? "yes" : "no");
  if (!model->compatibilityReport.empty()) {
    report << " (" << model->compatibilityReport << ")";
  }
  report << "\n";
  report << "Curated: " << (model->curated ? "yes" : "no") << "\n";
  report << "Recommended: " << (model->recommended ? "yes" : "no");

  capabilityReport_.setText(report.str(), juce::dontSendNotification);
}

void ModelBrowserPanel::updateCatalogWarning() {
  const auto selectedScope = selectedTaskScope();
  const bool mixOrMasterScope = selectedScope == "mix" || selectedScope == "master";
  const bool curatedCatalog = isCuratedLockEnabled() || isCuratedMode();

  juce::String warning;
  if (mixOrMasterScope && curatedCatalog) {
    const bool hasCuratedTaskModel = std::any_of(models_.begin(), models_.end(), [&](const ai::HubModelInfo& model) {
      return normalizeTaskScope(model.taskScope) == selectedScope && model.curated && model.compatible;
    });
    if (!hasCuratedTaskModel) {
      warning = "No curated " + juce::String(selectedScope) +
                " packs are available yet. Switch to Raw Search or install a local pack.";
    }
  }

  catalogWarningLabel_.setText(warning, juce::dontSendNotification);
  catalogWarningLabel_.setVisible(!warning.isEmpty());
}

std::vector<ai::HubModelInfo> ModelBrowserPanel::visibleModels() const {
  std::vector<ai::HubModelInfo> visible;
  const auto selectedScope = selectedTaskScope();
  for (const auto& model : models_) {
    const auto modelScope = normalizeTaskScope(model.taskScope);
    if (modelScope != selectedScope) {
      continue;
    }
    if (isCuratedLockEnabled() && !model.curated) {
      continue;
    }
    visible.push_back(model);
  }
  return visible;
}

std::string ModelBrowserPanel::normalizeTaskScope(std::string scope) {
  return normalizedTaskScope(std::move(scope));
}

} // namespace automix::app
