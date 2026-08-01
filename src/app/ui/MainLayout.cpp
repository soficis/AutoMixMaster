#include "app/ui/MainLayout.h"

#include "app/style/AutoMixLookAndFeel.h"
#include "app/ui/AudioPreviewManager.h"
#include "app/ui/ControlDeck.h"
#include "app/ui/GlowMeters.h"
#include "app/ui/HeaderBar.h"
#include "app/ui/HeroWaveform.h"
#include "app/ui/MainLayoutInternal.h"
#include "app/ui/ModelBrowserPanel.h"
#include "app/ui/StemPanel.h"
#include "app/ui/TaskCenterPanel.h"
#include "app/ui/TaskOrchestrator.h"
#include "app/ui/TransportBar.h"
#include "app/ui/VerificationEngine.h"
#include "renderers/RendererPipeline.h"
#include "util/FileUtils.h"

#include <juce_audio_utils/juce_audio_utils.h>

#include <algorithm>
#include <cmath>

namespace automix::app {

using namespace automix::app::detail;
using namespace automix::app::theme;

namespace {

constexpr float kMeterMinDb = -60.0f;
constexpr float kMeterMaxDb = 6.0f;

/// Linear amplitude to dB on the GlowMeters scale ([-60, +6] dBFS).
float linearToMeterDb(const float linear) {
  if (linear <= 1.0e-6f)
    return kMeterMinDb;
  return std::clamp(20.0f * std::log10(linear), kMeterMinDb, kMeterMaxDb);
}

} // namespace

// ── Updated StemPanel from session (needs StemPanel full type) ─

void automix::app::detail::updateStemPanelFromSession(StemPanel& panel, const domain::Session& session) {
  std::vector<StemPanel::StemDisplay> stems;
  stems.reserve(session.stems.size());
  for (const auto& s : session.stems) {
    StemPanel::StemDisplay d;
    d.id = s.id;
    d.name = s.name;
    d.role = s.role;
    d.enabled = s.enabled;
    d.volume = 1.0f;
    stems.push_back(std::move(d));
  }
  panel.setStems(stems);
}

// ─────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────

MainLayout::MainLayout() {
  setWantsKeyboardFocus(true);

  // 1. Create UI components
  headerBar_ = std::make_unique<HeaderBar>();
  heroWaveform_ = std::make_unique<HeroWaveform>();
  transportBar_ = std::make_unique<TransportBar>();
  controlDeck_ = std::make_unique<ControlDeck>();
  taskCenter_ = std::make_unique<TaskCenterPanel>();

  const bool batchRecursiveEnabled = loadBatchRecursivePreference();
  const bool writeReportSidecar = loadExportReportSidecarPreference();
  controlDeck_->getBatchRecursiveToggle().setToggleState(batchRecursiveEnabled, juce::dontSendNotification);
  setBatchRecursiveEnvironment(batchRecursiveEnabled);
  sessionManager_.session().batchRecursiveEnabled = batchRecursiveEnabled;
  sessionManager_.session().renderSettings.writePerExportReportJson = writeReportSidecar;

  addAndMakeVisible(*headerBar_);
  addAndMakeVisible(*heroWaveform_);
  addAndMakeVisible(*transportBar_);
  addAndMakeVisible(*controlDeck_);
  addAndMakeVisible(*taskCenter_);

  // Destructive Clear: MainLayout-owned (not part of TransportBar), confirmed on click.
  clearTracksButton_.setTooltip("Clear Imported Tracks (asks for confirmation)");
  clearTracksButton_.setWantsKeyboardFocus(true);
  clearTracksButton_.setColour(juce::TextButton::buttonColourId, colour(colours::surface));
  clearTracksButton_.setColour(juce::TextButton::buttonOnColourId, colour(colours::surfaceLight));
  clearTracksButton_.setColour(juce::TextButton::textColourOffId, colour(colours::warning));
  clearTracksButton_.setColour(juce::TextButton::textColourOnId, colour(colours::warning));
  clearTracksButton_.onClick = [this] { onClearTracks(); };
  addAndMakeVisible(clearTracksButton_);

  // 2. Create coordinators
  taskOrchestrator_ = std::make_unique<TaskOrchestrator>(*taskCenter_);
  previewManager_ = std::make_unique<AudioPreviewManager>(backgroundPool_, this);

  previewManager_->onPreviewReady = [this](const engine::AudioBuffer& buffer, double previousProgress) {
    heroWaveform_->setBuffer(buffer);
    updateTransportFromBuffer(buffer);
    if (previousProgress > 0.0 && previousProgress < 1.0)
      transportController_.seekToFraction(previousProgress);
  };
  previewManager_->onPreviewError = [this](const juce::String& errorText) {
    taskOrchestrator_->appendHistory("Preview build failed: " + errorText);
  };
  previewManager_->onHistoryLine = [this](const juce::String& line) {
    taskOrchestrator_->appendHistory(line);
  };

  // 3. Populate combo boxes and create controllers
  initComboBoxes();
  initControllers();

  // 4. Wire all UI callbacks
  wireHeaderCallbacks();
  wireTransportCallbacks();
  wireControlDeckCallbacks();
  wireHeroWaveformCallbacks();

  taskCenter_->onCancel = [this] { taskOrchestrator_->cancelActiveTask(); };

  // 5. Audio device & transport
  audioDeviceManager_.initialise(0, 2, nullptr, true);
  audioDeviceManager_.addAudioCallback(this);
  transportController_.addChangeListener(this);
  startTimerHz(30);
  updateTransportDisplay();

  // 6. Application command manager: keyboard shortcuts are dispatched here
  commandManager_.setFirstCommandTarget(this);
  commandManager_.registerAllCommandsForTarget(this);
}

// ── Controllers factory ────────────────────────────────────────

void MainLayout::initControllers() {
  auto safe = safeAsync(this);

  // --- ModelController ---
  {
    ModelController::Callbacks cb;
    cb.onStatus = cbFactory::onStatus(safe);
    cb.onTaskHistory = cbFactory::onHistory(safe);
    cb.onProgress = cbFactory::onProgressForTask(safe, ActiveTask::Model);
    cb.onReport = [safe](const std::string& text) {
      juce::MessageManager::callAsync([safe, text]() {
        if (safe)
          safe->taskOrchestrator_->appendHistory("Report: " + juce::String(text));
      });
    };
    cb.onModelPacksChanged = [safe]() {
      juce::MessageManager::callAsync([safe]() {
        if (safe) {
          safe->refreshModelPacks();
          if (safe->modelBrowserPanel_ != nullptr) {
            safe->modelBrowserPanel_->setActivePackDisplay(activePackMapForUi(safe->modelManager_));
            safe->modelBrowserPanel_->setInstalledPacks(safe->modelManager_.availablePacks());
            safe->modelBrowserPanel_->setInstalledModelIds(safe->modelController_->installedModelIds());
          }
        }
      });
    };
    cb.onCatalogReady = [safe](const bool cancelled) {
      juce::MessageManager::callAsync([safe, cancelled]() {
        if (!safe)
          return;
        if (cancelled)
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Model, "Model catalog cancelled");
        else
          safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Model, "Model catalog ready");

        if (safe->modelBrowserPanel_ != nullptr) {
          safe->modelBrowserPanel_->setDiscoveredModels(safe->modelController_->discoveredModels());
          safe->modelBrowserPanel_->setInstalledPacks(safe->modelManager_.availablePacks());
          safe->modelBrowserPanel_->setInstalledModelIds(safe->modelController_->installedModelIds());
          safe->modelBrowserPanel_->setActivePackDisplay(activePackMapForUi(safe->modelManager_));
          safe->modelBrowserPanel_->setActionsEnabled(true);
          if (safe->modelController_->discoveredModels().empty()) {
            safe->modelBrowserPanel_->setStatus("No compatible models found");
          } else {
            safe->modelBrowserPanel_->setStatus(
                juce::String(static_cast<int>(safe->modelController_->discoveredModels().size())) +
                " models found. Install one, then click 'Use Selected for Task'.");
          }
        }
      });
    };
    cb.onInstallComplete = [safe](const bool cancelled) {
      juce::MessageManager::callAsync([safe, cancelled]() {
        if (!safe)
          return;
        if (cancelled)
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Model, "Model install cancelled");
        else
          safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Model, "Model installed");
        if (safe->modelBrowserPanel_ != nullptr) {
          safe->modelBrowserPanel_->setInstalledModelIds(safe->modelController_->installedModelIds());
          safe->modelBrowserPanel_->setActionsEnabled(true);
        }
      });
    };
    cb.onUninstallComplete = [safe](const bool cancelled) {
      juce::MessageManager::callAsync([safe, cancelled]() {
        if (!safe)
          return;
        if (cancelled)
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Model, "Model uninstall cancelled");
        else
          safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Model, "Model uninstalled");
        if (safe->modelBrowserPanel_ != nullptr) {
          safe->modelBrowserPanel_->setInstalledModelIds(safe->modelController_->installedModelIds());
          safe->modelBrowserPanel_->setActionsEnabled(true);
        }
      });
    };
    cb.onUpdateCheckComplete = [safe](const bool cancelled) {
      juce::MessageManager::callAsync([safe, cancelled]() {
        if (!safe)
          return;
        if (cancelled)
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Model, "Model update check cancelled");
        else
          safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Model, "Model updates checked");
        if (safe->modelBrowserPanel_ != nullptr) {
          safe->modelBrowserPanel_->setActionsEnabled(true);
        }
      });
    };
    modelController_ = std::make_unique<ModelController>(
        modelManager_,
        backgroundPool_,
        std::move(cb),
        ModelController::createDefaultHubOps());
  }

  // --- ImportController ---
  {
    ImportController::Callbacks cb;
    cb.onStatus = cbFactory::onStatus(safe);
    cb.onTaskHistory = cbFactory::onHistory(safe);
    cb.onProgress = cbFactory::onProgressForTask(safe, ActiveTask::Import);
    cb.onImportComplete = [safe](ImportResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        for (const auto& line : result.logLines)
          safe->taskOrchestrator_->appendHistory(juce::String(line));

        if (result.cancelled) {
          safe->pendingAutoMixAfterSeparationImport_ = false;
          safe->skipNextAutoMixSeparationCheck_ = false;
          safe->pendingPipelineExportFolder_.clear();
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Import, "Import cancelled");
          return;
        }

        safe->sessionManager_.session().stems = result.stems;
        safe->sessionManager_.session().originalMixPath = result.originalMixPath;
        updateStemPanelFromSession(safe->controlDeck_->getStemPanel(), safe->sessionManager_.session());
        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Import, "Import complete");
        safe->rebuildPreview();

        if (safe->pendingAutoMixAfterSeparationImport_) {
          safe->pendingAutoMixAfterSeparationImport_ = false;
          safe->skipNextAutoMixSeparationCheck_ = true;
          safe->taskOrchestrator_->appendHistory("AI stem separation import finished. Continuing Auto Mix.");
          safe->onAutoMix();
        }
      });
    };
    importController_ = std::make_unique<ImportController>(backgroundPool_, std::move(cb));
  }

  // --- ExportController ---
  {
    ExportController::Callbacks cb;
    cb.onStatus = cbFactory::onStatus(safe);
    cb.onTaskHistory = cbFactory::onHistory(safe);
    cb.onProgress = cbFactory::onProgressForTask(safe, ActiveTask::Export);
    cb.onExportComplete = [safe](ExportResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        for (const auto& line : result.logs)
          safe->taskOrchestrator_->appendHistory(juce::String(line));

        safe->analysisEntries_ = result.analysisEntries;

        if (result.success) {
          safe->taskOrchestrator_->appendHistory("Export succeeded: " + juce::String(result.outputAudioPath));
          safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Export, "Export complete");
          safe->startExportVerification(result.outputAudioPath);
        } else if (result.cancelled) {
          safe->exportVerificationSession_.reset();
          safe->exportVerificationSettings_.reset();
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Export, "Export cancelled");
        } else {
          safe->exportVerificationSession_.reset();
          safe->exportVerificationSettings_.reset();
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::Export, result.crashMessage.toStdString());
        }
      });
    };
    exportController_ = std::make_unique<ExportController>(backgroundPool_, std::move(cb));
  }

  // --- ProcessingController ---
  {
    ProcessingController::Callbacks cb;
    cb.onStatus = cbFactory::onStatus(safe);
    cb.onTaskHistory = cbFactory::onHistory(safe);
    cb.onProgress = [safe](const double progress) {
      juce::MessageManager::callAsync([safe, progress]() {
        if (!safe)
          return;
        const auto task = safe->taskOrchestrator_->activeTask();
        if (task == ActiveTask::AutoMix || task == ActiveTask::AutoMaster || task == ActiveTask::Batch) {
          safe->taskOrchestrator_->setProgress(progress);
        }
      });
    };
    cb.onAutoMixComplete = [safe](AutoMixResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        if (result.cancelled) {
          safe->pendingPipelineExportFolder_.clear();
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::AutoMix, "Auto Mix cancelled");
          return;
        }
        if (result.errorText.isNotEmpty()) {
          safe->pendingPipelineExportFolder_.clear();
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::AutoMix, result.errorText.toStdString());
          return;
        }

        safe->analysisEntries_ = result.analysisEntries;
        if (result.mixPlan.has_value())
          safe->sessionManager_.session().mixPlan = result.mixPlan;
        safe->taskOrchestrator_->appendHistory(result.reportText);
        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::AutoMix, "Auto Mix complete");

        if (!safe->pendingPipelineExportFolder_.empty()) {
          safe->onAutoMaster();
        } else {
          safe->rebuildPreview();
        }
      });
    };
    cb.onAutoMasterComplete = [safe](AutoMasterResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        if (result.cancelled) {
          safe->pendingPipelineExportFolder_.clear();
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::AutoMaster, "Auto Master cancelled");
          return;
        }
        if (result.errorText.isNotEmpty()) {
          safe->pendingPipelineExportFolder_.clear();
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::AutoMaster, result.errorText.toStdString());
          return;
        }

        safe->sessionManager_.session().masterPlan = result.masterPlan;
        safe->taskOrchestrator_->appendHistory(result.reportAppend);
        safe->updateMeterPanel(result.previewReport);

        safe->updateTransportFromBuffer(result.previewMaster);
        safe->heroWaveform_->setBuffer(result.previewMaster);
        safe->previewManager_->setBuffer(std::move(result.previewMaster));

        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::AutoMaster, "Auto Master complete");

        if (!safe->pendingPipelineExportFolder_.empty()) {
          safe->triggerPipelineExport();
        }
      });
    };
    cb.onBatchComplete = [safe](BatchResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        if (result.errorText.isNotEmpty()) {
          safe->taskOrchestrator_->appendHistory("Batch error: " + result.errorText);
          safe->batchVerificationInputFolder_.reset();
          safe->batchVerificationSettings_.reset();
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::Batch, result.errorText.toStdString());
          juce::AlertWindow::showMessageBoxAsync(
              juce::AlertWindow::WarningIcon,
              "Batch failed",
              result.errorText);
          return;
        }

        safe->taskOrchestrator_->appendHistory(result.summary);

        juce::String completionMessage;
        completionMessage << "Completed: " << result.completed << "\n"
                          << "Failed: " << result.failed << "\n"
                          << "Cancelled: " << result.cancelled << "\n";
        if (!result.outputFolder.empty()) {
          completionMessage << "Output folder:\n" << juce::String(result.outputFolder) << "\n";
        }

        if (result.cancelled > 0 && result.completed == 0 && result.failed == 0) {
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Batch, "Batch cancelled");
          juce::AlertWindow::showMessageBoxAsync(
              juce::AlertWindow::WarningIcon,
              "Batch cancelled",
              completionMessage);
          safe->batchVerificationInputFolder_.reset();
          safe->batchVerificationSettings_.reset();
          return;
        }

        if (result.failed > 0) {
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::Batch, "Batch completed with failures");
          juce::AlertWindow::showMessageBoxAsync(
              juce::AlertWindow::WarningIcon,
              "Batch completed with failures",
              completionMessage);
          if (result.completed > 0 && !result.outputFolder.empty()) {
            safe->startBatchVerification(result.outputFolder);
          } else {
            safe->batchVerificationInputFolder_.reset();
            safe->batchVerificationSettings_.reset();
          }
          return;
        }

        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Batch, "Batch complete");
        juce::AlertWindow::showMessageBoxAsync(
            juce::AlertWindow::InfoIcon,
            "Batch complete",
            completionMessage);
        if (result.completed > 0 && !result.outputFolder.empty()) {
          safe->startBatchVerification(result.outputFolder);
        } else {
          safe->batchVerificationInputFolder_.reset();
          safe->batchVerificationSettings_.reset();
        }
      });
    };
    processingController_ = std::make_unique<ProcessingController>(backgroundPool_, std::move(cb));
  }

  // --- SessionController ---
  {
    SessionController::Callbacks cb;
    cb.onStatus = cbFactory::onStatus(safe);
    cb.onTaskHistory = cbFactory::onHistory(safe);
    cb.onProgress = cbFactory::onProgressForTask(safe, ActiveTask::Session);
    cb.onSaveComplete = [safe](SessionSaveResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() mutable {
        if (!safe)
          return;

        if (result.cancelled) {
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Session, "Session save cancelled");
          return;
        }
        if (!result.success) {
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::Session, result.errorText.toStdString());
          return;
        }

        safe->taskOrchestrator_->appendHistory("Session saved to " + juce::String(result.path));
        safe->headerBar_->setSessionName(juce::File(result.path).getFileNameWithoutExtension());
        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Session, "Session saved");
      });
    };
    cb.onLoadComplete = [safe](SessionLoadResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() mutable {
        if (!safe)
          return;

        if (result.cancelled) {
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Session, "Session load cancelled");
          return;
        }
        if (result.errorText.isNotEmpty() || !result.session.has_value()) {
          auto msg = result.errorText.isNotEmpty() ? result.errorText.toStdString() : "Unknown error";
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::Session, msg);
          return;
        }

        safe->applyLoadedSession(std::move(result.session.value()), juce::String(result.path));
        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Session, "Session loaded");
      });
    };
    sessionController_ = std::make_unique<SessionController>(backgroundPool_, std::move(cb));
  }
}

// ── Combo box initialization ───────────────────────────────────

void MainLayout::initComboBoxes() {
  refreshRenderers();
  refreshCodecAvailability();
  refreshModelPacks();
  populateMasterPresetSelectors();
  refreshProjectProfiles();
  applySessionUiSelections();
  syncSessionUiSelections();
}

// ─────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────

MainLayout::~MainLayout() {
  taskOrchestrator_->cancelAll();
  stopTimer();
  audioDeviceManager_.removeAudioCallback(this);
  transportController_.removeChangeListener(this);
  backgroundPool_.removeAllJobs(true, 5000);
}

// ─────────────────────────────────────────────────────────────────
// Layout
// ─────────────────────────────────────────────────────────────────

void MainLayout::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::background));
}

void MainLayout::resized() {
  auto area = getLocalBounds();

  // Transport row: transport bar (flex) + destructive Clear (fixed, right edge,
  // outside the TransportBar component so it stays a deliberate, confirmed action).
  auto transportRow = area.removeFromTop(kTransportHeight);
  {
    juce::FlexBox rowFb;
    rowFb.flexDirection = juce::FlexBox::Direction::row;
    rowFb.items.add(juce::FlexItem(*transportBar_).withFlex(1.0f));
    rowFb.items.add(
        juce::FlexItem(clearTracksButton_)
            .withWidth(68.0f)
            .withMargin(juce::FlexItem::Margin(8.0f, 4.0f, 8.0f, 4.0f)));
    rowFb.performLayout(transportRow);
  }

  juce::FlexBox fb;
  fb.flexDirection = juce::FlexBox::Direction::column;
  fb.justifyContent = juce::FlexBox::JustifyContent::flexStart;

  fb.items.add(juce::FlexItem(*headerBar_).withHeight(static_cast<float>(kHeaderHeight)));
  fb.items.add(juce::FlexItem(*heroWaveform_).withFlex(1.5f).withMinHeight(120.0f));
  fb.items.add(juce::FlexItem(*controlDeck_).withFlex(2.5f).withMinHeight(180.0f));
  fb.items.add(juce::FlexItem(*taskCenter_).withFlex(1.2f).withMinHeight(160.0f));

  fb.performLayout(area);
}

// ─────────────────────────────────────────────────────────────────
// Keyboard Shortcuts
// ─────────────────────────────────────────────────────────────────

namespace {

/// Formats a KeyPress for display in the cheatsheet (e.g. "Ctrl+Shift+M").
juce::String describeShortcutKey(const juce::KeyPress& key) {
  const auto& mods = key.getModifiers();
  juce::String text;
  if (mods.isCtrlDown())
    text << "Ctrl+";
  if (mods.isShiftDown())
    text << "Shift+";
  if (mods.isAltDown())
    text << "Alt+";
  if (key.getKeyCode() == juce::KeyPress::spaceKey)
    text << "Space";
  else if (key.getKeyCode() >= 33 && key.getKeyCode() < 176)
    text << juce::String::charToString(
        static_cast<juce::juce_wchar>(juce::CharacterFunctions::toUpperCase(key.getKeyCode())));
  else
    text << key.getTextDescription();
  return text;
}

/// Modal cheatsheet listing every command and its key binding(s).
class ShortcutsDialog final : public juce::Component {
public:
  explicit ShortcutsDialog(const std::vector<ShortcutEntry>& entries) {
    setWantsKeyboardFocus(true);
    setSize(kDialogWidth, kDialogHeight);

    titleLabel_.setText("Keyboard Shortcuts", juce::dontSendNotification);
    titleLabel_.setFont(typography::subhead());
    titleLabel_.setColour(juce::Label::textColourId, colour(colours::text));
    titleLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel_);

    for (const auto& entry : entries) {
      auto& row = rows_.emplace_back();
      row.name = entry.name;
      row.keys = describeShortcutKey(entry.defaultKey);
    }
    // Merge duplicate command rows (Redo has two bindings) into one row.
    for (size_t i = 0; i < rows_.size();) {
      const juce::String name(rows_[i].name);
      bool merged = false;
      for (size_t j = i + 1; j < rows_.size();) {
        if (rows_[j].name == name) {
          rows_[i].keys << ", " << rows_[j].keys;
          rows_.erase(rows_.begin() + static_cast<std::ptrdiff_t>(j));
          merged = true;
        } else {
          ++j;
        }
      }
      if (!merged)
        ++i;
    }
    for (const auto& row : rows_) {
      auto nameLabel = std::make_unique<juce::Label>();
      nameLabel->setText(row.name, juce::dontSendNotification);
      nameLabel->setFont(typography::body());
      nameLabel->setColour(juce::Label::textColourId, colour(colours::text));
      nameLabel->setJustificationType(juce::Justification::centredLeft);
      auto keyLabel = std::make_unique<juce::Label>();
      keyLabel->setText(row.keys, juce::dontSendNotification);
      keyLabel->setFont(typography::body());
      keyLabel->setColour(juce::Label::textColourId, colour(colours::secondary));
      keyLabel->setJustificationType(juce::Justification::centredRight);
      addAndMakeVisible(*nameLabel);
      addAndMakeVisible(*keyLabel);
      nameLabels_.push_back(std::move(nameLabel));
      keyLabels_.push_back(std::move(keyLabel));
    }

    closeButton_.setButtonText("Close");
    closeButton_.onClick = [this] { exitModalState(0); };
    addAndMakeVisible(closeButton_);
  }

  void paint(juce::Graphics& g) override {
    g.fillAll(colour(colours::surface));
    g.setColour(colour(colours::surfaceBorder));
    g.drawRect(getLocalBounds(), metrics::borderWidth);
  }

  void resized() override {
    auto area = getLocalBounds().reduced(spacing::marginLarge);
    titleLabel_.setBounds(area.removeFromTop(28));
    area.removeFromTop(spacing::gapMedium);
    constexpr int rowHeight = 26;
    for (size_t i = 0; i < nameLabels_.size(); ++i) {
      auto row = area.removeFromTop(rowHeight);
      nameLabels_[i]->setBounds(row.removeFromLeft(row.getWidth() * 3 / 5));
      keyLabels_[i]->setBounds(row);
    }
    area.removeFromTop(spacing::gapLarge);
    closeButton_.setBounds(area.removeFromTop(32).withSizeKeepingCentre(120, 32));
  }

  bool keyPressed(const juce::KeyPress& key) override {
    if (key == juce::KeyPress::escapeKey) {
      exitModalState(0);
      return true;
    }
    return juce::Component::keyPressed(key);
  }

private:
  static constexpr int kDialogWidth = 460;
  static constexpr int kDialogHeight = 560;

  juce::Label titleLabel_;
  juce::TextButton closeButton_{"Close"};
  struct DisplayRow {
    juce::String name;
    juce::String keys;
  };
  std::vector<DisplayRow> rows_;
  std::vector<std::unique_ptr<juce::Label>> nameLabels_;
  std::vector<std::unique_ptr<juce::Label>> keyLabels_;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ShortcutsDialog)
};

} // namespace

bool MainLayout::keyPressed(const juce::KeyPress& key) {
  auto* mappings = commandManager_.getKeyMappings();
  const juce::CommandID commandId = mappings->findCommandForKeyPress(key);
  if (commandId == 0)
    return false;
  return commandManager_.invokeDirectly(commandId, false);
}

juce::ApplicationCommandTarget* MainLayout::getNextCommandTarget() {
  return nullptr;
}

void MainLayout::getAllCommands(juce::Array<juce::CommandID>& commands) {
  for (const auto& entry : shortcutTable()) {
    const auto commandId = static_cast<juce::CommandID>(entry.command);
    if (!commands.contains(commandId))
      commands.add(commandId);
  }
}

void MainLayout::getCommandInfo(juce::CommandID commandID, juce::ApplicationCommandInfo& result) {
  bool found = false;
  for (const auto& entry : shortcutTable()) {
    if (static_cast<juce::CommandID>(entry.command) == commandID) {
      if (!found) {
        result.setInfo(juce::String(entry.name), juce::String(entry.name), "General", 0);
        found = true;
      }
      result.addDefaultKeypress(entry.defaultKey.getKeyCode(), entry.defaultKey.getModifiers());
    }
  }
}

bool MainLayout::perform(const juce::ApplicationCommandTarget::InvocationInfo& info) {
  switch (static_cast<ShortcutCommand>(info.commandID)) {
    case ShortcutCommand::saveSession:     onSaveSession();     break;
    case ShortcutCommand::loadSession:     onLoadSession();     break;
    case ShortcutCommand::import:          onImport();          break;
    case ShortcutCommand::autoMix:         onAutoMix();         break;
    case ShortcutCommand::autoMaster:      onAutoMaster();      break;
    case ShortcutCommand::autoMixMaster:   onAutoMixMaster();   break;
    case ShortcutCommand::exportProject:   onExport();          break;
    case ShortcutCommand::modelsDialog:    onModelsDialog();    break;
    case ShortcutCommand::undo:            onUndo();            break;
    case ShortcutCommand::redo:            onRedo();            break;
    case ShortcutCommand::playPause:       onTogglePlayPause(); break;
    case ShortcutCommand::shortcutsDialog: showShortcutsDialog(); break;
  }
  return true;
}

void MainLayout::onTogglePlayPause() {
  if (transportController_.isPlaying()) {
    transportController_.pause();
    transportBar_->setPlaying(false);
  } else {
    transportController_.play();
    transportBar_->setPlaying(true);
  }
}

void MainLayout::showShortcutsDialog() {
  auto* dialog = new ShortcutsDialog(shortcutTable());
  dialog->setAlwaysOnTop(true);
  dialog->centreWithSize(dialog->getWidth(), dialog->getHeight());
  dialog->setVisible(true);
  dialog->enterModalState(true, nullptr, true); // modal manager owns + deletes it
}

// ─────────────────────────────────────────────────────────────────
// Timer
// ─────────────────────────────────────────────────────────────────

void MainLayout::timerCallback() {
  updateTransportDisplay();

  // Reconcile the transport bar with the atomic play state (covers end-of-track
  // auto-stop, which is realtime-only and posts no change message).
  const bool playing = transportController_.isPlaying();
  if (playing != transportBarPlaying_) {
    transportBarPlaying_ = playing;
    transportBar_->setPlaying(playing);
  }

  // Live meters: copy the audio-thread targets into GlowMeters (message thread).
  auto& meters = controlDeck_->getGlowMeters();
  meters.setLevels(liveMeterLeftLevel_.load(std::memory_order_relaxed),
                   liveMeterRightLevel_.load(std::memory_order_relaxed));
  meters.setPeaks(liveMeterLeftPeak_.load(std::memory_order_relaxed),
                  liveMeterRightPeak_.load(std::memory_order_relaxed));
}

// ─────────────────────────────────────────────────────────────────
// Change Listener (transport state changes)
// ─────────────────────────────────────────────────────────────────

void MainLayout::changeListenerCallback(juce::ChangeBroadcaster* source) {
  if (source == &transportController_) {
    transportBarPlaying_ = transportController_.isPlaying();
    transportBar_->setPlaying(transportBarPlaying_);
  }
}

// ─────────────────────────────────────────────────────────────────
// Audio Device Callback
// ─────────────────────────────────────────────────────────────────

void MainLayout::audioDeviceIOCallbackWithContext(const float* const* /*inputChannelData*/,
                                                  int /*numInputChannels*/,
                                                  float* const* outputChannelData, int numOutputChannels,
                                                  int numSamples,
                                                  const juce::AudioIODeviceCallbackContext& /*context*/) {
  // Realtime-safe path: the preview buffer and transport state are published via
  // atomics, so this callback performs no locking, no heap allocation and no
  // message posting regardless of what the message thread is doing.
  const auto buffer = previewManager_->currentBuffer();
  const float outputGain = std::clamp(outputVolume_.load(std::memory_order_relaxed), 0.0f, 1.5f);

  if (buffer == nullptr || !transportController_.isPlaying() || buffer->getNumSamples() == 0) {
    for (int ch = 0; ch < numOutputChannels; ++ch)
      if (outputChannelData[ch])
        std::fill_n(outputChannelData[ch], numSamples, 0.0f);
    return;
  }

  const int totalSamples = buffer->getNumSamples();
  const int bufChannels = buffer->getNumChannels();
  int64_t pos = transportController_.positionSamples();

  // Live meter accumulation (RMS + peak per output channel, post-gain).
  float peakL = 0.0f;
  float peakR = 0.0f;
  double sumSqL = 0.0;
  double sumSqR = 0.0;

  int i = 0;
  for (; i < numSamples; ++i) {
    if (pos >= totalSamples)
      break;

    for (int ch = 0; ch < numOutputChannels; ++ch) {
      if (outputChannelData[ch] == nullptr)
        continue;
      const int srcCh = std::min(ch, bufChannels - 1);
      const float sample = buffer->getSample(srcCh, static_cast<int>(pos)) * outputGain;
      outputChannelData[ch][i] = sample;
      if (ch == 0) {
        peakL = std::max(peakL, std::abs(sample));
        sumSqL += static_cast<double>(sample) * sample;
      } else if (ch == 1) {
        peakR = std::max(peakR, std::abs(sample));
        sumSqR += static_cast<double>(sample) * sample;
      }
    }
    ++pos;
  }

  if (i < numSamples) {
    for (int ch = 0; ch < numOutputChannels; ++ch)
      if (outputChannelData[ch])
        std::fill_n(outputChannelData[ch] + i, numSamples - i, 0.0f);
    transportController_.stopFromAudioThread();
  }

  transportController_.setPositionRealtime(pos);

  // Publish live meter targets (relaxed atomic stores; consumed by the UI timer).
  const int meterSamples = std::max(i, 1);
  liveMeterLeftLevel_.store(
      linearToMeterDb(static_cast<float>(std::sqrt(sumSqL / meterSamples))), std::memory_order_relaxed);
  liveMeterRightLevel_.store(
      linearToMeterDb(static_cast<float>(std::sqrt(sumSqR / meterSamples))), std::memory_order_relaxed);
  liveMeterLeftPeak_.store(linearToMeterDb(peakL), std::memory_order_relaxed);
  liveMeterRightPeak_.store(linearToMeterDb(peakR), std::memory_order_relaxed);
}

void MainLayout::audioDeviceAboutToStart(juce::AudioIODevice* /*device*/) {}
void MainLayout::audioDeviceStopped() {}

// ─────────────────────────────────────────────────────────────────
// Wiring: Header
// ─────────────────────────────────────────────────────────────────

void MainLayout::wireHeaderCallbacks() {
  headerBar_->onSaveSession = [this] { onSaveSession(); };
  headerBar_->onLoadSession = [this] { onLoadSession(); };
  headerBar_->onModels = [this] { onModelsDialog(); };
  headerBar_->onSettings = [this] { onSettings(); };
  headerBar_->onProfileSelected = [this](const juce::String& profileId) {
    onHeaderProfileSelected(profileId);
  };
}

// ─────────────────────────────────────────────────────────────────
// Wiring: Transport
// ─────────────────────────────────────────────────────────────────

void MainLayout::wireTransportCallbacks() {
  transportBar_->onPlay = [this] {
    transportController_.play();
    transportBar_->setPlaying(true);
  };
  transportBar_->onPause = [this] {
    transportController_.pause();
    transportBar_->setPlaying(false);
  };
  transportBar_->onStop = [this] {
    transportController_.stop();
    transportBar_->setPlaying(false);
  };
  transportBar_->onSkipToStart = [this] {
    transportController_.seekToFraction(0.0);
  };
  transportBar_->onSkipToEnd = [this] {
    transportController_.seekToFraction(1.0);
  };
  transportBar_->onLoopToggle = [this] {
    auto& tl = sessionManager_.session().timeline;
    tl.loopEnabled = !tl.loopEnabled;
    transportController_.setLoopRangeSeconds(tl.loopInSeconds, tl.loopOutSeconds, tl.loopEnabled);
    heroWaveform_->setLoopRange(tl.loopEnabled,
                                transportController_.loopInProgress(),
                                transportController_.loopOutProgress());
  };
  transportBar_->onSeekRelative = [this](double offsetSeconds) {
    double newPos = transportController_.positionSeconds() + offsetSeconds;
    double total = transportController_.totalSeconds();
    if (total > 0.0) {
      double frac = std::clamp(newPos / total, 0.0, 1.0);
      transportController_.seekToFraction(frac);
    }
  };
  transportBar_->onVolumeChanged = [this](double volume) {
    outputVolume_.store(static_cast<float>(clampVolumeToEngineRange(volume)), std::memory_order_relaxed);
  };
  transportBar_->onShowShortcuts = [this] { showShortcutsDialog(); };
}

// ─────────────────────────────────────────────────────────────────
// Clear imported tracks (destructive — confirmation required)
// ─────────────────────────────────────────────────────────────────

void MainLayout::onClearTracks() {
  const int numTracks = static_cast<int>(sessionManager_.session().stems.size());
  if (numTracks == 0)
    return; // Nothing imported; nothing to clear.

  juce::AlertWindow::showOkCancelBox(
      juce::MessageBoxIconType::WarningIcon, "Clear imported tracks",
      "Clear " + juce::String(numTracks) +
          " imported tracks? This cannot be undone \xe2\x80\x94 use Undo after T4.1.",
      "Clear", "Cancel", this,
      juce::ModalCallbackFunction::create([safe = safeAsync(this), numTracks](int result) {
        if (safe != nullptr && confirmClear(numTracks, result == 1))
          safe->performClearTracks();
      }));
}

void MainLayout::performClearTracks() {
  transportController_.stop();
  transportBar_->setPlaying(false);
  transportBar_->setTimeDisplay(0.0, 0.0);
  sessionManager_.session().stems.clear();
  detail::updateStemPanelFromSession(controlDeck_->getStemPanel(), sessionManager_.session());
}

// ─────────────────────────────────────────────────────────────────
// Wiring: ControlDeck
// ─────────────────────────────────────────────────────────────────

void MainLayout::wireControlDeckCallbacks() {
  controlDeck_->onImport = [this] { onImport(); };
  controlDeck_->onAutoMix = [this] { onAutoMix(); };
  controlDeck_->onAutoMaster = [this] { onAutoMaster(); };
  controlDeck_->onAutoMixMaster = [this] { onAutoMixMaster(); };
  controlDeck_->onBatch = [this] { onBatch(); };
  controlDeck_->onExport = [this] { onExport(); };

  controlDeck_->getStemPanel().onSoloChanged = [this](const std::string&, bool) {
    rebuildPreview();
  };
  controlDeck_->getStemPanel().onMuteChanged = [this](const std::string&, bool) {
    rebuildPreview();
  };
  controlDeck_->getStemPanel().onVolumeChanged = [this](const std::string&, float) {
    rebuildPreview();
  };

  controlDeck_->getRendererBox().onChange = [this] {
    const auto rendererId = selectionState_.rendererIdForCombo(controlDeck_->getRendererBox().getSelectedId());
    if (rendererId.has_value()) {
      sessionManager_.session().renderSettings.rendererName = rendererId.value();
    }
    updateRendererChainPreview();
  };
  controlDeck_->getProfileBox().onChange = [this] {
    const auto profileId = selectionState_.projectProfileIdForCombo(controlDeck_->getProfileBox().getSelectedId());
    if (profileId.has_value()) {
      sessionManager_.session().projectProfileId = profileId.value();
      auto profile = domain::findProjectProfile(projectProfiles_, profileId.value());
      if (profile.has_value()) {
        ProfileController pc;
        pc.applyProfile(sessionManager_.session(), profile.value());
        applySessionUiSelections();
        syncSessionUiSelections();
        taskOrchestrator_->appendHistory("Applied profile: " + juce::String(profile->name));
      }
    }
  };
  controlDeck_->getMasterPresetBox().onChange = [this] {
    const auto preset = selectionState_.masterPresetForCombo(controlDeck_->getMasterPresetBox().getSelectedId());
    if (preset.has_value()) {
      sessionManager_.session().selectedMasterPreset = preset.value();
      if (sessionManager_.session().masterPlan.has_value())
        sessionManager_.session().masterPlan->preset = preset.value();
    }
  };
  controlDeck_->getPlatformPresetBox().onChange = [this] {
    const auto preset = selectionState_.platformPresetForCombo(controlDeck_->getPlatformPresetBox().getSelectedId());
    if (preset.has_value()) {
      sessionManager_.session().selectedPlatformPreset = preset.value();
    }
  };
  controlDeck_->getExportFormatBox().onChange = [this] {
    const auto format = selectionState_.codecFormatForCombo(controlDeck_->getExportFormatBox().getSelectedId());
    if (format.has_value())
      sessionManager_.session().renderSettings.outputFormat = format.value();
  };
  controlDeck_->getExportModeBox().onChange = [this] {
    const auto mode = selectionState_.exportSpeedModeForCombo(controlDeck_->getExportModeBox().getSelectedId());
    if (mode.has_value())
      sessionManager_.session().renderSettings.exportSpeedMode = mode.value();
  };
  controlDeck_->getRendererChainToggle().onClick = [this] {
    const bool enabled = controlDeck_->getRendererChainToggle().getToggleState();
    sessionManager_.session().renderSettings.rendererChainEnabled = enabled;
    controlDeck_->getRendererChainModeBox().setEnabled(enabled);
    updateRendererChainPreview();
  };
  controlDeck_->getRendererChainModeBox().onChange = [this] {
    const auto mode = selectionState_.rendererChainModeForCombo(controlDeck_->getRendererChainModeBox().getSelectedId());
    if (mode.has_value()) {
      sessionManager_.session().renderSettings.rendererChainMode = mode.value();
    }
    updateRendererChainPreview();
  };
  controlDeck_->getResidualBlendSlider().onValueChange = [this] {
    sessionManager_.session().residualBlend = controlDeck_->getResidualBlendSlider().getValue();
  };
  controlDeck_->getSeparatedStemsToggle().onClick = [this] {
    sessionManager_.session().aiStemsEnabled = controlDeck_->getSeparatedStemsToggle().getToggleState();
  };
  controlDeck_->getBatchRecursiveToggle().onClick = [this] {
    const bool enabled = controlDeck_->getBatchRecursiveToggle().getToggleState();
    sessionManager_.session().batchRecursiveEnabled = enabled;
    saveBatchRecursivePreference(enabled);
    setBatchRecursiveEnvironment(enabled);
  };
}

// ─────────────────────────────────────────────────────────────────
// Wiring: HeroWaveform
// ─────────────────────────────────────────────────────────────────

void MainLayout::wireHeroWaveformCallbacks() {
  heroWaveform_->onSeek = [this](double progress) {
    transportController_.seekToFraction(std::clamp(progress, 0.0, 1.0));
  };
  heroWaveform_->onFilesDropped = [this](std::vector<juce::File> files) {
    importFiles(std::move(files));
  };
  heroWaveform_->onZoomChanged = [this](double zoomFactor) {
    sessionManager_.session().timeline.zoom = zoomFactor;
  };
  heroWaveform_->onPresetDropped = [this](juce::File presetFile) {
    taskOrchestrator_->appendHistory("Preset dropped: " + presetFile.getFullPathName());
  };
}

// ── Undo / Redo ─────────────────────────────────────────────────

void MainLayout::onUndo() {
  taskOrchestrator_->appendHistory("Undo: no undo history available");
}

void MainLayout::onRedo() {
  taskOrchestrator_->appendHistory("Redo: no redo history available");
}

// ── Header Profile Quick-Switch ───────────────────────────────

void MainLayout::onHeaderProfileSelected(const juce::String& profileIdStr) {
  if (profileIdStr.isEmpty())
    return;
  const auto targetId = profileIdStr.toStdString();
  auto& box = controlDeck_->getProfileBox();
  for (int i = 1; i <= box.getNumItems(); ++i) {
    const auto id = selectionState_.projectProfileIdForCombo(i);
    if (id.has_value() && *id == targetId) {
      box.setSelectedId(i, juce::sendNotification);
      return;
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// Action: Import
// ─────────────────────────────────────────────────────────────────

void MainLayout::onImport() {
  if (taskOrchestrator_->isTaskRunning()) {
    taskOrchestrator_->setStatus("Busy", "A task is already running");
    return;
  }

  importChooser_ = std::make_unique<juce::FileChooser>(
      "Select stem files", juce::File(), "*.wav;*.aiff;*.aif;*.flac;*.mp3;*.ogg");

  constexpr int flags = juce::FileBrowserComponent::openMode |
                        juce::FileBrowserComponent::canSelectFiles |
                        juce::FileBrowserComponent::canSelectMultipleItems;

  importChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto files = chooser.getResults();
    if (files.isEmpty()) {
      importChooser_.reset();
      return;
    }

    std::vector<juce::File> selectedFiles;
    selectedFiles.reserve(static_cast<size_t>(files.size()));
    for (int i = 0; i < files.size(); ++i)
      selectedFiles.push_back(files.getReference(i));

    importChooser_.reset();
    importFiles(std::move(selectedFiles));
  });
}

void MainLayout::importFiles(std::vector<juce::File> files) {
  if (taskOrchestrator_->isTaskRunning()) {
    taskOrchestrator_->setStatus("Busy", "A task is already running");
    return;
  }

  if (!taskOrchestrator_->beginTask(ActiveTask::Import, "Importing stems", "", "Import started"))
    return;

  std::optional<std::filesystem::path> separationModelRoot;
  const bool useSeparation = controlDeck_->getSeparatedStemsToggle().getToggleState();
  if (useSeparation) {
    if (files.size() != 1) {
      taskOrchestrator_->setStatus("AI stem separation", "Applies to one full-mix file; importing provided stems as-is");
    }
    const auto separationPack = resolveActiveModelPackForTask("separation");
    if (separationPack.has_value()) {
      separationModelRoot = separationPack->rootPath;
      taskOrchestrator_->setStatus("Importing with AI stem separation", juce::String(separationPack->name));
    } else {
      taskOrchestrator_->setStatus("AI stem separation enabled", "No separation pack active; using fallback splitter");
    }
  }

  importController_->importFiles(
      std::move(files),
      useSeparation,
      sessionManager_.session().preferredStemCount,
      taskOrchestrator_->cancelFlag(ActiveTask::Import),
      std::move(separationModelRoot));
}

bool MainLayout::startAiSeparationBeforeAutoMixIfNeeded() {
  if (skipNextAutoMixSeparationCheck_) {
    skipNextAutoMixSeparationCheck_ = false;
    return false;
  }

  if (!controlDeck_->getSeparatedStemsToggle().getToggleState()) {
    return false;
  }

  const auto& stems = sessionManager_.session().stems;
  if (stems.size() != 1) {
    return false;
  }

  const auto& onlyStem = stems.front();
  if (onlyStem.origin == domain::StemOrigin::Separated || onlyStem.filePath.empty()) {
    return false;
  }

  std::error_code error;
  const auto sourcePath = std::filesystem::path(onlyStem.filePath);
  if (!std::filesystem::is_regular_file(sourcePath, error) || error) {
    taskOrchestrator_->appendHistory(
        juce::String("AI stem separation skipped before Auto Mix: source file not accessible: ") +
        juce::String(onlyStem.filePath));
    return false;
  }

  pendingAutoMixAfterSeparationImport_ = true;
  taskOrchestrator_->appendHistory("AI stem separation enabled: splitting single full-mix track before Auto Mix.");
  importFiles({juce::File(sourcePath.string())});
  return true;
}

// ─────────────────────────────────────────────────────────────────
// Action: Auto Mix
// ─────────────────────────────────────────────────────────────────

void MainLayout::onAutoMix() {
  if (startAiSeparationBeforeAutoMixIfNeeded()) {
    return;
  }

  if (sessionManager_.session().stems.empty()) {
    taskOrchestrator_->setStatus("No stems", "Import stems first");
    return;
  }

  if (!taskOrchestrator_->beginTask(ActiveTask::AutoMix, "Auto Mix", "Running...", "Auto Mix started"))
    return;

  auto mixPack = resolveActiveModelPackForTask("mix");

  processingController_->runAutoMix(
      sessionManager_.session(), mixPack,
      taskOrchestrator_->cancelFlag(ActiveTask::AutoMix));
}

// ─────────────────────────────────────────────────────────────────
// Action: Auto Master
// ─────────────────────────────────────────────────────────────────

void MainLayout::onAutoMaster() {
  if (sessionManager_.session().stems.empty()) {
    taskOrchestrator_->setStatus("No stems", "Import stems first");
    return;
  }

  if (!taskOrchestrator_->beginTask(ActiveTask::AutoMaster, "Auto Master", "Running...", "Auto Master started"))
    return;

  auto preset = domain::MasterPreset::DefaultStreaming;
  const auto platformPreset = selectionState_.platformPresetForCombo(controlDeck_->getPlatformPresetBox().getSelectedId());
  if (platformPreset.has_value())
    preset = platformPreset.value();
  if (preset == domain::MasterPreset::Custom) {
    const auto masterPreset = selectionState_.masterPresetForCombo(controlDeck_->getMasterPresetBox().getSelectedId());
    if (masterPreset.has_value())
      preset = masterPreset.value();
  }

  auto settings = buildCurrentRenderSettings("");

  auto masterPack = resolveActiveModelPackForTask("master");

  processingController_->runAutoMaster(
      sessionManager_.session(),
      settings,
      preset,
      masterPack,
      taskOrchestrator_->cancelFlag(ActiveTask::AutoMaster));
}

// ─────────────────────────────────────────────────────────────────
// Action: Auto Mix + Master (one-click pipeline)
// ─────────────────────────────────────────────────────────────────

void MainLayout::onAutoMixMaster() {
  if (sessionManager_.session().stems.empty()) {
    taskOrchestrator_->setStatus("No stems", "Import stems first");
    return;
  }
  if (taskOrchestrator_->isTaskRunning()) {
    taskOrchestrator_->setStatus("Busy", "A task is already running");
    return;
  }

  autoMixMasterExportChooser_ = std::make_unique<juce::FileChooser>(
      "Select output folder for Auto Mix + Master",
      juce::File::getSpecialLocation(juce::File::userDocumentsDirectory));

  constexpr int flags = juce::FileBrowserComponent::openMode |
                        juce::FileBrowserComponent::canSelectDirectories;

  autoMixMasterExportChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    autoMixMasterExportChooser_.reset();
    if (selected == juce::File())
      return;

    pendingPipelineExportFolder_ = selected.getFullPathName().toStdString();
    taskOrchestrator_->appendHistory(
        "Pipeline: Auto Mix -> Auto Master -> Export to "
        + selected.getFullPathName());

    onAutoMix();
  });
}

void MainLayout::triggerPipelineExport() {
  auto folder = juce::File(juce::String(pendingPipelineExportFolder_));
  pendingPipelineExportFolder_.clear();

  const auto format = selectionState_.codecFormatForCombo(controlDeck_->getExportFormatBox().getSelectedId());
  const auto ext = format.has_value() ? juce::String(format.value()) : "wav";
  const auto songTitle = deriveSongTitleFromSession(sessionManager_.session());
  const auto outputFile = buildUniqueDatedExportFile(folder, songTitle, ext);

  auto settings = buildCurrentRenderSettings(outputFile.getFullPathName().toStdString());

  if (!taskOrchestrator_->beginTask(ActiveTask::Export, "Exporting",
                                    "", "Pipeline export: " + outputFile.getFullPathName()))
    return;

  exportVerificationSession_ = sessionManager_.session();
  exportVerificationSettings_ = settings;
  exportController_->runExport(
      sessionManager_.session(),
      settings,
      analysisEntries_,
      taskOrchestrator_->cancelFlag(ActiveTask::Export));
}

// ─────────────────────────────────────────────────────────────────
// Action: Batch
// ─────────────────────────────────────────────────────────────────

void MainLayout::onBatch() {
  if (taskOrchestrator_->isTaskRunning()) {
    taskOrchestrator_->setStatus("Busy", "A task is already running");
    return;
  }

  batchImportChooser_ = std::make_unique<juce::FileChooser>(
      "Select folder for batch processing", juce::File());

  constexpr int flags = juce::FileBrowserComponent::openMode |
                        juce::FileBrowserComponent::canSelectDirectories;

  batchImportChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      batchImportChooser_.reset();
      return;
    }

    if (!taskOrchestrator_->beginTask(ActiveTask::Batch,
                                      "Batch processing",
                                      "",
                                      "Batch started: " + selected.getFullPathName())) {
      batchImportChooser_.reset();
      return;
    }

    setBatchRecursiveEnvironment(controlDeck_->getBatchRecursiveToggle().getToggleState());
    const auto selectedPathUtf8 = selected.getFullPathName().toStdString();
    const auto selectedPath = util::pathFromUtf8(selectedPathUtf8);
    batchVerificationInputFolder_ = selectedPath;
    batchVerificationRecursiveScan_ = controlDeck_->getBatchRecursiveToggle().getToggleState();

    auto settings = buildCurrentRenderSettings("");
    batchVerificationSettings_ = settings;

    auto mixPack = resolveActiveModelPackForTask("mix");
    auto masterPack = resolveActiveModelPackForTask("master");

    processingController_->runBatch(
        selectedPath,
        settings,
        taskOrchestrator_->cancelFlag(ActiveTask::Batch),
        mixPack,
        masterPack);

    batchImportChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Export
// ─────────────────────────────────────────────────────────────────

void MainLayout::onExport() {
  if (taskOrchestrator_->isTaskRunning()) {
    taskOrchestrator_->setStatus("Busy", "A task is already running");
    return;
  }
  if (sessionManager_.session().stems.empty()) {
    taskOrchestrator_->setStatus("No stems", "Import stems first");
    return;
  }

  exportChooser_ = std::make_unique<juce::FileChooser>(
      "Export master",
      juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
      "*.wav;*.flac;*.aiff;*.ogg;*.mp3");

  constexpr int flags = juce::FileBrowserComponent::saveMode |
                        juce::FileBrowserComponent::canSelectFiles |
                        juce::FileBrowserComponent::warnAboutOverwriting;

  exportChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      exportChooser_.reset();
      return;
    }

    auto settings = buildCurrentRenderSettings(selected.getFullPathName().toStdString());

    if (!taskOrchestrator_->beginTask(ActiveTask::Export, "Exporting", "", "Export started: " + selected.getFullPathName())) {
      exportChooser_.reset();
      return;
    }

    exportVerificationSession_ = sessionManager_.session();
    exportVerificationSettings_ = settings;
    exportController_->runExport(
        sessionManager_.session(),
        settings,
        analysisEntries_,
        taskOrchestrator_->cancelFlag(ActiveTask::Export));
    exportChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Verification (delegates to VerificationEngine)
// ─────────────────────────────────────────────────────────────────

void MainLayout::startExportVerification(const std::string& outputAudioPath) {
  if (outputAudioPath.empty()) {
    return;
  }

  if (!exportVerificationSession_.has_value() || !exportVerificationSettings_.has_value()) {
    taskOrchestrator_->appendHistory("Export verification skipped: missing export context.");
    return;
  }

  VerificationEngine::ExportContext context;
  context.session = std::move(exportVerificationSession_.value());
  context.settings = std::move(exportVerificationSettings_.value());
  context.outputAudioPath = outputAudioPath;
  context.analysisPack = resolveActiveModelPackForTask("analysis");
  exportVerificationSession_.reset();
  exportVerificationSettings_.reset();

  auto safe = safeAsync(this);
  VerificationEngine::runExportVerification(
      std::move(context),
      backgroundPool_,
      [safe](const juce::String& text) {
        juce::MessageManager::callAsync([safe, text]() {
          if (safe) safe->taskOrchestrator_->appendHistory(text);
        });
      });
}

void MainLayout::startBatchVerification(const std::string& outputFolder) {
  if (outputFolder.empty()) {
    return;
  }
  if (!batchVerificationInputFolder_.has_value() || !batchVerificationSettings_.has_value()) {
    taskOrchestrator_->appendHistory("Batch verification skipped: missing batch context.");
    return;
  }

  VerificationEngine::BatchContext context;
  context.inputFolder = batchVerificationInputFolder_.value();
  context.outputFolder = util::pathFromUtf8(outputFolder);
  context.settings = batchVerificationSettings_.value();
  context.analysisPack = resolveActiveModelPackForTask("analysis");
  context.recursiveScan = batchVerificationRecursiveScan_;
  batchVerificationInputFolder_.reset();
  batchVerificationSettings_.reset();

  auto safe = safeAsync(this);
  VerificationEngine::runBatchVerification(
      std::move(context),
      backgroundPool_,
      [safe](const juce::String& text) {
        juce::MessageManager::callAsync([safe, text]() {
          if (safe) safe->taskOrchestrator_->appendHistory(text);
        });
      });
}

// ─────────────────────────────────────────────────────────────────
// Action: Save Session
// ─────────────────────────────────────────────────────────────────

void MainLayout::onSaveSession() {
  if (taskOrchestrator_->isTaskRunning()) {
    taskOrchestrator_->setStatus("Busy", "A task is already running");
    return;
  }

  saveSessionChooser_ = std::make_unique<juce::FileChooser>(
      "Save session", juce::File(), "*.json");

  constexpr int flags = juce::FileBrowserComponent::saveMode |
                        juce::FileBrowserComponent::canSelectFiles;

  saveSessionChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      saveSessionChooser_.reset();
      return;
    }

    if (!taskOrchestrator_->beginTask(ActiveTask::Session, "Saving session", "", "Session save started")) {
      saveSessionChooser_.reset();
      return;
    }

    syncSessionUiSelections();
    auto path = selected.getFullPathName().toStdString();
    sessionController_->saveSession(path, sessionManager_.session(),
                                    taskOrchestrator_->cancelFlag(ActiveTask::Session));
    saveSessionChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Load Session
// ─────────────────────────────────────────────────────────────────

void MainLayout::onLoadSession() {
  if (taskOrchestrator_->isTaskRunning()) {
    taskOrchestrator_->setStatus("Busy", "A task is already running");
    return;
  }

  loadSessionChooser_ = std::make_unique<juce::FileChooser>(
      "Load session", juce::File(), "*.json");

  constexpr int flags = juce::FileBrowserComponent::openMode |
                        juce::FileBrowserComponent::canSelectFiles;

  loadSessionChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      loadSessionChooser_.reset();
      return;
    }

    if (!taskOrchestrator_->beginTask(ActiveTask::Session, "Loading session", "", "Session load started")) {
      loadSessionChooser_.reset();
      return;
    }

    auto path = selected.getFullPathName().toStdString();
    sessionController_->loadSession(path, taskOrchestrator_->cancelFlag(ActiveTask::Session));
    loadSessionChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Models Dialog
// ─────────────────────────────────────────────────────────────────

void MainLayout::onModelsDialog() {
  auto* panel = new ModelBrowserPanel();
  modelBrowserPanel_ = panel;
  const auto& discovered = modelController_->discoveredModels();
  panel->setDiscoveredModels(discovered);
  panel->setInstalledPacks(modelManager_.availablePacks());
  panel->setInstalledModelIds(modelController_->installedModelIds());
  panel->setActivePackDisplay(activePackMapForUi(modelManager_));
  if (discovered.empty()) {
    panel->setStatus("Fetch catalog, install models, then click 'Use Selected for Task' to activate one");
  }
  panel->setSize(600, 500);

  panel->onFetchCatalog = [this] {
    const bool curatedOnly = modelBrowserPanel_ == nullptr ||
                             modelBrowserPanel_->isCuratedLockEnabled() ||
                             modelBrowserPanel_->isCuratedMode();
    const std::string searchText = (!curatedOnly && modelBrowserPanel_ != nullptr) ? modelBrowserPanel_->rawSearchQuery() : "";

    if (!taskOrchestrator_->beginTask(ActiveTask::Model, "Fetching model catalog", "", "Model catalog fetch started"))
      return;
    if (modelBrowserPanel_ != nullptr) {
      modelBrowserPanel_->setActionsEnabled(false);
      modelBrowserPanel_->setStatus(curatedOnly ? "Fetching curated model catalog..." : "Searching raw model catalog...");
    }
    modelController_->fetchCatalog(taskOrchestrator_->cancelFlag(ActiveTask::Model), curatedOnly, searchText);
  };
  panel->onInstallModel = [this](const std::string& modelId) {
    if (modelController_ != nullptr &&
        modelController_->modelRequiresLicenseConsent(modelId) &&
        !modelController_->hasModelLicenseConsent(modelId)) {
      modelController_->acknowledgeModelLicenseConsent(modelId);
    }
    if (!taskOrchestrator_->beginTask(ActiveTask::Model, "Installing model", juce::String(modelId),
                                      "Model install started: " + juce::String(modelId)))
      return;
    if (modelBrowserPanel_ != nullptr) {
      modelBrowserPanel_->setActionsEnabled(false);
      modelBrowserPanel_->setStatus("Installing " + juce::String(modelId) + "...");
    }
    modelController_->installModel(modelId, taskOrchestrator_->cancelFlag(ActiveTask::Model));
  };
  panel->onUninstallModel = [this](const std::string& modelId) {
    if (!taskOrchestrator_->beginTask(ActiveTask::Model, "Uninstalling model", juce::String(modelId),
                                      "Model uninstall started: " + juce::String(modelId)))
      return;
    if (modelBrowserPanel_ != nullptr) {
      modelBrowserPanel_->setActionsEnabled(false);
      modelBrowserPanel_->setStatus("Uninstalling " + juce::String(modelId) + "...");
    }
    modelController_->uninstallModel(modelId, taskOrchestrator_->cancelFlag(ActiveTask::Model));
  };
  panel->onUseInstalledModel = [this](const std::string& modelId) {
    if (modelBrowserPanel_ == nullptr) {
      return;
    }
    if (modelController_ != nullptr &&
        modelController_->modelRequiresLicenseConsent(modelId) &&
        !modelController_->hasModelLicenseConsent(modelId)) {
      modelController_->acknowledgeModelLicenseConsent(modelId);
    }
    const auto taskScope = modelBrowserPanel_->selectedTaskScope();
    const bool activated = modelController_->activateInstalledModelForTask(modelId, taskScope);
    if (activated) {
      refreshModelPacks();
      modelBrowserPanel_->setActivePackDisplay(activePackMapForUi(modelManager_));
      modelBrowserPanel_->setStatus("Active " + juce::String(taskScope) + " model updated");
    } else {
      modelBrowserPanel_->setStatus("Could not activate selected model for task");
    }
  };
  panel->onSetActivePack = [this](const std::string& taskScope, const std::string& packId) {
    modelManager_.setActivePackId(taskScope, packId);
    updateSeparationModelBadge();
    taskOrchestrator_->appendHistory("Active " + juce::String(taskScope) + " pack set to " + juce::String(packId));
    if (modelBrowserPanel_ != nullptr) {
      modelBrowserPanel_->setActivePackDisplay(activePackMapForUi(modelManager_));
      modelBrowserPanel_->setInstalledPacks(modelManager_.availablePacks());
      modelBrowserPanel_->setStatus("Active " + juce::String(taskScope) + " model set to " + juce::String(packId));
    }
  };
  panel->onCheckUpdates = [this] {
    if (!taskOrchestrator_->beginTask(ActiveTask::Model, "Checking model updates", "", "Model update check started"))
      return;
    if (modelBrowserPanel_ != nullptr) {
      modelBrowserPanel_->setActionsEnabled(false);
      modelBrowserPanel_->setStatus("Checking model updates...");
    }
    modelController_->checkUpdates(taskOrchestrator_->cancelFlag(ActiveTask::Model));
  };
  panel->onVerifyIntegrity = [this] {
    modelController_->verifyIntegrity();
  };

  juce::DialogWindow::LaunchOptions options;
  options.content.setOwned(panel);
  options.dialogTitle = "Model Browser";
  options.dialogBackgroundColour = colour(colours::surface);
  options.escapeKeyTriggersCloseButton = true;
  options.useNativeTitleBar = true;
  options.resizable = true;
  options.launchAsync();
  taskOrchestrator_->appendHistory("Model browser opened");
}

// ─────────────────────────────────────────────────────────────────
// Action: Settings
// ─────────────────────────────────────────────────────────────────

void MainLayout::onSettings() {
  auto* settingsPanel = new SettingsPanel(
      audioDeviceManager_,
      sessionManager_.session().renderSettings.writePerExportReportJson,
      [this](const bool enabled) {
        sessionManager_.session().renderSettings.writePerExportReportJson = enabled;
        saveExportReportSidecarPreference(enabled);
        taskOrchestrator_->appendHistory(enabled
                                             ? "Export sidecar JSON enabled (.report.json)"
                                             : "Export sidecar JSON disabled");
      });
  settingsPanel->setSize(540, 430);

  juce::DialogWindow::LaunchOptions options;
  options.content.setOwned(settingsPanel);
  options.dialogTitle = "Settings";
  options.dialogBackgroundColour = colour(colours::surface);
  options.escapeKeyTriggersCloseButton = true;
  options.useNativeTitleBar = true;
  options.resizable = false;
  options.launchAsync();
  taskOrchestrator_->appendHistory("Settings dialog opened");
}

// ─────────────────────────────────────────────────────────────────
// UI Update: Transport Display
// ─────────────────────────────────────────────────────────────────

void MainLayout::updateTransportDisplay() {
  double current = transportController_.positionSeconds();
  double total = transportController_.totalSeconds();
  double progress = transportController_.progress();

  transportBar_->setTimeDisplay(current, total);
  heroWaveform_->setPlayheadProgress(progress);
}

// ─────────────────────────────────────────────────────────────────
// UI Update: Preview
// ─────────────────────────────────────────────────────────────────

void MainLayout::rebuildPreview() {
  if (sessionManager_.session().stems.empty())
    return;

  previewManager_->rebuildPreview(
      sessionManager_.session(),
      controlDeck_->getStemPanel().getStemDisplays(),
      transportController_.progress());
  taskOrchestrator_->appendHistory("Preview rebuild started");
}

// ─────────────────────────────────────────────────────────────────
// UI Update: Misc
// ─────────────────────────────────────────────────────────────────

void MainLayout::updateTransportFromBuffer(const engine::AudioBuffer& buffer) {
  if (buffer.getNumSamples() <= 0)
    return;

  const auto& session = sessionManager_.session();
  transportController_.setTimeline(
      static_cast<int64_t>(buffer.getNumSamples()),
      buffer.getSampleRate());
  transportController_.setLoopRangeSeconds(session.timeline.loopInSeconds,
                                           session.timeline.loopOutSeconds,
                                           session.timeline.loopEnabled);
  transportBar_->setLoopEnabled(session.timeline.loopEnabled);
  heroWaveform_->setLoopRange(session.timeline.loopEnabled,
                              transportController_.loopInProgress(),
                              transportController_.loopOutProgress());
  heroWaveform_->setZoom(session.timeline.zoom, 0.5);
}

void MainLayout::updateMeterPanel(const automaster::MasteringReport& report) {
  auto& meters = controlDeck_->getGlowMeters();
  meters.setLufs(report.integratedLufs, report.shortTermLufs);
  meters.setTruePeak(report.truePeakDbtp);
  meters.setLevels(static_cast<float>(report.samplePeakDbfs),
                   static_cast<float>(report.samplePeakDbfs));
  meters.setPeaks(static_cast<float>(report.truePeakDbtp),
                  static_cast<float>(report.truePeakDbtp));
}

void MainLayout::applyLoadedSession(domain::Session loadedSession, const juce::String& sourcePath) {
  sessionManager_.replaceSession(std::move(loadedSession));
  const auto& session = sessionManager_.session();
  headerBar_->setSessionName(juce::File(sourcePath).getFileNameWithoutExtension());
  updateStemPanelFromSession(controlDeck_->getStemPanel(), session);
  transportBar_->setLoopEnabled(session.timeline.loopEnabled);
  heroWaveform_->setZoom(session.timeline.zoom, 0.5);
  refreshRenderers();
  refreshCodecAvailability();
  refreshModelPacks();
  populateMasterPresetSelectors();
  refreshProjectProfiles();
  applySessionUiSelections();
  syncSessionUiSelections();
  taskOrchestrator_->appendHistory("Session loaded: " + sourcePath);
  rebuildPreview();
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Renderers
// ─────────────────────────────────────────────────────────────────

void MainLayout::refreshRenderers() {
  auto& box = controlDeck_->getRendererBox();
  box.clear(juce::dontSendNotification);
  selectionState_.clearRendererIds();

  renderers::RendererRegistry registry;
  rendererInfos_ = registry.list(loadConfiguredExternalRenderers());

  int comboId = 1;
  int preferredId = 0;
  for (const auto& info : rendererInfos_) {
    if (!info.available) {
      continue;
    }

    box.addItem(info.name, comboId);
    selectionState_.bindRendererId(comboId, info.id);
    if (preferredId == 0)
      preferredId = comboId;
    if (info.id == sessionManager_.session().renderSettings.rendererName)
      preferredId = comboId;
    ++comboId;
  }
  if (preferredId != 0)
    box.setSelectedId(preferredId, juce::dontSendNotification);

  updateRendererChainPreview();
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Codec/Format
// ─────────────────────────────────────────────────────────────────

void MainLayout::refreshCodecAvailability() {
  auto& box = controlDeck_->getExportFormatBox();
  box.clear(juce::dontSendNotification);
  selectionState_.clearCodecFormats();

  struct FormatEntry { std::string id; juce::String label; };
  std::vector<FormatEntry> formats = {
      {"wav", "WAV"}, {"flac", "FLAC"}, {"aiff", "AIFF"}, {"ogg", "OGG"}, {"mp3", "MP3"}};

  int comboId = 1;
  for (const auto& fmt : formats) {
    box.addItem(fmt.label, comboId);
    selectionState_.bindCodecFormat(comboId, fmt.id);
    ++comboId;
  }
  box.setSelectedId(1, juce::dontSendNotification);

  auto& modeBox = controlDeck_->getExportModeBox();
  modeBox.clear(juce::dontSendNotification);
  selectionState_.clearExportSpeedModes();
  modeBox.addItem("Final", 1);
  selectionState_.bindExportSpeedMode(1, "final");
  modeBox.addItem("Quick Preview", 2);
  selectionState_.bindExportSpeedMode(2, "quick");
  modeBox.setSelectedId(1, juce::dontSendNotification);

  auto& chainModeBox = controlDeck_->getRendererChainModeBox();
  chainModeBox.clear(juce::dontSendNotification);
  selectionState_.clearRendererChainModes();
  chainModeBox.addItem("Logical All", 1);
  selectionState_.bindRendererChainMode(1, "logical_all");
  chainModeBox.addItem("Master + rsgain", 2);
  selectionState_.bindRendererChainMode(2, "master_then_rsgain");
  chainModeBox.setSelectedId(1, juce::dontSendNotification);

  updateRendererChainPreview();
}

void MainLayout::updateRendererChainPreview() {
  domain::RenderSettings settings = sessionManager_.session().renderSettings;

  if (const auto rendererId = selectionState_.rendererIdForCombo(controlDeck_->getRendererBox().getSelectedId());
      rendererId.has_value()) {
    settings.rendererName = rendererId.value();
  }

  settings.rendererChainEnabled = controlDeck_->getRendererChainToggle().getToggleState();
  if (const auto chainMode =
          selectionState_.rendererChainModeForCombo(controlDeck_->getRendererChainModeBox().getSelectedId());
      chainMode.has_value()) {
    settings.rendererChainMode = chainMode.value();
  }

  const auto chain = renderers::resolveRendererChain(settings);
  controlDeck_->setRendererChainPreviewText(toChainPreviewText(chain));
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Model Packs
// ─────────────────────────────────────────────────────────────────

void MainLayout::refreshModelPacks() {
  modelManager_.setRootPaths({std::filesystem::path("ModelPacks"), std::filesystem::path("assets/modelhub")});
  const auto packs = modelManager_.scan();

  const auto pickDefaultForScope = [&](const std::string& scope) -> std::optional<std::string> {
    const auto normalizedScope = util::toLower(scope);
    const auto isPreferredSeparationPack = [](const ai::ModelPack& pack) {
      const auto id = util::toLower(pack.id);
      const auto name = util::toLower(pack.name);
      const auto modelFile = util::toLower(pack.modelFile);
      return id.find("htdemucs_6s") != std::string::npos ||
             name.find("htdemucs_6s") != std::string::npos ||
             modelFile.find("htdemucs_6s") != std::string::npos;
    };

    if (normalizedScope == "separation") {
      for (const auto& pack : packs) {
        if (util::toLower(pack.taskScope) != normalizedScope) {
          continue;
        }
        if (!isPreferredSeparationPack(pack)) {
          continue;
        }
        if (util::toLower(pack.rootPath.string()).find("modelhub") != std::string::npos) {
          return pack.id;
        }
      }

      for (const auto& pack : packs) {
        if (util::toLower(pack.taskScope) != normalizedScope) {
          continue;
        }
        if (isPreferredSeparationPack(pack)) {
          return pack.id;
        }
      }
    }

    for (const auto& pack : packs) {
      if (util::toLower(pack.taskScope) != normalizedScope) {
        continue;
      }
      if (util::toLower(pack.rootPath.string()).find("modelhub") != std::string::npos) {
        return pack.id;
      }
    }

    for (const auto& pack : packs) {
      if (util::toLower(pack.taskScope) == normalizedScope) {
        return pack.id;
      }
    }
    return std::nullopt;
  };

  for (const auto* scope : {"mix", "master", "analysis", "separation"}) {
    const auto activeId = modelManager_.activePackId(scope);
    const bool activeExists = !activeId.empty() &&
                              std::any_of(packs.begin(), packs.end(), [&](const ai::ModelPack& pack) {
                                return pack.id == activeId && util::toLower(pack.taskScope) == scope;
                              });
    if (activeExists) {
      continue;
    }

    const auto selected = pickDefaultForScope(scope);
    modelManager_.setActivePackId(scope, selected.value_or(""));
    if (selected.has_value() && taskOrchestrator_ != nullptr) {
      taskOrchestrator_->appendHistory("Model pack active for " + juce::String(scope) + ": " + juce::String(*selected));
    }
  }

  updateSeparationModelBadge();
}

void MainLayout::updateSeparationModelBadge() {
  if (controlDeck_ == nullptr) {
    return;
  }

  const auto activeId = modelManager_.activePackId("separation");
  if (activeId.empty()) {
    controlDeck_->setSeparationModelStatus("Separation model: none", false);
    return;
  }

  const auto& packs = modelManager_.availablePacks();
  const auto selected = std::find_if(packs.begin(), packs.end(), [&](const ai::ModelPack& pack) {
    return util::toLower(pack.taskScope) == "separation" && pack.id == activeId;
  });
  if (selected == packs.end()) {
    controlDeck_->setSeparationModelStatus("Separation model: none", false);
    return;
  }

  std::error_code error;
  const auto modelPath = selected->rootPath / selected->modelFile;
  const bool ready = std::filesystem::is_regular_file(modelPath, error) && !error;
  const auto displayName = selected->name.empty() ? selected->id : selected->name;
  auto badgeText = juce::String("Separation model: ") + juce::String(displayName);
  if (!ready) {
    badgeText << " (missing)";
  }
  controlDeck_->setSeparationModelStatus(badgeText, ready);
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Master Presets
// ─────────────────────────────────────────────────────────────────

void MainLayout::populateMasterPresetSelectors() {
  auto& masterBox = controlDeck_->getMasterPresetBox();
  auto& platformBox = controlDeck_->getPlatformPresetBox();
  masterBox.clear(juce::dontSendNotification);
  platformBox.clear(juce::dontSendNotification);
  selectionState_.clearMasterPresets();
  selectionState_.clearPlatformPresets();

  int mid = 1;
  auto addMaster = [&](const juce::String& label, domain::MasterPreset p) {
    masterBox.addItem(label, mid);
    selectionState_.bindMasterPreset(mid, p);
    ++mid;
  };
  addMaster("Default Streaming", domain::MasterPreset::DefaultStreaming);
  addMaster("Broadcast", domain::MasterPreset::Broadcast);
  addMaster("Udio Optimized", domain::MasterPreset::UdioOptimized);
  addMaster("Custom", domain::MasterPreset::Custom);

  int pid = 1;
  auto addPlatform = [&](const juce::String& label, domain::MasterPreset p) {
    platformBox.addItem(label, pid);
    selectionState_.bindPlatformPreset(pid, p);
    ++pid;
  };
  addPlatform("Spotify", domain::MasterPreset::Spotify);
  addPlatform("Apple Music", domain::MasterPreset::AppleMusic);
  addPlatform("YouTube", domain::MasterPreset::YouTube);
  addPlatform("Amazon Music", domain::MasterPreset::AmazonMusic);
  addPlatform("Tidal", domain::MasterPreset::Tidal);
  addPlatform("Broadcast EBU R128", domain::MasterPreset::BroadcastEbuR128);
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Project Profiles
// ─────────────────────────────────────────────────────────────────

void MainLayout::refreshProjectProfiles() {
  projectProfiles_ = domain::loadProjectProfiles(std::filesystem::current_path());
  auto& box = controlDeck_->getProfileBox();
  box.clear(juce::dontSendNotification);
  selectionState_.clearProjectProfileIds();

  int selectedId = 0;
  int comboId = 1;
  for (const auto& profile : projectProfiles_) {
    box.addItem(profile.name + " [" + profile.id + "]", comboId);
    selectionState_.bindProjectProfileId(comboId, profile.id);
    if (profile.id == sessionManager_.session().projectProfileId)
      selectedId = comboId;
    ++comboId;
  }
  if (selectedId == 0)
    selectedId = selectionState_.firstProjectProfileComboId();
  if (selectedId > 0)
    box.setSelectedId(selectedId, juce::dontSendNotification);
}

void MainLayout::applySessionUiSelections() {
  auto& session = sessionManager_.session();

  if (const auto rendererId = selectionState_.comboIdForRendererId(session.renderSettings.rendererName); rendererId.has_value()) {
    controlDeck_->getRendererBox().setSelectedId(rendererId.value(), juce::dontSendNotification);
  }

  if (const auto profileId = selectionState_.comboIdForProjectProfileId(session.projectProfileId); profileId.has_value()) {
    controlDeck_->getProfileBox().setSelectedId(profileId.value(), juce::dontSendNotification);
  }

  if (const auto formatId = selectionState_.comboIdForCodecFormat(session.renderSettings.outputFormat); formatId.has_value()) {
    controlDeck_->getExportFormatBox().setSelectedId(formatId.value(), juce::dontSendNotification);
  }

  if (const auto modeId = selectionState_.comboIdForExportSpeedMode(session.renderSettings.exportSpeedMode); modeId.has_value()) {
    controlDeck_->getExportModeBox().setSelectedId(modeId.value(), juce::dontSendNotification);
  }
  if (const auto chainModeId =
          selectionState_.comboIdForRendererChainMode(session.renderSettings.rendererChainMode);
      chainModeId.has_value()) {
    controlDeck_->getRendererChainModeBox().setSelectedId(chainModeId.value(), juce::dontSendNotification);
  }

  if (const auto masterId = selectionState_.comboIdForMasterPreset(session.selectedMasterPreset); masterId.has_value()) {
    controlDeck_->getMasterPresetBox().setSelectedId(masterId.value(), juce::dontSendNotification);
  }

  if (const auto platformId = selectionState_.comboIdForPlatformPreset(session.selectedPlatformPreset); platformId.has_value()) {
    controlDeck_->getPlatformPresetBox().setSelectedId(platformId.value(), juce::dontSendNotification);
  }

  controlDeck_->getResidualBlendSlider().setValue(std::clamp(session.residualBlend, 0.0, 10.0), juce::dontSendNotification);
  controlDeck_->getSeparatedStemsToggle().setToggleState(session.aiStemsEnabled, juce::dontSendNotification);
  controlDeck_->getBatchRecursiveToggle().setToggleState(session.batchRecursiveEnabled, juce::dontSendNotification);
  controlDeck_->getRendererChainToggle().setToggleState(
      session.renderSettings.rendererChainEnabled,
      juce::dontSendNotification);
  controlDeck_->getRendererChainModeBox().setEnabled(session.renderSettings.rendererChainEnabled);
  setBatchRecursiveEnvironment(session.batchRecursiveEnabled);
  updateRendererChainPreview();
}

void MainLayout::syncSessionUiSelections() {
  auto& session = sessionManager_.session();
  session.residualBlend = controlDeck_->getResidualBlendSlider().getValue();
  session.aiStemsEnabled = controlDeck_->getSeparatedStemsToggle().getToggleState();
  session.batchRecursiveEnabled = controlDeck_->getBatchRecursiveToggle().getToggleState();

  if (const auto renderer = selectionState_.rendererIdForCombo(controlDeck_->getRendererBox().getSelectedId()); renderer.has_value()) {
    session.renderSettings.rendererName = renderer.value();
  }
  if (const auto profile = selectionState_.projectProfileIdForCombo(controlDeck_->getProfileBox().getSelectedId()); profile.has_value()) {
    session.projectProfileId = profile.value();
  }
  if (const auto format = selectionState_.codecFormatForCombo(controlDeck_->getExportFormatBox().getSelectedId()); format.has_value()) {
    session.renderSettings.outputFormat = format.value();
  }
  if (const auto mode = selectionState_.exportSpeedModeForCombo(controlDeck_->getExportModeBox().getSelectedId()); mode.has_value()) {
    session.renderSettings.exportSpeedMode = mode.value();
  }
  session.renderSettings.rendererChainEnabled = controlDeck_->getRendererChainToggle().getToggleState();
  if (const auto chainMode = selectionState_.rendererChainModeForCombo(controlDeck_->getRendererChainModeBox().getSelectedId());
      chainMode.has_value()) {
    session.renderSettings.rendererChainMode = chainMode.value();
  }
  if (const auto masterPreset = selectionState_.masterPresetForCombo(controlDeck_->getMasterPresetBox().getSelectedId());
      masterPreset.has_value()) {
    session.selectedMasterPreset = masterPreset.value();
  }
  if (const auto platformPreset =
          selectionState_.platformPresetForCombo(controlDeck_->getPlatformPresetBox().getSelectedId());
      platformPreset.has_value()) {
    session.selectedPlatformPreset = platformPreset.value();
  }
  updateRendererChainPreview();
}

// ─────────────────────────────────────────────────────────────────
// Query: Build Render Settings
// ─────────────────────────────────────────────────────────────────

std::optional<ai::ModelPack> MainLayout::resolveActiveModelPackForTask(const std::string& taskScope) {
  const auto activeId = modelManager_.activePackId(taskScope);
  if (activeId.empty() || activeId == "none") {
    return std::nullopt;
  }

  auto selected = std::find_if(
      modelManager_.availablePacks().begin(),
      modelManager_.availablePacks().end(),
      [&](const ai::ModelPack& pack) { return pack.id == activeId; });
  if (selected == modelManager_.availablePacks().end()) {
    taskOrchestrator_->appendHistory("Model pack missing for task '" + juce::String(taskScope) + "': " + juce::String(activeId));
    return std::nullopt;
  }

  const std::string requiredType = taskScope == "mix" ? "mix_parameters"
                                   : taskScope == "master" ? "master_parameters"
                                                             : "";
  if (!requiredType.empty() && util::toLower(selected->type) != requiredType) {
    taskOrchestrator_->setStatus("Model pack blocked", "Incompatible model type for " + juce::String(taskScope));
    taskOrchestrator_->appendHistory("Model pack rejected for task '" + juce::String(taskScope) + "': " +
                                     juce::String(selected->id) + " type=" + juce::String(selected->type));
    return std::nullopt;
  }

  if (!selected->taskScope.empty() && util::toLower(selected->taskScope) != util::toLower(taskScope)) {
    taskOrchestrator_->setStatus("Model pack blocked", "Task scope mismatch for selected model pack");
    taskOrchestrator_->appendHistory("Model pack rejected for task '" + juce::String(taskScope) + "': " +
                                     juce::String(selected->id) + " task_scope=" + juce::String(selected->taskScope));
    return std::nullopt;
  }

  const auto modelPath = selected->rootPath / selected->modelFile;
  std::error_code error;
  if (!std::filesystem::is_regular_file(modelPath, error) || error) {
    taskOrchestrator_->setStatus("Model pack blocked", "Model file missing for selected pack");
    taskOrchestrator_->appendHistory("Model pack rejected for task '" + juce::String(taskScope) + "': missing model file " +
                                     juce::String(modelPath.string()));
    return std::nullopt;
  }

  const auto modelExtension = util::toLower(modelPath.extension().string());
  if ((taskScope == "mix" || taskScope == "master") && modelExtension != ".onnx") {
    taskOrchestrator_->setStatus("Model pack blocked", "Mix/Master packs must use ONNX models");
    taskOrchestrator_->appendHistory("Model pack rejected for task '" + juce::String(taskScope) + "': non-ONNX model " +
                                     juce::String(modelPath.filename().string()));
    return std::nullopt;
  }

  return *selected;
}

domain::RenderSettings MainLayout::buildCurrentRenderSettings(const std::string& outputPath) const {
  domain::RenderSettings settings = sessionManager_.session().renderSettings;
  settings.outputPath = outputPath;

  const auto format = selectionState_.codecFormatForCombo(controlDeck_->getExportFormatBox().getSelectedId());
  if (format.has_value())
    settings.outputFormat = format.value();

  const auto mode = selectionState_.exportSpeedModeForCombo(controlDeck_->getExportModeBox().getSelectedId());
  if (mode.has_value())
    settings.exportSpeedMode = mode.value();

  const auto renderer = selectionState_.rendererIdForCombo(controlDeck_->getRendererBox().getSelectedId());
  if (renderer.has_value())
    settings.rendererName = renderer.value();
  settings.rendererChainEnabled = controlDeck_->getRendererChainToggle().getToggleState();
  const auto chainMode = selectionState_.rendererChainModeForCombo(controlDeck_->getRendererChainModeBox().getSelectedId());
  if (chainMode.has_value()) {
    settings.rendererChainMode = chainMode.value();
  }
  settings.rendererChain.clear();

  return settings;
}

std::vector<renderers::ExternalRendererConfig> MainLayout::loadConfiguredExternalRenderers() {
  auto onError = [this](const juce::String& msg) { taskOrchestrator_->appendHistory(msg); };
  return automix::app::detail::loadConfiguredExternalRenderers(onError);
}

} // namespace automix::app
