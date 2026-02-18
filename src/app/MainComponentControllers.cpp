#include "app/MainComponent.h"

#include "util/StringUtils.h"

namespace automix::app {
namespace {

using ::automix::util::toLower;
using ::automix::util::toJuceText;

juce::String modelLabel(const ai::HubModelInfo& model) {
  juce::String label = model.repoId + " [" + model.useCase + "]";
  label += " dls=" + juce::String(model.downloads);
  if (model.recommended) {
    label += " *";
  }
  return label;
}

} // namespace

void MainComponent::initializeControllers() {
  const auto makeStatusCallback = [](juce::Component::SafePointer<MainComponent> safeMain)
      -> std::function<void(const std::string&)> {
    return [safeMain](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = safeMain, msg]() {
        if (safeThis != nullptr) {
          safeThis->statusLabel_.setText(juce::String(msg), juce::dontSendNotification);
        }
      });
    };
  };

  const auto makeTaskHistoryCallback = [](juce::Component::SafePointer<MainComponent> safeMain)
      -> std::function<void(const std::string&)> {
    return [safeMain](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = safeMain, msg]() {
        if (safeThis != nullptr) {
          safeThis->appendTaskHistory(juce::String(msg));
        }
      });
    };
  };

  {
    ModelController::Callbacks modelCallbacks;
    const auto safeMain = juce::Component::SafePointer<MainComponent>(this);
    modelCallbacks.onStatus = makeStatusCallback(safeMain);
    modelCallbacks.onTaskHistory = makeTaskHistoryCallback(safeMain);
    modelCallbacks.onReport = [safeMain](const std::string& text) {
      juce::MessageManager::callAsync([safeThis = safeMain, text]() {
        if (safeThis != nullptr) {
          safeThis->reportEditor_.setText(juce::String(text));
        }
      });
    };
    modelCallbacks.onModelPacksChanged = [safeMain]() {
      juce::MessageManager::callAsync([safeThis = safeMain]() {
        if (safeThis != nullptr) {
          safeThis->refreshModelPacks();
        }
      });
    };
    modelCallbacks.onCatalogReady = [safeMain]() {
      if (safeMain == nullptr || safeMain->modelController_ == nullptr) {
        return;
      }

      const auto& models = safeMain->modelController_->discoveredModels();
      if (models.empty()) {
        return;
      }

      juce::PopupMenu modelMenu;
      int itemId = 1000;
      for (const auto& model : models) {
        modelMenu.addItem(itemId++, modelLabel(model));
      }
      modelMenu.addSeparator();
      modelMenu.addItem(1900, "Refresh");
      const auto safeThis = safeMain;

      modelMenu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&safeMain->modelsMenuButton_),
                              [safeThis](const int selection) {
                                if (safeThis == nullptr) {
                                  return;
                                }
                                if (selection == 1900) {
                                  if (safeThis->taskRunning_.load()) {
                                    safeThis->statusLabel_.setText("Another background task is running", juce::dontSendNotification);
                                    return;
                                  }
                                  safeThis->beginCancelableTask("Models: fetching Hugging Face catalog...",
                                                                "Models catalog fetch started",
                                                                ActiveTask::Model);
                                  safeThis->modelController_->fetchCatalog(safeThis->cancelModel_);
                                  return;
                                }
                                const auto& discovered = safeThis->modelController_->discoveredModels();
                                if (selection < 1000 ||
                                    selection >= 1000 + static_cast<int>(discovered.size())) {
                                  return;
                                }
                                const auto& model = discovered[static_cast<size_t>(selection - 1000)];
                                if (safeThis->taskRunning_.load()) {
                                  safeThis->statusLabel_.setText("Another background task is running", juce::dontSendNotification);
                                  return;
                                }
                                safeThis->beginCancelableTask("Models: installing " + model.repoId,
                                                              "Model install started: " + model.repoId,
                                                              ActiveTask::Model);
                                safeThis->modelController_->installModel(model.repoId, safeThis->cancelModel_);
                              });
    };
    modelCallbacks.onRevealModelHubFolder = [safeMain](const std::filesystem::path& folderPath) {
      juce::MessageManager::callAsync([safeThis = safeMain, folderPath]() {
        if (safeThis == nullptr) {
          return;
        }
        juce::File(folderPath.string()).revealToUser();
      });
    };
    modelCallbacks.onAsyncTaskComplete = [safeMain]() {
      juce::MessageManager::callAsync([safeThis = safeMain]() {
        if (safeThis != nullptr) {
          safeThis->finishCancelableTask();
        }
      });
    };
    modelController_ = std::make_unique<ModelController>(modelManager_, backgroundPool_, std::move(modelCallbacks));
  }

  {
    ImportController::Callbacks importCallbacks;
    const auto safeMain = juce::Component::SafePointer<MainComponent>(this);
    importCallbacks.onStatus = makeStatusCallback(safeMain);
    importCallbacks.onTaskHistory = makeTaskHistoryCallback(safeMain);
    importCallbacks.onImportComplete = [safeMain](ImportResult result) {
      if (safeMain == nullptr) {
        return;
      }
      safeMain->finishCancelableTask();
      if (result.cancelled) {
        safeMain->statusLabel_.setText("Import cancelled", juce::dontSendNotification);
        safeMain->appendTaskHistory("Import cancelled");
        return;
      }
      safeMain->session_.stems = std::move(result.stems);
      safeMain->statusLabel_.setText(
          "Imported " + juce::String(static_cast<int>(safeMain->session_.stems.size())) + " stems",
          juce::dontSendNotification);
      safeMain->appendTaskHistory("Imported " + juce::String(static_cast<int>(safeMain->session_.stems.size())) + " stems");

      safeMain->analysisEntries_.clear();
      safeMain->analysisTableModel_.setEntries(&safeMain->analysisEntries_);
      safeMain->analysisTable_.updateContent();

      safeMain->refreshStemRoutingSelectors();
      safeMain->reportEditor_.setText(juce::String("Imported files:\n") + toJuceText(result.logLines));
      safeMain->rebuildPreviewBuffersAsync();
    };
    importController_ = std::make_unique<ImportController>(backgroundPool_, std::move(importCallbacks));
  }

  {
    ExportController::Callbacks exportCallbacks;
    const auto safeMain = juce::Component::SafePointer<MainComponent>(this);
    exportCallbacks.onStatus = makeStatusCallback(safeMain);
    exportCallbacks.onTaskHistory = makeTaskHistoryCallback(safeMain);
    exportCallbacks.onExportComplete = [safeMain](ExportResult result) {
      if (safeMain == nullptr) {
        return;
      }

      safeMain->finishCancelableTask();

      if (!result.analysisEntries.empty()) {
        safeMain->analysisEntries_ = std::move(result.analysisEntries);
        safeMain->analysisTableModel_.setEntries(&safeMain->analysisEntries_);
        safeMain->analysisTable_.updateContent();
      }

      if (!result.crashMessage.isEmpty()) {
        safeMain->statusLabel_.setText("Export crashed", juce::dontSendNotification);
        safeMain->reportEditor_.setText(result.crashMessage);
        safeMain->appendTaskHistory("Export crashed");
        return;
      }

      if (result.cancelled) {
        safeMain->statusLabel_.setText("Export cancelled", juce::dontSendNotification);
        safeMain->appendTaskHistory("Export cancelled");
        return;
      }

      const bool quickExportMode = toLower(result.exportSpeedMode) == "quick";
      if (quickExportMode) {
        safeMain->appendTaskHistory("Quick export mode active: stem-health preflight skipped");
      } else if (result.healthIssueCount > 0) {
        safeMain->appendTaskHistory(
            "Stem health check found " + juce::String(static_cast<int>(result.healthIssueCount)) + " issue(s)");
      } else {
        safeMain->appendTaskHistory("Stem health check passed");
      }

      safeMain->statusLabel_.setText(result.success ? "Export complete" : "Export failed",
                                     juce::dontSendNotification);
      if (result.healthHasCriticalIssues && result.success) {
        safeMain->statusLabel_.setText("Export complete with critical stem health warnings", juce::dontSendNotification);
      }
      safeMain->appendTaskHistory(result.success ? "Export completed" : "Export failed");
      juce::String report = juce::String("Renderer: ") + juce::String(result.rendererName) +
                            juce::String("\nExport mode: ") + juce::String(result.exportSpeedMode) +
                            juce::String("\nOutput: ") + juce::String(result.outputAudioPath) +
                            juce::String("\nReport: ") + juce::String(result.reportPath) +
                            juce::String("\n\nLogs:\n") + toJuceText(result.logs);
      if (!result.healthText.isEmpty()) {
        report += "\n\n";
        report += result.healthText;
      }
      safeMain->reportEditor_.setText(report);
    };
    exportCallbacks.onExternalRendererValidated = [safeMain](ExternalRendererValidationResult result) {
      if (safeMain == nullptr) {
        return;
      }

      renderers::ExternalRendererConfig config;
      config.id = "ExternalUserUI" + std::to_string(safeMain->userExternalRendererConfigs_.size() + 1);
      config.name = result.selectedName;
      config.binaryPath = result.selectedPath;
      config.licenseId = "User-supplied";
      safeMain->userExternalRendererConfigs_.push_back(config);
      safeMain->refreshRenderers();

      const juce::String statusText =
          result.valid ? "External renderer added" : "External renderer added (validation failed)";
      safeMain->statusLabel_.setText(statusText, juce::dontSendNotification);
      safeMain->appendTaskHistory(statusText);
      safeMain->reportEditor_.setText(safeMain->reportEditor_.getText() +
                                      "\nAdded external renderer: " + juce::String(result.selectedPath) +
                                      "\nValidation: " + juce::String(result.valid ? "passed" : "failed") +
                                      " (" + juce::String(result.diagnostics) + ")" +
                                      "\nLicense note: user-supplied tool is not distributed by this app.");
    };
    exportCallbacks.onLamePrefetchComplete = [safeMain](LamePrefetchResult result) {
      if (safeMain == nullptr) {
        return;
      }
      safeMain->prefetchLameButton_.setEnabled(true);
      safeMain->refreshCodecAvailability();

      if (result.success) {
        safeMain->statusLabel_.setText("LAME ready for MP3 export", juce::dontSendNotification);
        safeMain->appendTaskHistory("LAME prefetch completed");
      } else {
        safeMain->statusLabel_.setText("LAME prefetch failed", juce::dontSendNotification);
        safeMain->appendTaskHistory("LAME prefetch failed");
      }

      juce::String report = safeMain->reportEditor_.getText();
      if (!report.isEmpty()) {
        report += "\n";
      }
      report += "LAME prefetch: ";
      report += result.success ? "success" : "failed";
      if (!result.executablePath.empty()) {
        report += "\nPath: " + juce::String(result.executablePath);
      }
      if (!result.detail.empty()) {
        report += "\nDetail: " + juce::String(result.detail);
      }
      safeMain->reportEditor_.setText(report);
    };
    exportController_ = std::make_unique<ExportController>(backgroundPool_, std::move(exportCallbacks));
  }

  profileController_ = std::make_unique<ProfileController>();

  {
    PreviewController::Callbacks previewCallbacks;
    const auto safeMain = juce::Component::SafePointer<MainComponent>(this);
    previewCallbacks.onPreviewReady = [safeMain](PreviewBuildResult result) {
      if (safeMain == nullptr) {
        return;
      }
      if (result.generation != safeMain->previewBuildGeneration_.load()) {
        return;
      }
      if (!result.errorText.isEmpty()) {
        safeMain->reportEditor_.setText(safeMain->reportEditor_.getText() +
                                        "\nPreview rebuild skipped: " + result.errorText);
        return;
      }
      if (!result.success) {
        return;
      }

      safeMain->previewEngine_.setBuffers(result.rawMix, result.mastered);
      safeMain->updateTransportFromBuffer(result.preview);
      safeMain->transportController_.seekToFraction(result.previousProgress);
      safeMain->playbackCursorSamples_.store(safeMain->transportController_.positionSamples());
    };
    previewController_ = std::make_unique<PreviewController>(backgroundPool_, std::move(previewCallbacks));
  }

  {
    ProcessingController::Callbacks processingCallbacks;
    const auto safeMain = juce::Component::SafePointer<MainComponent>(this);
    processingCallbacks.onStatus = makeStatusCallback(safeMain);
    processingCallbacks.onTaskHistory = makeTaskHistoryCallback(safeMain);
    processingCallbacks.onAutoMixComplete = [safeMain](AutoMixResult result) {
      if (safeMain == nullptr) {
        return;
      }

      safeMain->finishCancelableTask();

      if (!result.errorText.isEmpty()) {
        safeMain->statusLabel_.setText("Auto Mix failed", juce::dontSendNotification);
        safeMain->reportEditor_.setText(result.errorText);
        safeMain->appendTaskHistory("Auto Mix failed");
        return;
      }

      if (result.cancelled) {
        safeMain->statusLabel_.setText("Auto Mix cancelled", juce::dontSendNotification);
        safeMain->appendTaskHistory("Auto Mix cancelled");
        return;
      }

      safeMain->analysisEntries_ = std::move(result.analysisEntries);
      safeMain->analysisTableModel_.setEntries(&safeMain->analysisEntries_);
      safeMain->analysisTable_.updateContent();

      if (result.mixPlan.has_value()) {
        safeMain->session_.mixPlan = result.mixPlan.value();
      }

      if (!result.reportText.isEmpty()) {
        safeMain->reportEditor_.setText(result.reportText);
      }

      safeMain->statusLabel_.setText("Auto Mix plan generated", juce::dontSendNotification);
      safeMain->appendTaskHistory("Auto Mix completed");
      safeMain->rebuildPreviewBuffersAsync();
    };
    processingCallbacks.onAutoMasterComplete = [safeMain](AutoMasterResult result) {
      if (safeMain == nullptr) {
        return;
      }

      safeMain->finishCancelableTask();

      if (!result.errorText.isEmpty()) {
        safeMain->statusLabel_.setText("Auto Master failed", juce::dontSendNotification);
        safeMain->reportEditor_.setText(result.errorText);
        safeMain->appendTaskHistory("Auto Master failed");
        return;
      }

      if (result.cancelled) {
        safeMain->statusLabel_.setText("Auto Master cancelled", juce::dontSendNotification);
        safeMain->appendTaskHistory("Auto Master cancelled");
        return;
      }

      safeMain->session_.masterPlan = std::move(result.masterPlan);
      safeMain->previewEngine_.setBuffers(result.rawMixBuffer, result.previewMaster);
      safeMain->previewEngine_.setSource(engine::PreviewSource::OriginalMix);
      safeMain->previewEngine_.stop();
      safeMain->updateTransportFromBuffer(safeMain->previewEngine_.buildCrossfadedPreview(1024));
      safeMain->updateMeterPanel(result.previewReport);

      safeMain->statusLabel_.setText("Auto Master plan generated", juce::dontSendNotification);
      safeMain->appendTaskHistory("Auto Master completed");
      if (!result.reportAppend.isEmpty()) {
        safeMain->reportEditor_.setText(safeMain->reportEditor_.getText() + result.reportAppend);
      }
    };
    processingCallbacks.onBatchComplete = [safeMain](BatchResult result) {
      if (safeMain == nullptr) {
        return;
      }

      safeMain->finishCancelableTask();

      if (!result.errorText.isEmpty()) {
        if (result.summary.isEmpty()) {
          safeMain->statusLabel_.setText("Batch preparation failed", juce::dontSendNotification);
          safeMain->reportEditor_.setText(result.errorText);
          safeMain->appendTaskHistory("Batch preparation failed");
        } else {
          safeMain->statusLabel_.setText("Batch folder has no supported audio files", juce::dontSendNotification);
          safeMain->appendTaskHistory("Batch preparation found no supported files");
        }
        return;
      }

      safeMain->statusLabel_.setText("Batch complete", juce::dontSendNotification);
      safeMain->reportEditor_.setText(result.summary);
      safeMain->appendTaskHistory("Batch completed");
    };
    processingController_ = std::make_unique<ProcessingController>(backgroundPool_, std::move(processingCallbacks));
  }

  {
    SessionController::Callbacks sessionCallbacks;
    const auto safeMain = juce::Component::SafePointer<MainComponent>(this);
    sessionCallbacks.onStatus = makeStatusCallback(safeMain);
    sessionCallbacks.onTaskHistory = makeTaskHistoryCallback(safeMain);
    sessionCallbacks.onSaveComplete = [safeMain](SessionSaveResult result) {
      if (safeMain == nullptr) {
        return;
      }
      safeMain->finishCancelableTask();
      if (result.cancelled) {
        safeMain->statusLabel_.setText("Session save cancelled", juce::dontSendNotification);
        safeMain->appendTaskHistory("Session save cancelled");
        return;
      }
      if (result.success) {
        safeMain->statusLabel_.setText("Session saved", juce::dontSendNotification);
        safeMain->appendTaskHistory("Session saved: " + juce::String(result.path));
      } else {
        safeMain->statusLabel_.setText("Save failed", juce::dontSendNotification);
        safeMain->reportEditor_.setText("Session save error:\n" + result.errorText);
        safeMain->appendTaskHistory("Session save failed");
      }
    };
    sessionCallbacks.onLoadComplete = [safeMain](SessionLoadResult result) {
      if (safeMain == nullptr) {
        return;
      }
      safeMain->finishCancelableTask();
      if (result.cancelled) {
        safeMain->statusLabel_.setText("Session load cancelled", juce::dontSendNotification);
        safeMain->appendTaskHistory("Session load cancelled");
        return;
      }
      if (result.session.has_value()) {
        safeMain->applyLoadedSession(std::move(result.session.value()), result.path);
      } else {
        safeMain->statusLabel_.setText("Load failed", juce::dontSendNotification);
        safeMain->reportEditor_.setText("Session load error:\n" + result.errorText);
        safeMain->appendTaskHistory("Session load failed");
      }
    };
    sessionController_ = std::make_unique<SessionController>(backgroundPool_, std::move(sessionCallbacks));
  }

  originalMixController_ = std::make_unique<OriginalMixController>();
}

} // namespace automix::app
