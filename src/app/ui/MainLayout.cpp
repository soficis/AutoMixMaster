#include "app/ui/MainLayout.h"

#include "app/ui/AudioPreviewManager.h"
#include "app/ui/ControlDeck.h"
#include "app/ui/GlowMeters.h"
#include "app/ui/HeaderBar.h"
#include "app/ui/HeroWaveform.h"
#include "app/ui/ModelBrowserPanel.h"
#include "app/ui/StemPanel.h"
#include "app/ui/TaskCenterPanel.h"
#include "app/ui/TaskOrchestrator.h"
#include "app/ui/TransportBar.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>

#include <juce_audio_utils/juce_audio_utils.h>
#include <nlohmann/json.hpp>

namespace automix::app {

using namespace theme;

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

namespace {

template <typename Comp>
auto safeAsync(Comp* comp) {
  return juce::Component::SafePointer<Comp>(comp);
}

void updateStemPanelFromSession(StemPanel& panel, const domain::Session& session) {
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

} // namespace

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

  addAndMakeVisible(*headerBar_);
  addAndMakeVisible(*heroWaveform_);
  addAndMakeVisible(*transportBar_);
  addAndMakeVisible(*controlDeck_);
  addAndMakeVisible(*taskCenter_);

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

  // 3. Populate combo boxes
  refreshRenderers();
  refreshCodecAvailability();
  refreshModelPacks();
  populateMasterPresetSelectors();
  refreshProjectProfiles();

  // 4. Wire all UI callbacks
  wireHeaderCallbacks();
  wireTransportCallbacks();
  wireControlDeckCallbacks();
  wireHeroWaveformCallbacks();

  taskCenter_->onCancel = [this] { taskOrchestrator_->cancelActiveTask(); };

  // 5. Create controllers
  auto safe = safeAsync(this);

  // --- ModelController ---
  {
    ModelController::Callbacks cb;
    cb.onStatus = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->setStatus(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->appendHistory(juce::String(msg));
      });
    };
    cb.onReport = [safe](const std::string& text) {
      juce::MessageManager::callAsync([safe, text]() {
        if (safe)
          safe->taskOrchestrator_->appendHistory("Report: " + juce::String(text).substring(0, 200));
      });
    };
    cb.onModelPacksChanged = [safe]() {
      juce::MessageManager::callAsync([safe]() {
        if (safe)
          safe->refreshModelPacks();
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
    cb.onStatus = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->setStatus(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->appendHistory(juce::String(msg));
      });
    };
    cb.onImportComplete = [safe](ImportResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        for (const auto& line : result.logLines)
          safe->taskOrchestrator_->appendHistory(juce::String(line));

        if (result.cancelled) {
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Import, "Import cancelled");
          return;
        }

        safe->sessionManager_.session().stems = result.stems;
        updateStemPanelFromSession(safe->controlDeck_->getStemPanel(), safe->sessionManager_.session());
        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Import, "Import complete");
        safe->rebuildPreview();
      });
    };
    importController_ = std::make_unique<ImportController>(backgroundPool_, std::move(cb));
  }

  // --- ExportController ---
  {
    ExportController::Callbacks cb;
    cb.onStatus = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->setStatus(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->appendHistory(juce::String(msg));
      });
    };
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
        } else if (result.cancelled) {
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::Export, "Export cancelled");
        } else {
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::Export, result.crashMessage.toStdString());
        }
      });
    };
    exportController_ = std::make_unique<ExportController>(backgroundPool_, std::move(cb));
  }

  // --- ProcessingController ---
  {
    ProcessingController::Callbacks cb;
    cb.onStatus = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->setStatus(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->appendHistory(juce::String(msg));
      });
    };
    cb.onAutoMixComplete = [safe](AutoMixResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        if (result.cancelled) {
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::AutoMix, "Auto Mix cancelled");
          return;
        }
        if (result.errorText.isNotEmpty()) {
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::AutoMix, result.errorText.toStdString());
          return;
        }

        safe->analysisEntries_ = result.analysisEntries;
        if (result.mixPlan.has_value())
          safe->sessionManager_.session().mixPlan = result.mixPlan;
        safe->taskOrchestrator_->appendHistory(result.reportText);
        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::AutoMix, "Auto Mix complete");
        safe->rebuildPreview();
      });
    };
    cb.onAutoMasterComplete = [safe](AutoMasterResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        if (result.cancelled) {
          safe->taskOrchestrator_->finishTaskCancelled(ActiveTask::AutoMaster, "Auto Master cancelled");
          return;
        }
        if (result.errorText.isNotEmpty()) {
          safe->taskOrchestrator_->finishTaskFailed(ActiveTask::AutoMaster, result.errorText.toStdString());
          return;
        }

        safe->sessionManager_.session().masterPlan = result.masterPlan;
        safe->taskOrchestrator_->appendHistory(result.reportAppend);
        safe->updateMeterPanel(result.previewReport);

        safe->previewManager_->setBuffer(result.previewMaster);
        safe->heroWaveform_->setBuffer(result.previewMaster);
        safe->updateTransportFromBuffer(result.previewMaster);

        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::AutoMaster, "Auto Master complete");
      });
    };
    cb.onBatchComplete = [safe](BatchResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        if (result.errorText.isNotEmpty())
          safe->taskOrchestrator_->appendHistory("Batch error: " + result.errorText);
        safe->taskOrchestrator_->appendHistory(result.summary);
        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::Batch, "Batch complete");
      });
    };
    processingController_ = std::make_unique<ProcessingController>(backgroundPool_, std::move(cb));
  }

  // --- SessionController ---
  {
    SessionController::Callbacks cb;
    cb.onStatus = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->setStatus(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskOrchestrator_->appendHistory(juce::String(msg));
      });
    };
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

  // 6. Audio device
  audioDeviceManager_.initialise(0, 2, nullptr, true);
  audioDeviceManager_.addAudioCallback(this);

  // 7. Transport
  transportController_.addChangeListener(this);
  startTimerHz(20);
  updateTransportDisplay();
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

  juce::FlexBox fb;
  fb.flexDirection = juce::FlexBox::Direction::column;
  fb.justifyContent = juce::FlexBox::JustifyContent::flexStart;

  fb.items.add(juce::FlexItem(*headerBar_).withHeight(static_cast<float>(kHeaderHeight)));
  fb.items.add(juce::FlexItem(*heroWaveform_).withFlex(2.0f).withMinHeight(120.0f));
  fb.items.add(juce::FlexItem(*transportBar_).withHeight(static_cast<float>(kTransportHeight)));
  fb.items.add(juce::FlexItem(*controlDeck_).withFlex(3.0f).withMinHeight(180.0f));
  fb.items.add(juce::FlexItem(*taskCenter_).withHeight(static_cast<float>(kTaskCenterHeight)));

  fb.performLayout(area);
}

// ─────────────────────────────────────────────────────────────────
// Keyboard Shortcuts
// ─────────────────────────────────────────────────────────────────

bool MainLayout::keyPressed(const juce::KeyPress& key) {
  auto ctrl = juce::ModifierKeys::ctrlModifier;
  if (key == juce::KeyPress('s', ctrl, 0)) {
    onSaveSession();
    return true;
  }
  if (key == juce::KeyPress('o', ctrl, 0)) {
    onLoadSession();
    return true;
  }
  if (key == juce::KeyPress('i', ctrl, 0)) {
    onImport();
    return true;
  }
  if (key == juce::KeyPress('m', ctrl, 0)) {
    onAutoMix();
    return true;
  }
  if (key == juce::KeyPress('e', ctrl, 0)) {
    onExport();
    return true;
  }
  if (key == juce::KeyPress('k', ctrl, 0)) {
    onModelsDialog();
    return true;
  }
  if (key == juce::KeyPress::spaceKey) {
    if (transportController_.isPlaying()) {
      transportController_.pause();
      transportBar_->setPlaying(false);
    } else {
      transportController_.play();
      transportBar_->setPlaying(true);
    }
    return true;
  }
  return false;
}

// ─────────────────────────────────────────────────────────────────
// Timer
// ─────────────────────────────────────────────────────────────────

void MainLayout::timerCallback() {
  updateTransportDisplay();
}

// ─────────────────────────────────────────────────────────────────
// Change Listener (transport state changes)
// ─────────────────────────────────────────────────────────────────

void MainLayout::changeListenerCallback(juce::ChangeBroadcaster* source) {
  if (source == &transportController_) {
    transportBar_->setPlaying(transportController_.isPlaying());
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
  std::lock_guard<std::mutex> lock(previewManager_->bufferMutex());
  const auto& buf = previewManager_->buffer();
  const float outputGain = std::clamp(outputVolume_.load(std::memory_order_relaxed), 0.0f, 1.5f);

  if (!transportController_.isPlaying() || buf.getNumSamples() == 0) {
    for (int ch = 0; ch < numOutputChannels; ++ch)
      if (outputChannelData[ch])
        std::fill_n(outputChannelData[ch], numSamples, 0.0f);
    return;
  }

  auto pos = transportController_.positionSamples();
  int totalSamples = buf.getNumSamples();
  int bufChannels = buf.getNumChannels();

  for (int i = 0; i < numSamples; ++i) {
    if (pos >= totalSamples) {
      transportController_.stop();
      for (int ch = 0; ch < numOutputChannels; ++ch)
        if (outputChannelData[ch])
          std::fill_n(outputChannelData[ch] + i, numSamples - i, 0.0f);
      break;
    }

    for (int ch = 0; ch < numOutputChannels; ++ch) {
      if (outputChannelData[ch]) {
        int srcCh = std::min(ch, bufChannels - 1);
        outputChannelData[ch][i] = buf.getSample(srcCh, static_cast<int>(pos)) * outputGain;
      }
    }
    ++pos;
  }

  transportController_.seekToSample(pos);
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
    outputVolume_.store(static_cast<float>(std::clamp(volume, 0.0, 1.5)), std::memory_order_relaxed);
  };
}

// ─────────────────────────────────────────────────────────────────
// Wiring: ControlDeck
// ─────────────────────────────────────────────────────────────────

void MainLayout::wireControlDeckCallbacks() {
  controlDeck_->onImport = [this] { onImport(); };
  controlDeck_->onAutoMix = [this] { onAutoMix(); };
  controlDeck_->onAutoMaster = [this] { onAutoMaster(); };
  controlDeck_->onBatch = [this] { onBatch(); };
  controlDeck_->onExport = [this] { onExport(); };

  controlDeck_->getStemPanel().onSoloChanged = [this](const std::string& /*stemId*/, bool /*solo*/) {
    rebuildPreview();
  };
  controlDeck_->getStemPanel().onMuteChanged = [this](const std::string& /*stemId*/, bool /*mute*/) {
    rebuildPreview();
  };
  controlDeck_->getStemPanel().onVolumeChanged = [this](const std::string& /*stemId*/, float /*volume*/) {
    rebuildPreview();
  };

  controlDeck_->getRendererBox().onChange = [this] {
    const auto rendererId = selectionState_.rendererIdForCombo(controlDeck_->getRendererBox().getSelectedId());
    if (rendererId.has_value())
      sessionManager_.session().renderSettings.rendererName = rendererId.value();
  };
  controlDeck_->getProfileBox().onChange = [this] {
    const auto profileId = selectionState_.projectProfileIdForCombo(controlDeck_->getProfileBox().getSelectedId());
    if (profileId.has_value()) {
      sessionManager_.session().projectProfileId = profileId.value();
      auto profile = domain::findProjectProfile(projectProfiles_, profileId.value());
      if (profile.has_value()) {
        ProfileController pc;
        pc.applyProfile(sessionManager_.session(), profile.value());
        taskOrchestrator_->appendHistory("Applied profile: " + juce::String(profile->name));
      }
    }
  };
  controlDeck_->getMasterPresetBox().onChange = [this] {
    const auto preset = selectionState_.masterPresetForCombo(controlDeck_->getMasterPresetBox().getSelectedId());
    if (preset.has_value() && sessionManager_.session().masterPlan.has_value())
      sessionManager_.session().masterPlan->preset = preset.value();
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
  controlDeck_->getResidualBlendSlider().onValueChange = [this] {
    sessionManager_.session().residualBlend = controlDeck_->getResidualBlendSlider().getValue();
  };
}

// ─────────────────────────────────────────────────────────────────
// Wiring: HeroWaveform
// ─────────────────────────────────────────────────────────────────

void MainLayout::wireHeroWaveformCallbacks() {
  heroWaveform_->onSeek = [this](double progress) {
    transportController_.seekToFraction(std::clamp(progress, 0.0, 1.0));
  };
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

    if (!taskOrchestrator_->beginTask(ActiveTask::Import, "Importing stems", "", "Import started")) {
      importChooser_.reset();
      return;
    }

    importController_->importFiles(
        std::move(selectedFiles),
        controlDeck_->getSeparatedStemsToggle().getToggleState(),
        sessionManager_.session().preferredStemCount,
        taskOrchestrator_->cancelFlag(ActiveTask::Import));

    importChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Auto Mix
// ─────────────────────────────────────────────────────────────────

void MainLayout::onAutoMix() {
  if (sessionManager_.session().stems.empty()) {
    taskOrchestrator_->setStatus("No stems", "Import stems first");
    return;
  }

  if (!taskOrchestrator_->beginTask(ActiveTask::AutoMix, "Auto Mix", "Running...", "Auto Mix started"))
    return;

  std::optional<ai::ModelPack> mixPack;
  auto activeId = modelManager_.activePackId("mix");
  if (activeId != "none") {
    for (const auto& p : modelManager_.availablePacks()) {
      if (p.id == activeId) {
        mixPack = p;
        break;
      }
    }
  }

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

  std::optional<ai::ModelPack> masterPack;
  auto activeId = modelManager_.activePackId("master");
  if (activeId != "none") {
    for (const auto& p : modelManager_.availablePacks()) {
      if (p.id == activeId) {
        masterPack = p;
        break;
      }
    }
  }

  processingController_->runAutoMaster(
      sessionManager_.session(),
      settings,
      preset,
      masterPack,
      taskOrchestrator_->cancelFlag(ActiveTask::AutoMaster));
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

    auto settings = buildCurrentRenderSettings("");
    processingController_->runBatch(
        selected.getFullPathName().toStdString(),
        settings,
        taskOrchestrator_->cancelFlag(ActiveTask::Batch));

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

    exportController_->runExport(
        sessionManager_.session(),
        settings,
        analysisEntries_,
        taskOrchestrator_->cancelFlag(ActiveTask::Export));
    exportChooser_.reset();
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
  panel->setDiscoveredModels(modelController_->discoveredModels());
  panel->setSize(600, 500);

  panel->onFetchCatalog = [this] {
    if (!taskOrchestrator_->beginTask(ActiveTask::Model, "Fetching model catalog", "", "Model catalog fetch started"))
      return;
    modelController_->fetchCatalog(taskOrchestrator_->cancelFlag(ActiveTask::Model));
  };
  panel->onInstallModel = [this](const std::string& repoId) {
    if (!taskOrchestrator_->beginTask(ActiveTask::Model, "Installing model", juce::String(repoId),
                                      "Model install started: " + juce::String(repoId)))
      return;
    modelController_->installModel(repoId, taskOrchestrator_->cancelFlag(ActiveTask::Model));
  };
  panel->onCheckUpdates = [this] {
    if (!taskOrchestrator_->beginTask(ActiveTask::Model, "Checking model updates", "", "Model update check started"))
      return;
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
  auto* selector = new juce::AudioDeviceSelectorComponent(
      audioDeviceManager_, 0, 0, 0, 2, false, false, true, false);
  selector->setSize(500, 400);

  juce::DialogWindow::LaunchOptions options;
  options.content.setOwned(selector);
  options.dialogTitle = "Audio Settings";
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
  refreshProjectProfiles();
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
    juce::String label = info.name;
    if (!info.available)
      label += " (unavailable)";
    box.addItem(label, comboId);
    selectionState_.bindRendererId(comboId, info.id);
    if (preferredId == 0 && info.available)
      preferredId = comboId;
    if (info.id == sessionManager_.session().renderSettings.rendererName)
      preferredId = comboId;
    ++comboId;
  }
  if (preferredId != 0)
    box.setSelectedId(preferredId, juce::dontSendNotification);
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
  selectionState_.bindExportSpeedMode(2, "preview");
  modeBox.setSelectedId(1, juce::dontSendNotification);
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Model Packs
// ─────────────────────────────────────────────────────────────────

void MainLayout::refreshModelPacks() {
  modelManager_.setRootPath("ModelPacks");
  modelManager_.scan();
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

  masterBox.setSelectedId(1, juce::dontSendNotification);
  platformBox.setSelectedId(1, juce::dontSendNotification);
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

// ─────────────────────────────────────────────────────────────────
// Query: Build Render Settings
// ─────────────────────────────────────────────────────────────────

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

  return settings;
}

std::vector<renderers::ExternalRendererConfig> MainLayout::loadConfiguredExternalRenderers() {
  std::vector<renderers::ExternalRendererConfig> configs;

  std::vector<std::filesystem::path> candidates;
  std::error_code ec;
  auto cwd = std::filesystem::current_path(ec);
  if (!ec) {
    candidates.push_back(cwd / "external_renderers.json");
    candidates.push_back(cwd / "assets" / "renderers" / "external_renderers.json");
    auto parent = cwd.parent_path();
    if (parent != cwd) {
      candidates.push_back(parent / "assets" / "renderers" / "external_renderers.json");
      auto grandparent = parent.parent_path();
      if (grandparent != parent)
        candidates.push_back(grandparent / "assets" / "renderers" / "external_renderers.json");
    }
  }

  for (const auto& path : candidates) {
    if (!std::filesystem::is_regular_file(path, ec) || ec)
      continue;

    try {
      std::ifstream in(path);
      if (!in.is_open())
        continue;

      nlohmann::json json;
      in >> json;

      if (!json.is_array())
        continue;

      for (const auto& entry : json) {
        renderers::ExternalRendererConfig config;
        config.id = entry.value("id", "");
        config.name = entry.value("name", "");
        config.version = entry.value("version", "unknown");
        config.licenseId = entry.value("licenseId", "unknown");

        std::string binaryPath = entry.value("binaryPath", "");
        if (binaryPath.empty() || config.id.empty())
          continue;

        std::filesystem::path binary(binaryPath);
        config.binaryPath = binary.is_absolute() ? binary : (path.parent_path() / binary);
        config.bundledByDefault = entry.value("bundledByDefault", false);

        if (entry.contains("pinnedProfileIds") && entry.at("pinnedProfileIds").is_array())
          config.pinnedProfileIds = entry.at("pinnedProfileIds").get<std::vector<std::string>>();

        configs.push_back(std::move(config));
      }

      break;
    } catch (const std::exception& error) {
      taskOrchestrator_->appendHistory("External renderer config parse failed: "
                                       + juce::String(path.string())
                                       + " (" + juce::String(error.what()) + ")");
      continue;
    } catch (...) {
      taskOrchestrator_->appendHistory("External renderer config parse failed: "
                                       + juce::String(path.string())
                                       + " (unknown error)");
      continue;
    }
  }

  return configs;
}

} // namespace automix::app
