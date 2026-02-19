#include "app/ui/MainLayout.h"

#include "app/ui/ControlDeck.h"
#include "app/ui/GlowMeters.h"
#include "app/ui/HeaderBar.h"
#include "app/ui/HeroWaveform.h"
#include "app/ui/StemPanel.h"
#include "app/ui/TaskCenterPanel.h"
#include "app/ui/TransportBar.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <unordered_map>

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

domain::StemMixDecision* findOrCreateStemDecision(domain::Session& session,
                                                  const std::string& stemId) {
  if (!session.mixPlan.has_value()) {
    session.mixPlan = domain::MixPlan {};
  }

  for (auto& decision : session.mixPlan->stemDecisions) {
    if (decision.stemId == stemId) {
      return &decision;
    }
  }

  auto& decision = session.mixPlan->stemDecisions.emplace_back();
  decision.stemId = stemId;
  return &decision;
}

double trimVolumeToGainDb(const float volume) {
  constexpr double kMinLinearGain = 0.0001;
  const double linearGain = std::clamp(static_cast<double>(volume), kMinLinearGain, 1.5);
  return 20.0 * std::log10(linearGain);
}

void applyStemDisplayStateToPreviewSession(
    domain::Session& previewSession,
    const std::vector<StemPanel::StemDisplay>& displays) {
  if (displays.empty()) {
    return;
  }

  std::unordered_map<std::string, const StemPanel::StemDisplay*> displayByStemId;
  displayByStemId.reserve(displays.size());
  bool anySoloed = false;
  for (const auto& display : displays) {
    displayByStemId.emplace(display.id, &display);
    anySoloed = anySoloed || display.solo;
  }

  for (auto& stem : previewSession.stems) {
    const auto displayIt = displayByStemId.find(stem.id);
    if (displayIt == displayByStemId.end()) {
      continue;
    }

    const auto& display = *displayIt->second;
    stem.enabled = display.enabled && !display.mute && (!anySoloed || display.solo);

    constexpr float kVolumeEpsilon = 0.0001f;
    if (std::abs(display.volume - 1.0f) <= kVolumeEpsilon) {
      continue;
    }

    auto* decision = findOrCreateStemDecision(previewSession, display.id);
    decision->gainDb += trimVolumeToGainDb(display.volume);
  }
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

  // 2. Populate combo boxes
  refreshRenderers();
  refreshCodecAvailability();
  refreshModelPacks();
  populateMasterPresetSelectors();
  refreshProjectProfiles();

  // 3. Wire all UI callbacks
  wireHeaderCallbacks();
  wireTransportCallbacks();
  wireControlDeckCallbacks();
  wireTaskCenterCallbacks();
  wireHeroWaveformCallbacks();

  // 4. Create controllers
  auto safe = safeAsync(this);

  // --- ModelController ---
  {
    ModelController::Callbacks cb;
    cb.onStatus = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskCenter_->setCurrentTask(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->appendTaskHistory(juce::String(msg));
      });
    };
    cb.onReport = [safe](const std::string& text) {
      juce::MessageManager::callAsync([safe, text]() {
        if (safe)
          safe->appendTaskHistory("Report: " + juce::String(text).substring(0, 200));
      });
    };
    cb.onModelPacksChanged = [safe]() {
      juce::MessageManager::callAsync([safe]() {
        if (safe)
          safe->refreshModelPacks();
      });
    };
    cb.onCatalogReady = [safe]() {
      juce::MessageManager::callAsync([safe]() {
        if (safe)
          safe->appendTaskHistory("Model catalog ready");
      });
    };
    modelController_ = std::make_unique<ModelController>(modelManager_, backgroundPool_, std::move(cb));
  }

  // --- ImportController ---
  {
    ImportController::Callbacks cb;
    cb.onStatus = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->taskCenter_->setCurrentTask(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->appendTaskHistory(juce::String(msg));
      });
    };
    cb.onImportComplete = [safe](ImportResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;
        for (const auto& line : result.logLines)
          safe->appendTaskHistory(juce::String(line));

        safe->session_.stems = result.stems;
        updateStemPanelFromSession(safe->controlDeck_->getStemPanel(), safe->session_);
        safe->taskRunning_.store(false);
        safe->taskCenter_->setCanCancel(false);
        safe->taskCenter_->setCurrentTask("Import complete", "");
        safe->taskCenter_->setProgress(1.0);
        safe->rebuildPreviewBuffersAsync();
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
          safe->taskCenter_->setCurrentTask(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->appendTaskHistory(juce::String(msg));
      });
    };
    cb.onExportComplete = [safe](ExportResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;
        for (const auto& line : result.logs)
          safe->appendTaskHistory(juce::String(line));

        safe->analysisEntries_ = result.analysisEntries;

        if (result.success) {
          safe->taskCenter_->setCurrentTask("Export complete", result.outputAudioPath);
          safe->appendTaskHistory("Export succeeded: " + juce::String(result.outputAudioPath));
        } else if (result.cancelled) {
          safe->taskCenter_->setCurrentTask("Export cancelled", "");
        } else {
          safe->taskCenter_->setCurrentTask("Export failed", result.crashMessage.toStdString());
        }
        safe->taskRunning_.store(false);
        safe->taskCenter_->setCanCancel(false);
        safe->taskCenter_->setProgress(1.0);
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
          safe->taskCenter_->setCurrentTask(juce::String(msg), "");
      });
    };
    cb.onTaskHistory = [safe](const std::string& msg) {
      juce::MessageManager::callAsync([safe, msg]() {
        if (safe)
          safe->appendTaskHistory(juce::String(msg));
      });
    };
    cb.onAutoMixComplete = [safe](AutoMixResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;
        safe->taskRunning_.store(false);
        safe->taskCenter_->setCanCancel(false);

        if (result.cancelled) {
          safe->taskCenter_->setCurrentTask("Auto Mix cancelled", "");
          return;
        }
        if (result.errorText.isNotEmpty()) {
          safe->taskCenter_->setCurrentTask("Auto Mix failed", "");
          safe->appendTaskHistory("Error: " + result.errorText);
          return;
        }

        safe->analysisEntries_ = result.analysisEntries;
        if (result.mixPlan.has_value())
          safe->session_.mixPlan = result.mixPlan;
        safe->appendTaskHistory(result.reportText);
        safe->taskCenter_->setCurrentTask("Auto Mix complete", "");
        safe->taskCenter_->setProgress(1.0);
        safe->rebuildPreviewBuffersAsync();
      });
    };
    cb.onAutoMasterComplete = [safe](AutoMasterResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;
        safe->taskRunning_.store(false);
        safe->taskCenter_->setCanCancel(false);

        if (result.cancelled) {
          safe->taskCenter_->setCurrentTask("Auto Master cancelled", "");
          return;
        }
        if (result.errorText.isNotEmpty()) {
          safe->taskCenter_->setCurrentTask("Auto Master failed", "");
          safe->appendTaskHistory("Error: " + result.errorText);
          return;
        }

        safe->session_.masterPlan = result.masterPlan;
        safe->appendTaskHistory(result.reportAppend);
        safe->taskCenter_->setCurrentTask("Auto Master complete", "");
        safe->taskCenter_->setProgress(1.0);

        // Update meters from mastering report
        safe->updateMeterPanel(result.previewReport);

        // Update preview buffers
        {
          std::lock_guard<std::mutex> lock(safe->playbackBufferMutex_);
          safe->playbackBuffer_ = result.previewMaster;
        }
        safe->heroWaveform_->setBuffer(result.previewMaster);
        safe->updateTransportFromBuffer(result.previewMaster);
      });
    };
    cb.onBatchComplete = [safe](BatchResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;
        safe->taskRunning_.store(false);
        safe->taskCenter_->setCanCancel(false);
        if (result.errorText.isNotEmpty()) {
          safe->appendTaskHistory("Batch error: " + result.errorText);
        }
        safe->appendTaskHistory(result.summary);
        safe->taskCenter_->setCurrentTask("Batch complete", "");
        safe->taskCenter_->setProgress(1.0);
      });
    };
    processingController_ = std::make_unique<ProcessingController>(backgroundPool_, std::move(cb));
  }

  // --- PreviewController ---
  {
    PreviewController::Callbacks cb;
    cb.onPreviewReady = [safe](PreviewBuildResult result) {
      juce::MessageManager::callAsync([safe, result = std::move(result)]() {
        if (!safe)
          return;

        // Discard stale results
        if (result.generation < safe->previewBuildGeneration_.load())
          return;

        if (!result.success) {
          if (result.errorText.isNotEmpty())
            safe->appendTaskHistory("Preview build failed: " + result.errorText);
          return;
        }

        // Update playback buffer
        {
          std::lock_guard<std::mutex> lock(safe->playbackBufferMutex_);
          safe->playbackBuffer_ = result.preview;
        }

        // Update waveform and transport
        safe->heroWaveform_->setBuffer(result.preview);
        safe->updateTransportFromBuffer(result.preview);

        // Restore playback position if possible
        if (result.previousProgress > 0.0 && result.previousProgress < 1.0)
          safe->transportController_.seekToFraction(result.previousProgress);

        safe->appendTaskHistory("Preview updated");
      });
    };
    previewController_ = std::make_unique<PreviewController>(backgroundPool_, std::move(cb));
  }

  // 5. Audio device
  audioDeviceManager_.initialise(0, 2, nullptr, true);
  audioDeviceManager_.addAudioCallback(this);

  // 6. Transport
  transportController_.addChangeListener(this);
  startTimerHz(20);
  updateTransportDisplay();
}

// ─────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────

MainLayout::~MainLayout() {
  cancelRender_.store(true);
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
    onModelsMenu();
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
  std::lock_guard<std::mutex> lock(playbackBufferMutex_);
  const float outputGain = std::clamp(outputVolume_.load(std::memory_order_relaxed), 0.0f, 1.5f);

  if (!transportController_.isPlaying() || playbackBuffer_.getNumSamples() == 0) {
    for (int ch = 0; ch < numOutputChannels; ++ch)
      if (outputChannelData[ch])
        std::fill_n(outputChannelData[ch], numSamples, 0.0f);
    return;
  }

  auto pos = transportController_.positionSamples();
  int totalSamples = playbackBuffer_.getNumSamples();
  int bufChannels = playbackBuffer_.getNumChannels();

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
        outputChannelData[ch][i] = playbackBuffer_.getSample(srcCh, static_cast<int>(pos)) * outputGain;
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
  headerBar_->onModels = [this] { onModelsMenu(); };
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
    auto& tl = session_.timeline;
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

  // Stem panel callbacks
  controlDeck_->getStemPanel().onSoloChanged = [this](const std::string& /*stemId*/, bool /*solo*/) {
    rebuildPreviewBuffersAsync();
  };
  controlDeck_->getStemPanel().onMuteChanged = [this](const std::string& /*stemId*/, bool /*mute*/) {
    rebuildPreviewBuffersAsync();
  };
  controlDeck_->getStemPanel().onVolumeChanged = [this](const std::string& /*stemId*/, float /*volume*/) {
    rebuildPreviewBuffersAsync();
  };

  // Settings combo boxes: update session on change
  controlDeck_->getRendererBox().onChange = [this] {
    auto it = rendererIdByComboId_.find(controlDeck_->getRendererBox().getSelectedId());
    if (it != rendererIdByComboId_.end())
      session_.renderSettings.rendererName = it->second;
  };
  controlDeck_->getProfileBox().onChange = [this] {
    auto it = projectProfileIdByComboId_.find(controlDeck_->getProfileBox().getSelectedId());
    if (it != projectProfileIdByComboId_.end()) {
      session_.projectProfileId = it->second;
      auto profile = domain::findProjectProfile(projectProfiles_, it->second);
      if (profile.has_value()) {
        ProfileController pc;
        pc.applyProfile(session_, profile.value());
        appendTaskHistory("Applied profile: " + juce::String(profile->name));
      }
    }
  };
  controlDeck_->getMasterPresetBox().onChange = [this] {
    auto it = masterPresetByComboId_.find(controlDeck_->getMasterPresetBox().getSelectedId());
    if (it != masterPresetByComboId_.end() && session_.masterPlan.has_value())
      session_.masterPlan->preset = it->second;
  };
  controlDeck_->getExportFormatBox().onChange = [this] {
    auto it = codecFormatByComboId_.find(controlDeck_->getExportFormatBox().getSelectedId());
    if (it != codecFormatByComboId_.end())
      session_.renderSettings.outputFormat = it->second;
  };
  controlDeck_->getExportModeBox().onChange = [this] {
    auto it = exportSpeedModeByComboId_.find(controlDeck_->getExportModeBox().getSelectedId());
    if (it != exportSpeedModeByComboId_.end())
      session_.renderSettings.exportSpeedMode = it->second;
  };
  controlDeck_->getResidualBlendSlider().onValueChange = [this] {
    session_.residualBlend = controlDeck_->getResidualBlendSlider().getValue();
  };
}

// ─────────────────────────────────────────────────────────────────
// Wiring: TaskCenter
// ─────────────────────────────────────────────────────────────────

void MainLayout::wireTaskCenterCallbacks() {
  taskCenter_->onCancel = [this] { onCancel(); };
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
  if (taskRunning_.load()) {
    taskCenter_->setCurrentTask("Busy", "A task is already running");
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

    cancelRender_.store(false);
    taskRunning_.store(true);
    taskCenter_->setCanCancel(true);
    taskCenter_->setCurrentTask("Importing stems", "");
    taskCenter_->setProgress(-1.0);
    appendTaskHistory("Import started");

    importController_->importFiles(
        std::move(selectedFiles),
        controlDeck_->getSeparatedStemsToggle().getToggleState(),
        session_.preferredStemCount);

    importChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Auto Mix
// ─────────────────────────────────────────────────────────────────

void MainLayout::onAutoMix() {
  if (taskRunning_.load()) {
    taskCenter_->setCurrentTask("Busy", "A task is already running");
    return;
  }
  if (session_.stems.empty()) {
    taskCenter_->setCurrentTask("No stems", "Import stems first");
    return;
  }

  cancelRender_.store(false);
  taskRunning_.store(true);
  taskCenter_->setCanCancel(true);
  taskCenter_->setCurrentTask("Auto Mix", "Running...");
  taskCenter_->setProgress(-1.0);
  appendTaskHistory("Auto Mix started");

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

  processingController_->runAutoMix(session_, mixPack, cancelRender_);
}

// ─────────────────────────────────────────────────────────────────
// Action: Auto Master
// ─────────────────────────────────────────────────────────────────

void MainLayout::onAutoMaster() {
  if (taskRunning_.load()) {
    taskCenter_->setCurrentTask("Busy", "A task is already running");
    return;
  }
  if (session_.stems.empty()) {
    taskCenter_->setCurrentTask("No stems", "Import stems first");
    return;
  }

  cancelRender_.store(false);
  taskRunning_.store(true);
  taskCenter_->setCanCancel(true);
  taskCenter_->setCurrentTask("Auto Master", "Running...");
  taskCenter_->setProgress(-1.0);
  appendTaskHistory("Auto Master started");

  // Determine preset
  auto preset = domain::MasterPreset::DefaultStreaming;
  auto platformIt = platformPresetByComboId_.find(controlDeck_->getPlatformPresetBox().getSelectedId());
  if (platformIt != platformPresetByComboId_.end())
    preset = platformIt->second;
  if (preset == domain::MasterPreset::Custom) {
    auto masterIt = masterPresetByComboId_.find(controlDeck_->getMasterPresetBox().getSelectedId());
    if (masterIt != masterPresetByComboId_.end())
      preset = masterIt->second;
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

  processingController_->runAutoMaster(session_, settings, preset, masterPack, cancelRender_);
}

// ─────────────────────────────────────────────────────────────────
// Action: Batch
// ─────────────────────────────────────────────────────────────────

void MainLayout::onBatch() {
  if (taskRunning_.load()) {
    taskCenter_->setCurrentTask("Busy", "A task is already running");
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

    cancelRender_.store(false);
    taskRunning_.store(true);
    taskCenter_->setCanCancel(true);
    taskCenter_->setCurrentTask("Batch processing", "");
    taskCenter_->setProgress(-1.0);
    appendTaskHistory("Batch started: " + selected.getFullPathName());

    auto settings = buildCurrentRenderSettings("");
    processingController_->runBatch(
        selected.getFullPathName().toStdString(), settings, cancelRender_);

    batchImportChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Export
// ─────────────────────────────────────────────────────────────────

void MainLayout::onExport() {
  if (taskRunning_.load()) {
    taskCenter_->setCurrentTask("Busy", "A task is already running");
    return;
  }
  if (session_.stems.empty()) {
    taskCenter_->setCurrentTask("No stems", "Import stems first");
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

    cancelRender_.store(false);
    taskRunning_.store(true);
    taskCenter_->setCanCancel(true);
    taskCenter_->setCurrentTask("Exporting", "");
    taskCenter_->setProgress(-1.0);
    appendTaskHistory("Export started: " + selected.getFullPathName());

    exportController_->runExport(session_, settings, analysisEntries_, cancelRender_);
    exportChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Cancel
// ─────────────────────────────────────────────────────────────────

void MainLayout::onCancel() {
  cancelRender_.store(true);
  taskCenter_->setCurrentTask("Cancelling", "");
  appendTaskHistory("Cancel requested");
}

// ─────────────────────────────────────────────────────────────────
// Action: Save Session
// ─────────────────────────────────────────────────────────────────

void MainLayout::onSaveSession() {
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

    auto path = selected.getFullPathName().toStdString();
    try {
      sessionRepository_.save(path, session_);
      appendTaskHistory("Session saved to " + juce::String(path));
      headerBar_->setSessionName(selected.getFileNameWithoutExtension());
    } catch (const std::exception& e) {
      appendTaskHistory("Failed to save session: " + juce::String(e.what()));
    }
    saveSessionChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Load Session
// ─────────────────────────────────────────────────────────────────

void MainLayout::onLoadSession() {
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

    auto path = selected.getFullPathName().toStdString();
    try {
      auto loaded = sessionRepository_.load(path);
      applyLoadedSession(std::move(loaded), selected.getFullPathName());
    } catch (const std::exception& e) {
      appendTaskHistory("Failed to load session: " + juce::String(e.what()));
    }
    loadSessionChooser_.reset();
  });
}

// ─────────────────────────────────────────────────────────────────
// Action: Models Menu
// ─────────────────────────────────────────────────────────────────

void MainLayout::onModelsMenu() {
  juce::PopupMenu menu;
  menu.addItem("Fetch Catalog", [this] { modelController_->fetchCatalog(); });
  menu.addItem("Show Installed", [this] { modelController_->showInstalled(); });
  menu.addItem("Check Updates", [this] { modelController_->checkUpdates(); });
  menu.addItem("Verify Integrity", [this] { modelController_->verifyIntegrity(); });
  menu.showMenuAsync(juce::PopupMenu::Options()
                         .withTargetScreenArea(headerBar_->getScreenBounds()));
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
  appendTaskHistory("Settings dialog opened");
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
// UI Update: Misc
// ─────────────────────────────────────────────────────────────────

void MainLayout::appendTaskHistory(const juce::String& line) {
  taskHistoryLines_.push_back(line);
  taskCenter_->appendHistory(line);
}

void MainLayout::updateTransportFromBuffer(const engine::AudioBuffer& buffer) {
  if (buffer.getNumSamples() <= 0) {
    return;
  }

  transportController_.setTimeline(
      static_cast<int64_t>(buffer.getNumSamples()),
      buffer.getSampleRate());
  transportController_.setLoopRangeSeconds(session_.timeline.loopInSeconds,
                                           session_.timeline.loopOutSeconds,
                                           session_.timeline.loopEnabled);
  transportBar_->setLoopEnabled(session_.timeline.loopEnabled);
  heroWaveform_->setLoopRange(session_.timeline.loopEnabled,
                              transportController_.loopInProgress(),
                              transportController_.loopOutProgress());
  heroWaveform_->setZoom(session_.timeline.zoom, 0.5);
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

void MainLayout::rebuildPreviewBuffersAsync() {
  if (session_.stems.empty())
    return;

  // Build a session copy with solo/mute/volume state applied from the StemPanel
  domain::Session previewSession = session_;
  const auto displays = controlDeck_->getStemPanel().getStemDisplays();
  applyStemDisplayStateToPreviewSession(previewSession, displays);

  auto gen = ++previewBuildGeneration_;
  double currentProgress = transportController_.progress();

  PreviewBuildRequest request;
  request.session = std::move(previewSession);
  request.generation = gen;
  request.previousProgress = currentProgress;

  previewController_->rebuildPreview(std::move(request));
  appendTaskHistory("Preview rebuild started");
}

void MainLayout::applyLoadedSession(domain::Session loadedSession, const juce::String& sourcePath) {
  session_ = std::move(loadedSession);
  headerBar_->setSessionName(juce::File(sourcePath).getFileNameWithoutExtension());
  updateStemPanelFromSession(controlDeck_->getStemPanel(), session_);
  transportBar_->setLoopEnabled(session_.timeline.loopEnabled);
  heroWaveform_->setZoom(session_.timeline.zoom, 0.5);
  refreshRenderers();
  refreshCodecAvailability();
  refreshModelPacks();
  refreshProjectProfiles();
  appendTaskHistory("Session loaded: " + sourcePath);
  rebuildPreviewBuffersAsync();
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Renderers
// ─────────────────────────────────────────────────────────────────

void MainLayout::refreshRenderers() {
  auto& box = controlDeck_->getRendererBox();
  box.clear(juce::dontSendNotification);
  rendererIdByComboId_.clear();

  renderers::RendererRegistry registry;
  rendererInfos_ = registry.list(loadConfiguredExternalRenderers());

  int comboId = 1;
  int preferredId = 0;
  for (const auto& info : rendererInfos_) {
    juce::String label = info.name;
    if (!info.available)
      label += " (unavailable)";
    box.addItem(label, comboId);
    rendererIdByComboId_[comboId] = info.id;
    if (preferredId == 0 && info.available)
      preferredId = comboId;
    if (info.id == session_.renderSettings.rendererName)
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
  codecFormatByComboId_.clear();

  // Standard formats
  struct FormatEntry { std::string id; juce::String label; };
  std::vector<FormatEntry> formats = {
      {"wav", "WAV"}, {"flac", "FLAC"}, {"aiff", "AIFF"}, {"ogg", "OGG"}, {"mp3", "MP3"}};

  int comboId = 1;
  for (const auto& fmt : formats) {
    box.addItem(fmt.label, comboId);
    codecFormatByComboId_[comboId] = fmt.id;
    ++comboId;
  }
  box.setSelectedId(1, juce::dontSendNotification);

  // Export speed mode
  auto& modeBox = controlDeck_->getExportModeBox();
  modeBox.clear(juce::dontSendNotification);
  exportSpeedModeByComboId_.clear();
  modeBox.addItem("Final", 1);
  exportSpeedModeByComboId_[1] = "final";
  modeBox.addItem("Quick Preview", 2);
  exportSpeedModeByComboId_[2] = "preview";
  modeBox.setSelectedId(1, juce::dontSendNotification);
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Model Packs
// ─────────────────────────────────────────────────────────────────

void MainLayout::refreshModelPacks() {
  modelManager_.setRootPath("ModelPacks");
  const auto packs = modelManager_.scan();
  // Model packs are available to controllers; no combo boxes needed at this time
}

// ─────────────────────────────────────────────────────────────────
// Refresh: Master Presets
// ─────────────────────────────────────────────────────────────────

void MainLayout::populateMasterPresetSelectors() {
  auto& masterBox = controlDeck_->getMasterPresetBox();
  auto& platformBox = controlDeck_->getPlatformPresetBox();
  masterBox.clear(juce::dontSendNotification);
  platformBox.clear(juce::dontSendNotification);
  masterPresetByComboId_.clear();
  platformPresetByComboId_.clear();

  int mid = 1;
  auto addMaster = [&](const juce::String& label, domain::MasterPreset p) {
    masterBox.addItem(label, mid);
    masterPresetByComboId_[mid++] = p;
  };
  addMaster("Default Streaming", domain::MasterPreset::DefaultStreaming);
  addMaster("Broadcast", domain::MasterPreset::Broadcast);
  addMaster("Udio Optimized", domain::MasterPreset::UdioOptimized);
  addMaster("Custom", domain::MasterPreset::Custom);

  int pid = 1;
  auto addPlatform = [&](const juce::String& label, domain::MasterPreset p) {
    platformBox.addItem(label, pid);
    platformPresetByComboId_[pid++] = p;
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
  projectProfileIdByComboId_.clear();

  int selectedId = 0;
  int comboId = 1;
  for (const auto& profile : projectProfiles_) {
    box.addItem(profile.name + " [" + profile.id + "]", comboId);
    projectProfileIdByComboId_[comboId] = profile.id;
    if (profile.id == session_.projectProfileId)
      selectedId = comboId;
    ++comboId;
  }
  if (selectedId == 0 && !projectProfileIdByComboId_.empty())
    selectedId = projectProfileIdByComboId_.begin()->first;
  if (selectedId > 0)
    box.setSelectedId(selectedId, juce::dontSendNotification);
}

// ─────────────────────────────────────────────────────────────────
// Query: Build Render Settings
// ─────────────────────────────────────────────────────────────────

domain::RenderSettings MainLayout::buildCurrentRenderSettings(const std::string& outputPath) const {
  domain::RenderSettings settings = session_.renderSettings;
  settings.outputPath = outputPath;

  auto fmtIt = codecFormatByComboId_.find(controlDeck_->getExportFormatBox().getSelectedId());
  if (fmtIt != codecFormatByComboId_.end())
    settings.outputFormat = fmtIt->second;

  auto modeIt = exportSpeedModeByComboId_.find(controlDeck_->getExportModeBox().getSelectedId());
  if (modeIt != exportSpeedModeByComboId_.end())
    settings.exportSpeedMode = modeIt->second;

  auto rendIt = rendererIdByComboId_.find(controlDeck_->getRendererBox().getSelectedId());
  if (rendIt != rendererIdByComboId_.end())
    settings.rendererName = rendIt->second;

  return settings;
}

std::vector<renderers::ExternalRendererConfig> MainLayout::loadConfiguredExternalRenderers() {
  std::vector<renderers::ExternalRendererConfig> configs;

  // Search for external_renderers.json in standard locations
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

      break; // Use first valid config file found
    } catch (const std::exception& error) {
      appendTaskHistory("External renderer config parse failed: "
                        + juce::String(path.string())
                        + " (" + juce::String(error.what()) + ")");
      continue;
    } catch (...) {
      appendTaskHistory("External renderer config parse failed: "
                        + juce::String(path.string())
                        + " (unknown error)");
      continue;
    }
  }

  return configs;
}

} // namespace automix::app
