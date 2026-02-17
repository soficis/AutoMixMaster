#include "app/MainComponent.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "engine/OfflineRenderPipeline.h"
#include "util/LameDownloader.h"
#include "util/StringUtils.h"
#include "util/WavWriter.h"

namespace automix::app {
namespace {

using ::automix::util::toLower;
using ::automix::util::toJuceText;

std::vector<std::string> splitDelimited(const std::string& text, const char delimiter) {
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    if (!token.empty()) {
      out.push_back(token);
    }
  }
  return out;
}

constexpr const char* kExportSpeedModeFinal = "final";
constexpr const char* kExportSpeedModeBalanced = "balanced";
constexpr const char* kExportSpeedModeQuick = "quick";

const ai::ModelPack* findPackById(const ai::ModelManager& manager, const std::string& id) {
  if (id.empty() || id == "none") {
    return nullptr;
  }
  for (const auto& pack : manager.availablePacks()) {
    if (pack.id == id) {
      return &pack;
    }
  }
  return nullptr;
}

std::string formatDuration(const double seconds) {
  const auto clamped = std::max(0.0, seconds);
  const int total = static_cast<int>(std::lround(clamped));
  const int mins = total / 60;
  const int secs = total % 60;
  std::ostringstream output;
  output << mins << ':';
  if (secs < 10) {
    output << '0';
  }
  output << secs;
  return output.str();
}

} // namespace

void MainComponent::AnalysisTableModel::setEntries(const std::vector<analysis::StemAnalysisEntry>* entries) {
  entries_ = entries;
}

int MainComponent::AnalysisTableModel::getNumRows() {
  return entries_ == nullptr ? 0 : static_cast<int>(entries_->size());
}

void MainComponent::AnalysisTableModel::paintRowBackground(juce::Graphics& g,
                                                           const int rowNumber,
                                                           const int width,
                                                           const int height,
                                                           const bool rowIsSelected) {
  juce::ignoreUnused(rowNumber, width, height);
  g.fillAll(rowIsSelected ? juce::Colours::darkslategrey : juce::Colours::black.withAlpha(0.15f));
}

void MainComponent::AnalysisTableModel::paintCell(juce::Graphics& g,
                                                  const int rowNumber,
                                                  const int columnId,
                                                  const int width,
                                                  const int height,
                                                  const bool rowIsSelected) {
  juce::ignoreUnused(rowIsSelected);
  if (entries_ == nullptr || rowNumber < 0 || rowNumber >= static_cast<int>(entries_->size())) {
    return;
  }

  const auto& entry = entries_->at(static_cast<size_t>(rowNumber));
  juce::String text;
  switch (columnId) {
    case 1:
      text = entry.stemName;
      break;
    case 2:
      text = juce::String(entry.metrics.peakDb, 2);
      break;
    case 3:
      text = juce::String(entry.metrics.rmsDb, 2);
      break;
    case 4:
      text = juce::String(entry.metrics.crestDb, 2);
      break;
    case 5:
      text = juce::String(entry.metrics.lowEnergy, 3);
      break;
    case 6:
      text = juce::String(entry.metrics.midEnergy, 3);
      break;
    case 7:
      text = juce::String(entry.metrics.highEnergy, 3);
      break;
    case 8:
      text = juce::String(entry.metrics.silenceRatio, 3);
      break;
    default:
      break;
  }

  g.setColour(juce::Colours::white);
  g.drawText(text, 4, 0, width - 8, height, juce::Justification::centredLeft, true);
}

MainComponent::MainComponent() {
  addAndMakeVisible(importButton_);
  addAndMakeVisible(originalMixButton_);
  addAndMakeVisible(clearOriginalMixButton_);
  addAndMakeVisible(regenerateCacheButton_);
  addAndMakeVisible(saveSessionButton_);
  addAndMakeVisible(loadSessionButton_);
  addAndMakeVisible(modelsMenuButton_);
  addAndMakeVisible(autoMixButton_);
  addAndMakeVisible(autoMasterButton_);
  addAndMakeVisible(batchImportButton_);
  addAndMakeVisible(previewOriginalButton_);
  addAndMakeVisible(previewRenderedButton_);
  addAndMakeVisible(playPauseButton_);
  addAndMakeVisible(stopButton_);
  addAndMakeVisible(loopInButton_);
  addAndMakeVisible(loopOutButton_);
  addAndMakeVisible(clearLoopButton_);
  addAndMakeVisible(addExternalRendererButton_);
  addAndMakeVisible(prefetchLameButton_);
  addAndMakeVisible(exportButton_);
  addAndMakeVisible(cancelButton_);
  addAndMakeVisible(separatedStemsToggle_);
  addAndMakeVisible(residualBlendLabel_);
  addAndMakeVisible(residualBlendSlider_);
  addAndMakeVisible(rendererBox_);
  addAndMakeVisible(exportFormatLabel_);
  addAndMakeVisible(exportFormatBox_);
  addAndMakeVisible(exportSpeedModeLabel_);
  addAndMakeVisible(exportSpeedModeBox_);
  addAndMakeVisible(projectProfileLabel_);
  addAndMakeVisible(projectProfileBox_);
  addAndMakeVisible(exportBitrateLabel_);
  addAndMakeVisible(exportBitrateSlider_);
  addAndMakeVisible(mp3ModeLabel_);
  addAndMakeVisible(mp3ModeBox_);
  addAndMakeVisible(mp3VbrLabel_);
  addAndMakeVisible(mp3VbrSlider_);
  addAndMakeVisible(gpuProviderLabel_);
  addAndMakeVisible(gpuProviderBox_);
  addAndMakeVisible(masterPresetLabel_);
  addAndMakeVisible(masterPresetBox_);
  addAndMakeVisible(platformPresetLabel_);
  addAndMakeVisible(platformPresetBox_);
  addAndMakeVisible(soloStemLabel_);
  addAndMakeVisible(soloStemBox_);
  addAndMakeVisible(muteStemLabel_);
  addAndMakeVisible(muteStemBox_);
  addAndMakeVisible(transportSlider_);
  addAndMakeVisible(zoomLabel_);
  addAndMakeVisible(zoomSlider_);
  addAndMakeVisible(fineScrubToggle_);
  addAndMakeVisible(aiModelsLabel_);
  addAndMakeVisible(roleModelBox_);
  addAndMakeVisible(mixModelBox_);
  addAndMakeVisible(masterModelBox_);
  addAndMakeVisible(statusLabel_);
  addAndMakeVisible(meterLufsLabel_);
  addAndMakeVisible(meterShortTermLabel_);
  addAndMakeVisible(meterTruePeakLabel_);
  addAndMakeVisible(waveformPreview_);
  addAndMakeVisible(analysisTable_);
  addAndMakeVisible(reportEditor_);
  addAndMakeVisible(taskCenterLabel_);
  addAndMakeVisible(taskCenterEditor_);

  importButton_.addListener(this);
  originalMixButton_.addListener(this);
  clearOriginalMixButton_.addListener(this);
  regenerateCacheButton_.addListener(this);
  saveSessionButton_.addListener(this);
  loadSessionButton_.addListener(this);
  modelsMenuButton_.addListener(this);
  autoMixButton_.addListener(this);
  autoMasterButton_.addListener(this);
  batchImportButton_.addListener(this);
  previewOriginalButton_.addListener(this);
  previewRenderedButton_.addListener(this);
  playPauseButton_.addListener(this);
  stopButton_.addListener(this);
  loopInButton_.addListener(this);
  loopOutButton_.addListener(this);
  clearLoopButton_.addListener(this);
  addExternalRendererButton_.addListener(this);
  prefetchLameButton_.addListener(this);
  exportButton_.addListener(this);
  cancelButton_.addListener(this);

  rendererBox_.addListener(this);
  exportFormatBox_.addListener(this);
  exportSpeedModeBox_.addListener(this);
  mp3ModeBox_.addListener(this);
  projectProfileBox_.addListener(this);
  gpuProviderBox_.addListener(this);
  masterPresetBox_.addListener(this);
  platformPresetBox_.addListener(this);
  roleModelBox_.addListener(this);
  mixModelBox_.addListener(this);
  masterModelBox_.addListener(this);
  soloStemBox_.addListener(this);
  muteStemBox_.addListener(this);

  residualBlendSlider_.addListener(this);
  exportBitrateSlider_.addListener(this);
  mp3VbrSlider_.addListener(this);
  transportSlider_.addListener(this);
  zoomSlider_.addListener(this);
  fineScrubToggle_.addListener(this);

  residualBlendLabel_.setJustificationType(juce::Justification::centredLeft);
  residualBlendSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  residualBlendSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
  residualBlendSlider_.setRange(0.0, 10.0, 0.1);
  residualBlendSlider_.setValue(0.0, juce::dontSendNotification);

  exportFormatLabel_.setJustificationType(juce::Justification::centredLeft);
  exportSpeedModeLabel_.setJustificationType(juce::Justification::centredLeft);
  exportSpeedModeBox_.addItem("Final", 1);
  exportSpeedModeByComboId_[1] = kExportSpeedModeFinal;
  exportSpeedModeBox_.addItem("Balanced", 2);
  exportSpeedModeByComboId_[2] = kExportSpeedModeBalanced;
  exportSpeedModeBox_.addItem("Quick", 3);
  exportSpeedModeByComboId_[3] = kExportSpeedModeQuick;
  exportSpeedModeBox_.setSelectedId(1, juce::dontSendNotification);
  projectProfileLabel_.setJustificationType(juce::Justification::centredLeft);

  exportBitrateLabel_.setJustificationType(juce::Justification::centredLeft);
  exportBitrateSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  exportBitrateSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
  exportBitrateSlider_.setRange(64.0, 320.0, 1.0);
  exportBitrateSlider_.setValue(320.0, juce::dontSendNotification);
  mp3ModeLabel_.setJustificationType(juce::Justification::centredLeft);
  mp3ModeBox_.addItem("CBR", 1);
  mp3ModeBox_.addItem("VBR", 2);
  mp3ModeBox_.setSelectedId(1, juce::dontSendNotification);
  mp3VbrLabel_.setJustificationType(juce::Justification::centredLeft);
  mp3VbrSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  mp3VbrSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 20);
  mp3VbrSlider_.setRange(0.0, 9.0, 1.0);
  mp3VbrSlider_.setValue(4.0, juce::dontSendNotification);

  gpuProviderLabel_.setJustificationType(juce::Justification::centredLeft);
  gpuProviderBox_.addItem("Auto", 1);
  gpuProviderBox_.addItem("CPU", 2);
  gpuProviderBox_.addItem("DirectML", 3);
  gpuProviderBox_.addItem("CoreML", 4);
  gpuProviderBox_.addItem("CUDA", 5);
  gpuProviderBox_.setSelectedId(1, juce::dontSendNotification);

  masterPresetLabel_.setJustificationType(juce::Justification::centredLeft);
  platformPresetLabel_.setJustificationType(juce::Justification::centredLeft);
  soloStemLabel_.setJustificationType(juce::Justification::centredLeft);
  muteStemLabel_.setJustificationType(juce::Justification::centredLeft);

  transportSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  transportSlider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
  transportSlider_.setRange(0.0, 1.0, 0.0001);
  transportSlider_.setValue(0.0, juce::dontSendNotification);
  transportSlider_.setSkewFactor(1.0);

  zoomLabel_.setJustificationType(juce::Justification::centredLeft);
  zoomSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  zoomSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
  zoomSlider_.setRange(1.0, 32.0, 0.1);
  zoomSlider_.setValue(1.0, juce::dontSendNotification);
  fineScrubToggle_.setToggleState(false, juce::dontSendNotification);

  aiModelsLabel_.setJustificationType(juce::Justification::centredLeft);
  roleModelBox_.setTextWhenNothingSelected("Role model");
  mixModelBox_.setTextWhenNothingSelected("Mix model");
  masterModelBox_.setTextWhenNothingSelected("Master model");

  statusLabel_.setText("Ready", juce::dontSendNotification);
  meterLufsLabel_.setJustificationType(juce::Justification::centredLeft);
  meterShortTermLabel_.setJustificationType(juce::Justification::centredLeft);
  meterTruePeakLabel_.setJustificationType(juce::Justification::centredLeft);

  analysisTableModel_.setEntries(&analysisEntries_);
  analysisTable_.setModel(&analysisTableModel_);
  analysisTable_.getHeader().addColumn("Stem", 1, 180);
  analysisTable_.getHeader().addColumn("Peak", 2, 70);
  analysisTable_.getHeader().addColumn("RMS", 3, 70);
  analysisTable_.getHeader().addColumn("Crest", 4, 70);
  analysisTable_.getHeader().addColumn("Low", 5, 60);
  analysisTable_.getHeader().addColumn("Mid", 6, 60);
  analysisTable_.getHeader().addColumn("High", 7, 60);
  analysisTable_.getHeader().addColumn("Silence", 8, 70);

  reportEditor_.setMultiLine(true);
  reportEditor_.setReadOnly(true);
  reportEditor_.setScrollbarsShown(true);

  taskCenterLabel_.setJustificationType(juce::Justification::centredLeft);
  taskCenterEditor_.setMultiLine(true);
  taskCenterEditor_.setReadOnly(true);
  taskCenterEditor_.setScrollbarsShown(true);
  taskCenterEditor_.setText("Task history will appear here.");

  cancelButton_.setEnabled(false);
  clearOriginalMixButton_.setEnabled(false);
  session_.sessionName = "Untitled Session";
  session_.renderSettings.exportSpeedMode = kExportSpeedModeFinal;
  session_.timeline.zoom = zoomSlider_.getValue();
  session_.timeline.fineScrub = fineScrubToggle_.getToggleState();

  refreshRenderers();
  initializeControllers();
  refreshCodecAvailability();
  refreshModelPacks();
  populateMasterPresetSelectors();
  refreshProjectProfiles();
  refreshStemRoutingSelectors();

  const auto deviceError = audioDeviceManager_.initialise(0, 2, nullptr, true);
  audioDeviceManager_.addAudioCallback(this);
  if (!deviceError.isEmpty()) {
    statusLabel_.setText("Audio device init warning: " + deviceError, juce::dontSendNotification);
    appendTaskHistory("Audio device init warning: " + deviceError);
  }

  transportController_.addChangeListener(this);
  startTimerHz(20);
  updateTransportLoopAndZoomUI();
  updateTransportDisplay();
}

MainComponent::~MainComponent() {
  cancelImport_.store(true);
  cancelModel_.store(true);
  cancelSession_.store(true);
  cancelMix_.store(true);
  cancelMaster_.store(true);
  cancelBatch_.store(true);
  cancelExport_.store(true);
  activeTask_ = ActiveTask::None;
  backgroundPool_.removeAllJobs(true, 5000);

  stopTimer();
  audioDeviceManager_.removeAudioCallback(this);
  transportController_.removeChangeListener(this);

  importButton_.removeListener(this);
  originalMixButton_.removeListener(this);
  clearOriginalMixButton_.removeListener(this);
  regenerateCacheButton_.removeListener(this);
  saveSessionButton_.removeListener(this);
  loadSessionButton_.removeListener(this);
  modelsMenuButton_.removeListener(this);
  autoMixButton_.removeListener(this);
  autoMasterButton_.removeListener(this);
  batchImportButton_.removeListener(this);
  previewOriginalButton_.removeListener(this);
  previewRenderedButton_.removeListener(this);
  playPauseButton_.removeListener(this);
  stopButton_.removeListener(this);
  loopInButton_.removeListener(this);
  loopOutButton_.removeListener(this);
  clearLoopButton_.removeListener(this);
  addExternalRendererButton_.removeListener(this);
  prefetchLameButton_.removeListener(this);
  exportButton_.removeListener(this);
  cancelButton_.removeListener(this);

  rendererBox_.removeListener(this);
  exportFormatBox_.removeListener(this);
  exportSpeedModeBox_.removeListener(this);
  mp3ModeBox_.removeListener(this);
  projectProfileBox_.removeListener(this);
  gpuProviderBox_.removeListener(this);
  masterPresetBox_.removeListener(this);
  platformPresetBox_.removeListener(this);
  roleModelBox_.removeListener(this);
  mixModelBox_.removeListener(this);
  masterModelBox_.removeListener(this);
  soloStemBox_.removeListener(this);
  muteStemBox_.removeListener(this);

  residualBlendSlider_.removeListener(this);
  exportBitrateSlider_.removeListener(this);
  mp3VbrSlider_.removeListener(this);
  transportSlider_.removeListener(this);
  zoomSlider_.removeListener(this);
  fineScrubToggle_.removeListener(this);
}

void MainComponent::resized() {
  auto area = getLocalBounds().reduced(8);

  auto top = area.removeFromTop(34);
  auto toolsRow = area.removeFromTop(30);
  auto meterRow = area.removeFromTop(24);
  auto blendRow = area.removeFromTop(28);
  auto exportRow = area.removeFromTop(28);
  auto presetRow = area.removeFromTop(28);
  auto stemRow = area.removeFromTop(28);
  auto modelRow = area.removeFromTop(30);
  auto waveformRow = area.removeFromTop(110);
  auto transportControlRow = area.removeFromTop(28);
  auto transportRow = area.removeFromTop(24);

  importButton_.setBounds(top.removeFromLeft(84).reduced(2));
  originalMixButton_.setBounds(top.removeFromLeft(96).reduced(2));
  clearOriginalMixButton_.setBounds(top.removeFromLeft(104).reduced(2));
  regenerateCacheButton_.setBounds(top.removeFromLeft(110).reduced(2));
  saveSessionButton_.setBounds(top.removeFromLeft(88).reduced(2));
  loadSessionButton_.setBounds(top.removeFromLeft(88).reduced(2));
  modelsMenuButton_.setBounds(top.removeFromLeft(84).reduced(2));
  autoMixButton_.setBounds(top.removeFromLeft(84).reduced(2));
  autoMasterButton_.setBounds(top.removeFromLeft(96).reduced(2));
  batchImportButton_.setBounds(top.removeFromLeft(96).reduced(2));
  exportButton_.setBounds(top.removeFromLeft(76).reduced(2));
  cancelButton_.setBounds(top.removeFromLeft(70).reduced(2));

  previewOriginalButton_.setBounds(toolsRow.removeFromLeft(100).reduced(2));
  previewRenderedButton_.setBounds(toolsRow.removeFromLeft(100).reduced(2));
  playPauseButton_.setBounds(toolsRow.removeFromLeft(96).reduced(2));
  stopButton_.setBounds(toolsRow.removeFromLeft(70).reduced(2));
  loopInButton_.setBounds(toolsRow.removeFromLeft(95).reduced(2));
  loopOutButton_.setBounds(toolsRow.removeFromLeft(100).reduced(2));
  clearLoopButton_.setBounds(toolsRow.removeFromLeft(90).reduced(2));
  separatedStemsToggle_.setBounds(toolsRow.removeFromLeft(160).reduced(2));
  rendererBox_.setBounds(toolsRow.removeFromLeft(220).reduced(2));
  addExternalRendererButton_.setBounds(toolsRow.removeFromLeft(170).reduced(2));
  prefetchLameButton_.setBounds(toolsRow.removeFromLeft(130).reduced(2));

  meterLufsLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));
  meterShortTermLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));
  meterTruePeakLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));

  residualBlendLabel_.setBounds(blendRow.removeFromLeft(132).reduced(2));
  residualBlendSlider_.setBounds(blendRow.removeFromLeft(250).reduced(2));

  exportFormatLabel_.setBounds(exportRow.removeFromLeft(50).reduced(2));
  exportFormatBox_.setBounds(exportRow.removeFromLeft(110).reduced(2));
  exportSpeedModeLabel_.setBounds(exportRow.removeFromLeft(48).reduced(2));
  exportSpeedModeBox_.setBounds(exportRow.removeFromLeft(104).reduced(2));
  projectProfileLabel_.setBounds(exportRow.removeFromLeft(56).reduced(2));
  projectProfileBox_.setBounds(exportRow.removeFromLeft(150).reduced(2));
  exportBitrateLabel_.setBounds(exportRow.removeFromLeft(72).reduced(2));
  exportBitrateSlider_.setBounds(exportRow.removeFromLeft(118).reduced(2));
  mp3ModeLabel_.setBounds(exportRow.removeFromLeft(64).reduced(2));
  mp3ModeBox_.setBounds(exportRow.removeFromLeft(72).reduced(2));
  mp3VbrLabel_.setBounds(exportRow.removeFromLeft(46).reduced(2));
  mp3VbrSlider_.setBounds(exportRow.removeFromLeft(98).reduced(2));
  gpuProviderLabel_.setBounds(exportRow.removeFromLeft(76).reduced(2));
  gpuProviderBox_.setBounds(exportRow.removeFromLeft(116).reduced(2));

  masterPresetLabel_.setBounds(presetRow.removeFromLeft(96).reduced(2));
  masterPresetBox_.setBounds(presetRow.removeFromLeft(200).reduced(2));
  platformPresetLabel_.setBounds(presetRow.removeFromLeft(70).reduced(2));
  platformPresetBox_.setBounds(presetRow.removeFromLeft(190).reduced(2));

  soloStemLabel_.setBounds(stemRow.removeFromLeft(40).reduced(2));
  soloStemBox_.setBounds(stemRow.removeFromLeft(220).reduced(2));
  muteStemLabel_.setBounds(stemRow.removeFromLeft(44).reduced(2));
  muteStemBox_.setBounds(stemRow.removeFromLeft(220).reduced(2));

  aiModelsLabel_.setBounds(modelRow.removeFromLeft(64).reduced(2));
  roleModelBox_.setBounds(modelRow.removeFromLeft(250).reduced(2));
  mixModelBox_.setBounds(modelRow.removeFromLeft(250).reduced(2));
  masterModelBox_.setBounds(modelRow.removeFromLeft(250).reduced(2));

  waveformPreview_.setBounds(waveformRow.reduced(2));
  zoomLabel_.setBounds(transportControlRow.removeFromLeft(46).reduced(2));
  zoomSlider_.setBounds(transportControlRow.removeFromLeft(220).reduced(2));
  fineScrubToggle_.setBounds(transportControlRow.removeFromLeft(100).reduced(2));
  transportSlider_.setBounds(transportRow.reduced(2));

  statusLabel_.setBounds(area.removeFromTop(24).reduced(2));
  taskCenterLabel_.setBounds(area.removeFromTop(22).reduced(2));
  taskCenterEditor_.setBounds(area.removeFromTop(98).reduced(2));
  analysisTable_.setBounds(area.removeFromTop(150));
  reportEditor_.setBounds(area.reduced(0, 6));
}

void MainComponent::buttonClicked(juce::Button* button) {
  if (button == &importButton_) {
    onImport();
    return;
  }

  if (button == &originalMixButton_) {
    onImportOriginalMix();
    return;
  }

  if (button == &clearOriginalMixButton_) {
    onClearOriginalMix();
    return;
  }

  if (button == &regenerateCacheButton_) {
    onRegenerateCachedRenders();
    return;
  }

  if (button == &saveSessionButton_) {
    onSaveSession();
    return;
  }

  if (button == &loadSessionButton_) {
    onLoadSession();
    return;
  }

  if (button == &modelsMenuButton_) {
    onModelsMenu();
    return;
  }

  if (button == &autoMixButton_) {
    onAutoMix();
    return;
  }

  if (button == &autoMasterButton_) {
    onAutoMaster();
    return;
  }

  if (button == &batchImportButton_) {
    onBatchImport();
    return;
  }

  if (button == &previewOriginalButton_) {
    onPreviewOriginal();
    return;
  }

  if (button == &previewRenderedButton_) {
    onPreviewRendered();
    return;
  }

  if (button == &playPauseButton_) {
    if (transportController_.isPlaying()) {
      transportController_.pause();
      previewEngine_.stop();
    } else {
      playbackCursorSamples_.store(transportController_.positionSamples());
      transportController_.play();
      previewEngine_.play();
    }
    updateTransportDisplay();
    return;
  }

  if (button == &stopButton_) {
    transportController_.stop();
    previewEngine_.stop();
    playbackCursorSamples_.store(0);
    updateTransportDisplay();
    return;
  }

  if (button == &loopInButton_) {
    session_.timeline.loopInSeconds = transportController_.positionSeconds();
    if (session_.timeline.loopOutSeconds <= session_.timeline.loopInSeconds) {
      session_.timeline.loopOutSeconds = std::max(session_.timeline.loopInSeconds + 0.5, transportController_.totalSeconds());
    }
    session_.timeline.loopEnabled = session_.timeline.loopOutSeconds > session_.timeline.loopInSeconds;
    transportController_.setLoopRangeSeconds(session_.timeline.loopInSeconds,
                                             session_.timeline.loopOutSeconds,
                                             session_.timeline.loopEnabled);
    appendTaskHistory("Loop In set at " + juce::String(session_.timeline.loopInSeconds, 2) + "s");
    return;
  }

  if (button == &loopOutButton_) {
    session_.timeline.loopOutSeconds = std::max(session_.timeline.loopInSeconds + 0.5, transportController_.positionSeconds());
    session_.timeline.loopEnabled = session_.timeline.loopOutSeconds > session_.timeline.loopInSeconds;
    transportController_.setLoopRangeSeconds(session_.timeline.loopInSeconds,
                                             session_.timeline.loopOutSeconds,
                                             session_.timeline.loopEnabled);
    appendTaskHistory("Loop Out set at " + juce::String(session_.timeline.loopOutSeconds, 2) + "s");
    return;
  }

  if (button == &clearLoopButton_) {
    session_.timeline.loopEnabled = false;
    transportController_.clearLoopRange();
    appendTaskHistory("Loop cleared");
    return;
  }

  if (button == &fineScrubToggle_) {
    session_.timeline.fineScrub = fineScrubToggle_.getToggleState();
    appendTaskHistory(juce::String("Fine Scrub ") + (session_.timeline.fineScrub ? "enabled" : "disabled"));
    return;
  }

  if (button == &addExternalRendererButton_) {
    onAddExternalRenderer();
    return;
  }

  if (button == &prefetchLameButton_) {
    onPrefetchLame();
    return;
  }

  if (button == &exportButton_) {
    onExport();
    return;
  }

  if (button == &cancelButton_) {
    onCancel();
  }
}

void MainComponent::comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) {
  if (comboBoxThatHasChanged == &roleModelBox_) {
    const auto it = roleModelIdByComboId_.find(roleModelBox_.getSelectedId());
    modelManager_.setActivePackId("role", it != roleModelIdByComboId_.end() ? it->second : "none");
    return;
  }

  if (comboBoxThatHasChanged == &mixModelBox_) {
    const auto it = mixModelIdByComboId_.find(mixModelBox_.getSelectedId());
    modelManager_.setActivePackId("mix", it != mixModelIdByComboId_.end() ? it->second : "none");
    return;
  }

  if (comboBoxThatHasChanged == &masterModelBox_) {
    const auto it = masterModelIdByComboId_.find(masterModelBox_.getSelectedId());
    modelManager_.setActivePackId("master", it != masterModelIdByComboId_.end() ? it->second : "none");
    return;
  }

  if (comboBoxThatHasChanged == &projectProfileBox_) {
    const auto it = projectProfileIdByComboId_.find(projectProfileBox_.getSelectedId());
    if (it != projectProfileIdByComboId_.end()) {
      if (const auto profile = domain::findProjectProfile(projectProfiles_, it->second); profile.has_value()) {
        applyProjectProfile(profile.value());
      }
    }
    return;
  }

  if (comboBoxThatHasChanged == &exportSpeedModeBox_) {
    session_.renderSettings.exportSpeedMode = selectedExportSpeedMode();
    if (isQuickExportModeSelected()) {
      applyQuickExportDefaults();
      appendTaskHistory("Export mode set to Quick (MP3 VBR)");
    } else if (session_.renderSettings.exportSpeedMode == kExportSpeedModeBalanced) {
      appendTaskHistory("Export mode set to Balanced");
    } else {
      appendTaskHistory("Export mode set to Final");
    }
    updateExportCodecControls();
    return;
  }

  if (comboBoxThatHasChanged == &gpuProviderBox_) {
    switch (gpuProviderBox_.getSelectedId()) {
      case 2:
        session_.renderSettings.gpuExecutionProvider = "cpu";
        break;
      case 3:
        session_.renderSettings.gpuExecutionProvider = "directml";
        break;
      case 4:
        session_.renderSettings.gpuExecutionProvider = "coreml";
        break;
      case 5:
        session_.renderSettings.gpuExecutionProvider = "cuda";
        break;
      default:
        session_.renderSettings.gpuExecutionProvider = "auto";
        break;
    }
    return;
  }

  if (comboBoxThatHasChanged == &rendererBox_) {
    const auto it = rendererIdByComboId_.find(rendererBox_.getSelectedId());
    if (it != rendererIdByComboId_.end()) {
      session_.renderSettings.rendererName = it->second;
    }
    return;
  }

  if (comboBoxThatHasChanged == &exportFormatBox_) {
    const auto formatIt = codecFormatByComboId_.find(exportFormatBox_.getSelectedId());
    const std::string format = formatIt != codecFormatByComboId_.end() ? formatIt->second : "wav";
    const auto availability = util::WavWriter::getAvailableFormats();
    const auto isAvailable = [&availability](const std::string& candidateFormat) {
      const auto it = std::find_if(availability.begin(), availability.end(), [&](const auto& entry) {
        return toLower(entry.format) == toLower(candidateFormat);
      });
      return it != availability.end() && it->available;
    };

    updateExportCodecControls();

    if (!isAvailable(format)) {
      for (const auto& [comboId, formatName] : codecFormatByComboId_) {
        if (isAvailable(formatName)) {
          exportFormatBox_.setSelectedId(comboId, juce::dontSendNotification);
          break;
        }
      }

      juce::String message = "Selected export format is unavailable";
      const auto detailIt = std::find_if(availability.begin(), availability.end(), [&](const auto& entry) {
        return toLower(entry.format) == toLower(format);
      });
      if (detailIt != availability.end() && !detailIt->detail.empty()) {
        message += ": " + juce::String(detailIt->detail);
      }
      statusLabel_.setText(message, juce::dontSendNotification);
    }
    return;
  }

  if (comboBoxThatHasChanged == &mp3ModeBox_) {
    session_.renderSettings.mp3UseVbr = mp3ModeBox_.getSelectedId() == 2;
    updateExportCodecControls();
    return;
  }

  if (comboBoxThatHasChanged == &soloStemBox_ || comboBoxThatHasChanged == &muteStemBox_) {
    rebuildPreviewBuffers();
    return;
  }
}

void MainComponent::sliderValueChanged(juce::Slider* slider) {
  if (slider == &residualBlendSlider_) {
    session_.residualBlend = residualBlendSlider_.getValue();
    return;
  }

  if (slider == &zoomSlider_) {
    session_.timeline.zoom = zoomSlider_.getValue();
    updateTransportLoopAndZoomUI();
    return;
  }

  if (slider == &mp3VbrSlider_) {
    session_.renderSettings.mp3VbrQuality =
        std::clamp(static_cast<int>(std::lround(mp3VbrSlider_.getValue())), 0, 9);
    return;
  }

  if (slider == &transportSlider_ && !ignoreTransportSliderChange_) {
    const double target = transportSlider_.getValue();
    if (fineScrubToggle_.getToggleState()) {
      const double current = transportController_.progress();
      const double blended = current + (target - current) * 0.18;
      lastFineScrubProgress_ = blended;
      transportController_.seekToFraction(blended);
    } else {
      lastFineScrubProgress_ = target;
      transportController_.seekToFraction(target);
    }
    playbackCursorSamples_.store(transportController_.positionSamples());
    return;
  }
}

void MainComponent::timerCallback() {
  if (transportController_.isPlaying()) {
    const auto currentCursor = playbackCursorSamples_.load();
    const auto totalSamples = transportController_.totalSamples();
    if (totalSamples > 0) {
      transportController_.seekToSample(std::clamp<int64_t>(currentCursor, 0, totalSamples));
    }
  }

  updateTransportDisplay();
}

void MainComponent::changeListenerCallback(juce::ChangeBroadcaster* source) {
  if (source == &transportController_) {
    updateTransportDisplay();
  }
}

void MainComponent::audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                                     const int numInputChannels,
                                                     float* const* outputChannelData,
                                                     const int numOutputChannels,
                                                     const int numSamples,
                                                     const juce::AudioIODeviceCallbackContext& context) {
  juce::ignoreUnused(inputChannelData, numInputChannels, context);

  for (int ch = 0; ch < numOutputChannels; ++ch) {
    if (outputChannelData[ch] != nullptr) {
      juce::FloatVectorOperations::clear(outputChannelData[ch], numSamples);
    }
  }

  if (!previewEngine_.isPlaying() || !transportController_.isPlaying() || numOutputChannels <= 0 || numSamples <= 0) {
    return;
  }

  int64_t cursor = playbackCursorSamples_.load();
  bool reachedEnd = false;

  {
    std::scoped_lock lock(playbackBufferMutex_);
    if (playbackBuffer_.getNumSamples() == 0 || playbackBuffer_.getNumChannels() == 0) {
      return;
    }

    const int sourceChannels = playbackBuffer_.getNumChannels();
    const int totalSamples = playbackBuffer_.getNumSamples();
    const int64_t clampedStart = std::clamp<int64_t>(cursor, 0, totalSamples);

    for (int sample = 0; sample < numSamples; ++sample) {
      const int64_t sourceIndex = clampedStart + sample;
      if (sourceIndex >= totalSamples) {
        reachedEnd = true;
        break;
      }
      for (int ch = 0; ch < numOutputChannels; ++ch) {
        const int sourceChannel = std::min(ch, sourceChannels - 1);
        const float value = playbackBuffer_.getSample(sourceChannel, static_cast<int>(sourceIndex));
        if (outputChannelData[ch] != nullptr) {
          outputChannelData[ch][sample] = value;
        }
      }
      cursor = sourceIndex + 1;
    }
  }

  playbackCursorSamples_.store(cursor);

  if (reachedEnd) {
    previewEngine_.stop();
    juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this)]() {
      if (safeThis == nullptr) {
        return;
      }
      safeThis->transportController_.pause();
      safeThis->statusLabel_.setText("Preview reached end", juce::dontSendNotification);
    });
  }
}

void MainComponent::audioDeviceAboutToStart(juce::AudioIODevice* device) {
  juce::ignoreUnused(device);
  playbackCursorSamples_.store(0);
}

void MainComponent::audioDeviceStopped() {}


} // namespace automix::app
