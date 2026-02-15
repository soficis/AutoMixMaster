#include "app/MainComponent.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "ai/AutoMasterStrategyAI.h"
#include "ai/AutoMixStrategyAI.h"
#include "ai/IModelInference.h"
#include "ai/OnnxModelInference.h"
#include "ai/RtNeuralInference.h"
#include "ai/StemSeparator.h"
#include "automaster/OriginalMixReference.h"
#include "analysis/StemHealthAssistant.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/BatchQueueRunner.h"
#include "engine/OfflineRenderPipeline.h"
#include "renderers/ExternalLimiterRenderer.h"
#include "renderers/RendererFactory.h"
#include "util/LameDownloader.h"
#include "util/WavWriter.h"

namespace automix::app {
namespace {

juce::String toJuceText(const std::vector<std::string>& lines) {
  juce::String output;
  for (const auto& line : lines) {
    output += juce::String(line);
    output += "\n";
  }
  return output;
}

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

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string extensionForFormat(const std::string& format) {
  const auto normalized = toLower(format);
  if (normalized == "wav") {
    return ".wav";
  }
  if (normalized == "flac") {
    return ".flac";
  }
  if (normalized == "aiff" || normalized == "aif") {
    return ".aiff";
  }
  if (normalized == "ogg" || normalized == "vorbis") {
    return ".ogg";
  }
  if (normalized == "mp3") {
    return ".mp3";
  }
  return ".wav";
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

std::unique_ptr<ai::IModelInference> createInferenceBackend(const ai::ModelPack* pack,
                                                            const std::string& providerPreference,
                                                            std::string* diagnosticsOut) {
  if (pack == nullptr) {
    return nullptr;
  }

  std::unique_ptr<ai::IModelInference> backend;
  const auto engine = toLower(pack->engine);
  if (engine.find("onnx") != std::string::npos || engine == "unknown") {
    auto onnx = std::make_unique<ai::OnnxModelInference>();

    auto resolvedProvider = providerPreference;
    if ((resolvedProvider.empty() || toLower(resolvedProvider) == "auto") && !pack->providerAffinity.empty()) {
      resolvedProvider = pack->providerAffinity.front();
    }
    onnx->setExecutionProviderPreference(resolvedProvider);
    onnx->setGraphOptimizationEnabled(true);
    onnx->setWarmupEnabled(true);
    onnx->setPreferQuantizedVariants(toLower(pack->preferredPrecision) != "fp32");
    onnx->setPreferredPrecision(pack->preferredPrecision.empty() ? "auto" : pack->preferredPrecision);
    onnx->setThreadConfiguration(pack->defaultIntraOpThreads.value_or(0), pack->defaultInterOpThreads.value_or(0));
    onnx->setProfilingEnabled(pack->enableProfiling);
    backend = std::move(onnx);
  }
  if (!backend && engine.find("rtneural") != std::string::npos) {
    backend = std::make_unique<ai::RtNeuralInference>();
  }
  if (!backend) {
    backend = std::make_unique<ai::RtNeuralInference>();
  }

  const auto modelPath = pack->rootPath / pack->modelFile;
  if (!backend->loadModel(modelPath)) {
    return nullptr;
  }

  if (diagnosticsOut != nullptr) {
    if (const auto* onnx = dynamic_cast<const ai::OnnxModelInference*>(backend.get()); onnx != nullptr) {
      *diagnosticsOut = onnx->backendDiagnostics();
    }
  }

  return backend;
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

struct ExportHealthCacheEntry {
  std::string key;
  juce::String text;
  bool hasCriticalIssues = false;
  size_t issueCount = 0;
};

std::mutex& exportHealthCacheMutex() {
  static std::mutex mutex;
  return mutex;
}

std::optional<ExportHealthCacheEntry>& exportHealthCache() {
  static std::optional<ExportHealthCacheEntry> cache;
  return cache;
}

std::string buildExportHealthCacheKey(const domain::Session& session,
                                      const std::vector<analysis::StemAnalysisEntry>& analysisEntries) {
  std::ostringstream key;
  key << "stems=" << session.stems.size() << "|entries=" << analysisEntries.size() << '|';

  for (const auto& stem : session.stems) {
    key << stem.id << ':' << stem.filePath << ':' << stem.enabled;
    if (stem.busId.has_value()) {
      key << ':' << stem.busId.value();
    }

    std::error_code error;
    const auto writeTime = std::filesystem::last_write_time(stem.filePath, error);
    if (!error) {
      const auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(writeTime.time_since_epoch()).count();
      key << ':' << ticks;
    }
    const auto size = std::filesystem::file_size(stem.filePath, error);
    if (!error) {
      key << ':' << size;
    }
    key << '|';
  }

  if (session.mixPlan.has_value()) {
    key << "mixPlan:" << session.mixPlan->dryWet
        << ':' << session.mixPlan->mixBusHeadroomDb
        << ':' << session.mixPlan->stemDecisions.size();
  } else {
    key << "mixPlan:none";
  }

  return key.str();
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
  stopTimer();
  audioDeviceManager_.removeAudioCallback(this);
  transportController_.removeChangeListener(this);

  importButton_.removeListener(this);
  originalMixButton_.removeListener(this);
  clearOriginalMixButton_.removeListener(this);
  regenerateCacheButton_.removeListener(this);
  saveSessionButton_.removeListener(this);
  loadSessionButton_.removeListener(this);
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

void MainComponent::refreshModelPacks() {
  modelManager_.setRootPath("ModelPacks");
  const auto packs = modelManager_.scan();

  roleModelBox_.clear(juce::dontSendNotification);
  mixModelBox_.clear(juce::dontSendNotification);
  masterModelBox_.clear(juce::dontSendNotification);
  roleModelIdByComboId_.clear();
  mixModelIdByComboId_.clear();
  masterModelIdByComboId_.clear();

  roleModelBox_.addItem("none", 1);
  mixModelBox_.addItem("none", 1);
  masterModelBox_.addItem("none", 1);
  roleModelIdByComboId_[1] = "none";
  mixModelIdByComboId_[1] = "none";
  masterModelIdByComboId_[1] = "none";

  int itemId = 2;
  for (const auto& pack : packs) {
    const juce::String label = pack.id + " [" + pack.engine + "]";
    if (pack.type == "role_classifier") {
      roleModelBox_.addItem(label, itemId);
      roleModelIdByComboId_[itemId] = pack.id;
      ++itemId;
    } else if (pack.type == "mix_parameters") {
      mixModelBox_.addItem(label, itemId);
      mixModelIdByComboId_[itemId] = pack.id;
      ++itemId;
    } else if (pack.type == "master_parameters") {
      masterModelBox_.addItem(label, itemId);
      masterModelIdByComboId_[itemId] = pack.id;
      ++itemId;
    } else {
      roleModelBox_.addItem(label, itemId);
      roleModelIdByComboId_[itemId] = pack.id;
      ++itemId;
      mixModelBox_.addItem(label, itemId);
      mixModelIdByComboId_[itemId] = pack.id;
      ++itemId;
      masterModelBox_.addItem(label, itemId);
      masterModelIdByComboId_[itemId] = pack.id;
      ++itemId;
    }
  }

  roleModelBox_.setSelectedId(1, juce::dontSendNotification);
  mixModelBox_.setSelectedId(1, juce::dontSendNotification);
  masterModelBox_.setSelectedId(1, juce::dontSendNotification);
  modelManager_.setActivePackId("role", "none");
  modelManager_.setActivePackId("mix", "none");
  modelManager_.setActivePackId("master", "none");
}

std::vector<renderers::ExternalRendererConfig> MainComponent::loadConfiguredExternalRenderers() const {
  std::vector<renderers::ExternalRendererConfig> configs;
#ifdef ENABLE_EXTERNAL_TOOL_SUPPORT
  const char* rawValue = std::getenv("AUTOMIX_EXTERNAL_RENDERERS");
  if (rawValue != nullptr && *rawValue != '\0') {
    int index = 1;
    for (const auto& item : splitDelimited(rawValue, ';')) {
      const auto pieces = splitDelimited(item, '|');
      if (pieces.size() < 2) {
        continue;
      }

      renderers::ExternalRendererConfig config;
      config.id = "ExternalUser" + std::to_string(index++);
      config.name = pieces[0];
      config.binaryPath = pieces[1];
      if (pieces.size() >= 3) {
        config.licenseId = pieces[2];
      }
      configs.push_back(config);
    }
  }
#endif
  configs.insert(configs.end(), userExternalRendererConfigs_.begin(), userExternalRendererConfigs_.end());
  return configs;
}

void MainComponent::refreshRenderers() {
  rendererBox_.clear(juce::dontSendNotification);
  rendererIdByComboId_.clear();

  renderers::RendererRegistry registry;
  rendererInfos_ = registry.list(loadConfiguredExternalRenderers());

  int comboId = 1;
  int preferredId = 0;
  for (const auto& info : rendererInfos_) {
    juce::String label = info.name;
    if (info.linkMode == renderers::RendererLinkMode::External) {
      label += " [external]";
    }
    if (!info.available) {
      label += " (unavailable)";
    }
    rendererBox_.addItem(label, comboId);
    rendererIdByComboId_[comboId] = info.id;

    if (preferredId == 0 && info.available) {
      preferredId = comboId;
    }
    if (info.id == session_.renderSettings.rendererName) {
      preferredId = comboId;
    }
    ++comboId;
  }

  if (preferredId == 0 && !rendererIdByComboId_.empty()) {
    preferredId = rendererIdByComboId_.begin()->first;
  }
  if (preferredId != 0) {
    rendererBox_.setSelectedId(preferredId, juce::dontSendNotification);
    const auto selected = rendererIdByComboId_.find(preferredId);
    if (selected != rendererIdByComboId_.end()) {
      session_.renderSettings.rendererName = selected->second;
    }
  }
}

void MainComponent::refreshCodecAvailability() {
  exportFormatBox_.clear(juce::dontSendNotification);
  codecFormatByComboId_.clear();

  const auto availability = util::WavWriter::getAvailableFormats();
  std::vector<std::string> tooltipLines;

  int selectedId = 0;
  int firstAvailableId = 0;
  int comboId = 1;
  for (const auto& entry : availability) {
    juce::String label = juce::String(entry.format).toUpperCase();
    if (!entry.available) {
      label += " (unavailable)";
    }

    exportFormatBox_.addItem(label, comboId);
    codecFormatByComboId_[comboId] = entry.format;
    tooltipLines.push_back(entry.format + ": " + entry.detail);

    if (entry.available && firstAvailableId == 0) {
      firstAvailableId = comboId;
    }
    if (toLower(session_.renderSettings.outputFormat) == toLower(entry.format) && entry.available) {
      selectedId = comboId;
    }
    ++comboId;
  }

  if (selectedId == 0) {
    selectedId = firstAvailableId > 0 ? firstAvailableId : 1;
  }
  exportFormatBox_.setSelectedId(selectedId, juce::dontSendNotification);
  mp3ModeBox_.setSelectedId(session_.renderSettings.mp3UseVbr ? 2 : 1, juce::dontSendNotification);
  mp3VbrSlider_.setValue(session_.renderSettings.mp3VbrQuality, juce::dontSendNotification);

  int exportModeSelectedId = 1;
  for (const auto& [comboId, mode] : exportSpeedModeByComboId_) {
    if (mode == session_.renderSettings.exportSpeedMode) {
      exportModeSelectedId = comboId;
      break;
    }
  }
  exportSpeedModeBox_.setSelectedId(exportModeSelectedId, juce::dontSendNotification);

  exportFormatBox_.setTooltip(toJuceText(tooltipLines));
  updateExportCodecControls();
}

std::string MainComponent::selectedExportSpeedMode() const {
  const auto it = exportSpeedModeByComboId_.find(exportSpeedModeBox_.getSelectedId());
  if (it == exportSpeedModeByComboId_.end()) {
    return kExportSpeedModeFinal;
  }
  return it->second;
}

bool MainComponent::isQuickExportModeSelected() const {
  return selectedExportSpeedMode() == kExportSpeedModeQuick;
}

void MainComponent::applyQuickExportDefaults() {
  const auto availability = util::WavWriter::getAvailableFormats();
  const auto isAvailable = [&availability](const std::string& formatName) {
    const auto entry = std::find_if(availability.begin(), availability.end(), [&](const auto& candidate) {
      return toLower(candidate.format) == toLower(formatName);
    });
    return entry != availability.end() && entry->available;
  };

  int mp3ComboId = 0;
  for (const auto& [comboId, format] : codecFormatByComboId_) {
    if (toLower(format) == "mp3") {
      mp3ComboId = comboId;
      break;
    }
  }

  if (mp3ComboId != 0 && isAvailable("mp3")) {
    exportFormatBox_.setSelectedId(mp3ComboId, juce::dontSendNotification);
    session_.renderSettings.outputFormat = "mp3";
  } else {
    for (const auto& [comboId, formatName] : codecFormatByComboId_) {
      if (isAvailable(formatName)) {
        exportFormatBox_.setSelectedId(comboId, juce::dontSendNotification);
        session_.renderSettings.outputFormat = formatName;
        break;
      }
    }
    statusLabel_.setText("Quick mode: MP3 unavailable, using fallback codec", juce::dontSendNotification);
    appendTaskHistory("Quick mode fallback codec selected (MP3 unavailable)");
  }

  exportBitrateSlider_.setValue(320.0, juce::dontSendNotification);
  mp3ModeBox_.setSelectedId(2, juce::dontSendNotification);
  mp3VbrSlider_.setValue(0.0, juce::dontSendNotification);
  session_.renderSettings.lossyBitrateKbps = 320;
  session_.renderSettings.lossyQuality = 8;
  session_.renderSettings.mp3UseVbr = true;
  session_.renderSettings.mp3VbrQuality = 0;
}

void MainComponent::updateExportCodecControls() {
  const auto formatIt = codecFormatByComboId_.find(exportFormatBox_.getSelectedId());
  const std::string selectedFormat = formatIt != codecFormatByComboId_.end() ? formatIt->second : "wav";
  const bool lossy = util::WavWriter::isLossyFormat(selectedFormat);
  const bool mp3 = toLower(selectedFormat) == "mp3";
  const bool vbr = mp3 && mp3ModeBox_.getSelectedId() == 2;
  const bool quickMode = isQuickExportModeSelected();

  exportFormatBox_.setEnabled(!quickMode);
  exportFormatLabel_.setEnabled(!quickMode);
  exportBitrateSlider_.setEnabled(!quickMode && lossy && !vbr);
  exportBitrateLabel_.setEnabled(!quickMode && lossy && !vbr);
  mp3ModeBox_.setEnabled(!quickMode && mp3);
  mp3ModeLabel_.setEnabled(!quickMode && mp3);
  mp3VbrSlider_.setEnabled(!quickMode && vbr);
  mp3VbrLabel_.setEnabled(!quickMode && vbr);
}

void MainComponent::refreshProjectProfiles() {
  projectProfiles_ = domain::loadProjectProfiles(std::filesystem::current_path());
  projectProfileBox_.clear(juce::dontSendNotification);
  projectProfileIdByComboId_.clear();

  int selectedId = 0;
  int comboId = 1;
  for (const auto& profile : projectProfiles_) {
    projectProfileBox_.addItem(profile.name + " [" + profile.id + "]", comboId);
    projectProfileIdByComboId_[comboId] = profile.id;
    if (profile.id == session_.projectProfileId) {
      selectedId = comboId;
    }
    ++comboId;
  }

  if (selectedId == 0 && !projectProfileIdByComboId_.empty()) {
    selectedId = projectProfileIdByComboId_.begin()->first;
  }

  if (selectedId > 0) {
    projectProfileBox_.setSelectedId(selectedId, juce::dontSendNotification);
    const auto it = projectProfileIdByComboId_.find(selectedId);
    if (it != projectProfileIdByComboId_.end()) {
      if (const auto profile = domain::findProjectProfile(projectProfiles_, it->second); profile.has_value()) {
        applyProjectProfile(profile.value());
      }
    }
  }
}

void MainComponent::applyProjectProfile(const domain::ProjectProfile& profile) {
  session_.projectProfileId = profile.id;
  session_.safetyPolicyId = profile.safetyPolicyId;
  session_.preferredStemCount = profile.preferredStemCount;

  if (profile.gpuProvider == "cpu") {
    gpuProviderBox_.setSelectedId(2, juce::dontSendNotification);
  } else if (profile.gpuProvider == "directml") {
    gpuProviderBox_.setSelectedId(3, juce::dontSendNotification);
  } else if (profile.gpuProvider == "coreml") {
    gpuProviderBox_.setSelectedId(4, juce::dontSendNotification);
  } else if (profile.gpuProvider == "cuda") {
    gpuProviderBox_.setSelectedId(5, juce::dontSendNotification);
  } else {
    gpuProviderBox_.setSelectedId(1, juce::dontSendNotification);
  }
  session_.renderSettings.gpuExecutionProvider = profile.gpuProvider;

  for (const auto& [comboId, format] : codecFormatByComboId_) {
    if (toLower(format) == toLower(profile.outputFormat)) {
      exportFormatBox_.setSelectedId(comboId, juce::dontSendNotification);
      break;
    }
  }
  exportBitrateSlider_.setValue(profile.lossyBitrateKbps, juce::dontSendNotification);
  mp3ModeBox_.setSelectedId(profile.mp3UseVbr ? 2 : 1, juce::dontSendNotification);
  mp3VbrSlider_.setValue(profile.mp3VbrQuality, juce::dontSendNotification);
  session_.renderSettings.outputFormat = profile.outputFormat;
  session_.renderSettings.lossyBitrateKbps = profile.lossyBitrateKbps;
  session_.renderSettings.mp3UseVbr = profile.mp3UseVbr;
  session_.renderSettings.mp3VbrQuality = profile.mp3VbrQuality;
  session_.renderSettings.exportSpeedMode = selectedExportSpeedMode();
  if (isQuickExportModeSelected()) {
    applyQuickExportDefaults();
  }
  updateExportCodecControls();

  for (const auto& [comboId, rendererId] : rendererIdByComboId_) {
    if (rendererId == profile.rendererName) {
      rendererBox_.setSelectedId(comboId, juce::dontSendNotification);
      session_.renderSettings.rendererName = rendererId;
      break;
    }
  }

  const auto selectModelComboById = [](juce::ComboBox& combo,
                                       const std::map<int, std::string>& idsByCombo,
                                       const std::string& modelId) {
    for (const auto& [comboId, id] : idsByCombo) {
      if (id == modelId) {
        combo.setSelectedId(comboId, juce::dontSendNotification);
        return;
      }
    }
    combo.setSelectedId(1, juce::dontSendNotification);
  };

  selectModelComboById(roleModelBox_, roleModelIdByComboId_, profile.roleModelPackId);
  selectModelComboById(mixModelBox_, mixModelIdByComboId_, profile.mixModelPackId);
  selectModelComboById(masterModelBox_, masterModelIdByComboId_, profile.masterModelPackId);

  modelManager_.setActivePackId("role", profile.roleModelPackId);
  modelManager_.setActivePackId("mix", profile.mixModelPackId);
  modelManager_.setActivePackId("master", profile.masterModelPackId);

  const auto normalizedPlatform = toLower(profile.platformPreset);
  for (const auto& [comboId, preset] : platformPresetByComboId_) {
    if (toLower(domain::toString(preset)) == normalizedPlatform) {
      platformPresetBox_.setSelectedId(comboId, juce::dontSendNotification);
      break;
    }
  }

  appendTaskHistory("Applied profile " + profile.name + " (safety=" + profile.safetyPolicyId +
                    ", stems=" + std::to_string(profile.preferredStemCount) + ")");
}

void MainComponent::refreshStemRoutingSelectors() {
  const auto previousSolo = stemIdBySoloComboId_.count(soloStemBox_.getSelectedId()) > 0
                                ? stemIdBySoloComboId_[soloStemBox_.getSelectedId()]
                                : std::string();
  const auto previousMute = stemIdByMuteComboId_.count(muteStemBox_.getSelectedId()) > 0
                                ? stemIdByMuteComboId_[muteStemBox_.getSelectedId()]
                                : std::string();

  soloStemBox_.clear(juce::dontSendNotification);
  muteStemBox_.clear(juce::dontSendNotification);
  stemIdBySoloComboId_.clear();
  stemIdByMuteComboId_.clear();

  soloStemBox_.addItem("None", 1);
  muteStemBox_.addItem("None", 1);
  stemIdBySoloComboId_[1] = "";
  stemIdByMuteComboId_[1] = "";

  int comboId = 2;
  int nextSolo = 1;
  int nextMute = 1;
  for (const auto& stem : session_.stems) {
    const auto label = juce::String(stem.name + " [" + stem.id + "]");
    soloStemBox_.addItem(label, comboId);
    muteStemBox_.addItem(label, comboId);
    stemIdBySoloComboId_[comboId] = stem.id;
    stemIdByMuteComboId_[comboId] = stem.id;

    if (stem.id == previousSolo) {
      nextSolo = comboId;
    }
    if (stem.id == previousMute) {
      nextMute = comboId;
    }

    ++comboId;
  }

  soloStemBox_.setSelectedId(nextSolo, juce::dontSendNotification);
  muteStemBox_.setSelectedId(nextMute, juce::dontSendNotification);
}

void MainComponent::rebuildPreviewBuffers() {
  rebuildPreviewBuffersAsync();
}

void MainComponent::applyLoadedSession(domain::Session loadedSession, const juce::String& sourcePath) {
  session_ = std::move(loadedSession);
  if (session_.renderSettings.exportSpeedMode != kExportSpeedModeFinal &&
      session_.renderSettings.exportSpeedMode != kExportSpeedModeBalanced &&
      session_.renderSettings.exportSpeedMode != kExportSpeedModeQuick) {
    session_.renderSettings.exportSpeedMode = kExportSpeedModeFinal;
  }

  residualBlendSlider_.setValue(session_.residualBlend, juce::dontSendNotification);
  clearOriginalMixButton_.setEnabled(session_.originalMixPath.has_value() && !session_.originalMixPath->empty());
  exportBitrateSlider_.setValue(session_.renderSettings.lossyBitrateKbps, juce::dontSendNotification);
  mp3ModeBox_.setSelectedId(session_.renderSettings.mp3UseVbr ? 2 : 1, juce::dontSendNotification);
  mp3VbrSlider_.setValue(session_.renderSettings.mp3VbrQuality, juce::dontSendNotification);

  int exportModeSelectedId = 1;
  for (const auto& [comboId, mode] : exportSpeedModeByComboId_) {
    if (mode == session_.renderSettings.exportSpeedMode) {
      exportModeSelectedId = comboId;
      break;
    }
  }
  exportSpeedModeBox_.setSelectedId(exportModeSelectedId, juce::dontSendNotification);

  zoomSlider_.setValue(session_.timeline.zoom, juce::dontSendNotification);
  fineScrubToggle_.setToggleState(session_.timeline.fineScrub, juce::dontSendNotification);

  if (session_.renderSettings.gpuExecutionProvider == "cpu") {
    gpuProviderBox_.setSelectedId(2, juce::dontSendNotification);
  } else if (session_.renderSettings.gpuExecutionProvider == "directml") {
    gpuProviderBox_.setSelectedId(3, juce::dontSendNotification);
  } else if (session_.renderSettings.gpuExecutionProvider == "coreml") {
    gpuProviderBox_.setSelectedId(4, juce::dontSendNotification);
  } else if (session_.renderSettings.gpuExecutionProvider == "cuda") {
    gpuProviderBox_.setSelectedId(5, juce::dontSendNotification);
  } else {
    gpuProviderBox_.setSelectedId(1, juce::dontSendNotification);
  }

  if (rendererIdByComboId_.empty()) {
    refreshRenderers();
  }
  if (codecFormatByComboId_.empty()) {
    refreshCodecAvailability();
  }
  if (roleModelIdByComboId_.empty() || mixModelIdByComboId_.empty() || masterModelIdByComboId_.empty()) {
    refreshModelPacks();
  }
  refreshStemRoutingSelectors();

  for (const auto& [comboId, rendererId] : rendererIdByComboId_) {
    if (rendererId == session_.renderSettings.rendererName) {
      rendererBox_.setSelectedId(comboId, juce::dontSendNotification);
      break;
    }
  }

  for (const auto& [comboId, format] : codecFormatByComboId_) {
    if (toLower(format) == toLower(session_.renderSettings.outputFormat)) {
      exportFormatBox_.setSelectedId(comboId, juce::dontSendNotification);
      break;
    }
  }
  for (const auto& [comboId, profileId] : projectProfileIdByComboId_) {
    if (profileId == session_.projectProfileId) {
      projectProfileBox_.setSelectedId(comboId, juce::dontSendNotification);
      break;
    }
  }
  updateExportCodecControls();

  analysisEntries_.clear();
  analysisTableModel_.setEntries(&analysisEntries_);
  analysisTable_.updateContent();
  transportController_.setLoopRangeSeconds(session_.timeline.loopInSeconds,
                                           session_.timeline.loopOutSeconds,
                                           session_.timeline.loopEnabled);
  updateTransportLoopAndZoomUI();

  statusLabel_.setText("Session loaded", juce::dontSendNotification);
  reportEditor_.setText("Loaded session: " + sourcePath);
  appendTaskHistory("Session loaded: " + sourcePath);
  rebuildPreviewBuffersAsync();
}

void MainComponent::rebuildPreviewBuffersAsync() {
  if (session_.stems.empty()) {
    ++previewBuildGeneration_;
    waveformPreview_.setBuffer(engine::AudioBuffer{});
    transportController_.setTimeline(0, 44100.0);
    return;
  }

  const auto generation = ++previewBuildGeneration_;
  auto previewSession = session_;
  const auto soloIt = stemIdBySoloComboId_.find(soloStemBox_.getSelectedId());
  const auto muteIt = stemIdByMuteComboId_.find(muteStemBox_.getSelectedId());
  const auto soloStemId = soloIt != stemIdBySoloComboId_.end() ? soloIt->second : std::string();
  const auto muteStemId = muteIt != stemIdByMuteComboId_.end() ? muteIt->second : std::string();
  const auto previousProgress = transportController_.progress();

  juce::Component::SafePointer<MainComponent> safeThis(this);
  std::thread([safeThis,
               generation,
               previewSession = std::move(previewSession),
               soloStemId,
               muteStemId,
               previousProgress]() mutable {
    try {
      if (!soloStemId.empty()) {
        for (auto& stem : previewSession.stems) {
          stem.enabled = stem.id == soloStemId;
        }
      }
      if (!muteStemId.empty()) {
        for (auto& stem : previewSession.stems) {
          if (stem.id == muteStemId) {
            stem.enabled = false;
          }
        }
      }

      auto settings = previewSession.renderSettings;
      settings.outputSampleRate = settings.outputSampleRate > 0 ? settings.outputSampleRate : 44100;
      settings.blockSize = settings.blockSize > 0 ? settings.blockSize : 1024;
      settings.outputBitDepth = std::clamp(settings.outputBitDepth, 16, 32);
      settings.rendererName = "BuiltIn";
      settings.outputPath.clear();
      settings.outputFormat = "wav";

      engine::OfflineRenderPipeline pipeline;
      const auto raw = pipeline.renderRawMix(previewSession, settings, {}, nullptr);
      if (raw.cancelled || raw.mixBuffer.getNumSamples() == 0) {
        return;
      }

      auto mastered = raw.mixBuffer;
      if (previewSession.masterPlan.has_value()) {
        automaster::HeuristicAutoMasterStrategy strategy;
        mastered = strategy.applyPlan(raw.mixBuffer, previewSession.masterPlan.value(), nullptr);
      }

      if (safeThis == nullptr || generation != safeThis->previewBuildGeneration_.load()) {
        return;
      }

      engine::AudioPreviewEngine previewEngine;
      previewEngine.setBuffers(raw.mixBuffer, mastered);
      const auto preview = previewEngine.buildCrossfadedPreview(1024);

      juce::MessageManager::callAsync(
          [safeThis,
           generation,
           rawMix = raw.mixBuffer,
           mastered = std::move(mastered),
           preview = std::move(preview),
           previousProgress]() mutable {
            if (safeThis == nullptr) {
              return;
            }
            if (generation != safeThis->previewBuildGeneration_.load()) {
              return;
            }

            safeThis->previewEngine_.setBuffers(rawMix, mastered);
            safeThis->updateTransportFromBuffer(preview);
            safeThis->transportController_.seekToFraction(previousProgress);
            safeThis->playbackCursorSamples_.store(safeThis->transportController_.positionSamples());
          });
    } catch (const std::exception& error) {
      const auto message = juce::String(error.what());
      juce::MessageManager::callAsync([safeThis, generation, message]() {
        if (safeThis == nullptr) {
          return;
        }
        if (generation != safeThis->previewBuildGeneration_.load()) {
          return;
        }
        safeThis->reportEditor_.setText(safeThis->reportEditor_.getText() + "\nPreview rebuild skipped: " + message);
      });
    }
  }).detach();
}

void MainComponent::updateTransportFromBuffer(const engine::AudioBuffer& buffer) {
  {
    std::scoped_lock lock(playbackBufferMutex_);
    playbackBuffer_ = buffer;
  }
  playbackCursorSamples_.store(0);
  waveformPreview_.setBuffer(buffer);
  waveformPreview_.setPlayheadProgress(0.0);
  transportController_.setTimeline(buffer.getNumSamples(), buffer.getSampleRate());
  transportController_.setLoopRangeSeconds(session_.timeline.loopInSeconds,
                                           session_.timeline.loopOutSeconds,
                                           session_.timeline.loopEnabled);
  transportController_.stop();
  updateTransportLoopAndZoomUI();

  ignoreTransportSliderChange_ = true;
  transportSlider_.setValue(0.0, juce::dontSendNotification);
  ignoreTransportSliderChange_ = false;
}

void MainComponent::updateTransportDisplay() {
  const auto progress = transportController_.progress();

  ignoreTransportSliderChange_ = true;
  transportSlider_.setValue(progress, juce::dontSendNotification);
  ignoreTransportSliderChange_ = false;

  waveformPreview_.setPlayheadProgress(progress);
  updateTransportLoopAndZoomUI();

  if (transportController_.state() == engine::TransportController::State::Playing) {
    playPauseButton_.setButtonText("Pause");
  } else {
    playPauseButton_.setButtonText("Play");
  }

  const auto positionText = formatDuration(transportController_.positionSeconds());
  const auto totalText = formatDuration(transportController_.totalSeconds());
  juce::String tooltip = positionText + " / " + totalText;
  if (transportController_.loopEnabled()) {
    tooltip += " [Loop " + juce::String(formatDuration(transportController_.loopInSeconds())) +
               " - " + juce::String(formatDuration(transportController_.loopOutSeconds())) + "]";
  }
  transportSlider_.setTooltip(tooltip);
}

void MainComponent::updateTransportLoopAndZoomUI() {
  const double zoom = std::clamp(session_.timeline.zoom, 1.0, 32.0);
  waveformPreview_.setZoom(zoom, transportController_.progress());
  waveformPreview_.setLoopRange(transportController_.loopEnabled(),
                                transportController_.loopInProgress(),
                                transportController_.loopOutProgress());
}

void MainComponent::appendTaskHistory(const juce::String& line) {
  const auto timestamp = juce::Time::getCurrentTime().toString(true, true);
  const auto entry = "[" + timestamp + "] " + line;
  taskHistoryLines_.push_back(entry);
  constexpr size_t kMaxTaskHistory = 120;
  bool trimmed = false;
  if (taskHistoryLines_.size() > kMaxTaskHistory) {
    taskHistoryLines_.erase(taskHistoryLines_.begin(),
                            taskHistoryLines_.begin() + static_cast<long>(taskHistoryLines_.size() - kMaxTaskHistory));
    trimmed = true;
  }

  const auto currentText = taskCenterEditor_.getText();
  if (!trimmed && currentText.isNotEmpty() && currentText != "Task history will appear here.") {
    taskCenterEditor_.moveCaretToEnd(false);
    taskCenterEditor_.insertTextAtCaret(entry + "\n");
    return;
  }

  juce::String rebuiltText;
  for (const auto& item : taskHistoryLines_) {
    rebuiltText += item;
    rebuiltText += "\n";
  }
  taskCenterEditor_.setText(rebuiltText, false);
}

void MainComponent::populateMasterPresetSelectors() {
  masterPresetBox_.clear(juce::dontSendNotification);
  platformPresetBox_.clear(juce::dontSendNotification);
  masterPresetByComboId_.clear();
  platformPresetByComboId_.clear();

  int masterId = 1;
  auto addMasterPreset = [&](const juce::String& label, const domain::MasterPreset preset) {
    masterPresetBox_.addItem(label, masterId);
    masterPresetByComboId_[masterId] = preset;
    ++masterId;
  };

  addMasterPreset("Default Streaming", domain::MasterPreset::DefaultStreaming);
  addMasterPreset("Broadcast", domain::MasterPreset::Broadcast);
  addMasterPreset("Udio Optimized", domain::MasterPreset::UdioOptimized);
  addMasterPreset("Custom", domain::MasterPreset::Custom);

  int platformId = 1;
  auto addPlatformPreset = [&](const juce::String& label, const domain::MasterPreset preset) {
    platformPresetBox_.addItem(label, platformId);
    platformPresetByComboId_[platformId] = preset;
    ++platformId;
  };

  addPlatformPreset("Spotify", domain::MasterPreset::Spotify);
  addPlatformPreset("Apple Music", domain::MasterPreset::AppleMusic);
  addPlatformPreset("YouTube", domain::MasterPreset::YouTube);
  addPlatformPreset("Amazon Music", domain::MasterPreset::AmazonMusic);
  addPlatformPreset("Tidal", domain::MasterPreset::Tidal);
  addPlatformPreset("Broadcast EBU R128", domain::MasterPreset::BroadcastEbuR128);

  masterPresetBox_.setSelectedId(1, juce::dontSendNotification);
  platformPresetBox_.setSelectedId(1, juce::dontSendNotification);
}

domain::MasterPreset MainComponent::selectedMasterPreset() const {
  const auto it = masterPresetByComboId_.find(masterPresetBox_.getSelectedId());
  if (it == masterPresetByComboId_.end()) {
    return domain::MasterPreset::DefaultStreaming;
  }
  return it->second;
}

domain::MasterPreset MainComponent::selectedPlatformPreset() const {
  const auto it = platformPresetByComboId_.find(platformPresetBox_.getSelectedId());
  if (it == platformPresetByComboId_.end()) {
    return domain::MasterPreset::DefaultStreaming;
  }
  return it->second;
}

domain::RenderSettings MainComponent::buildCurrentRenderSettings(const std::string& outputPath) const {
  domain::RenderSettings settings;
  settings.exportSpeedMode = selectedExportSpeedMode();
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;
  settings.outputBitDepth = 24;
  if (settings.exportSpeedMode == kExportSpeedModeBalanced) {
    settings.blockSize = 2048;
  } else if (settings.exportSpeedMode == kExportSpeedModeQuick) {
    settings.blockSize = 4096;
    settings.outputBitDepth = 16;
  }
  settings.processingThreads = 0;
  settings.preferHardwareAcceleration = true;

  const auto formatIt = codecFormatByComboId_.find(exportFormatBox_.getSelectedId());
  settings.outputFormat = formatIt != codecFormatByComboId_.end() ? formatIt->second : "wav";

  settings.lossyBitrateKbps = std::clamp(static_cast<int>(std::lround(exportBitrateSlider_.getValue())), 64, 320);
  settings.lossyQuality =
      std::clamp(static_cast<int>(std::lround((static_cast<double>(settings.lossyBitrateKbps) - 64.0) / 25.6)), 0, 10);
  settings.mp3UseVbr = toLower(settings.outputFormat) == "mp3" && mp3ModeBox_.getSelectedId() == 2;
  settings.mp3VbrQuality = std::clamp(static_cast<int>(std::lround(mp3VbrSlider_.getValue())), 0, 9);

  if (settings.exportSpeedMode == kExportSpeedModeQuick) {
    const auto availability = util::WavWriter::getAvailableFormats();
    const auto hasAvailableMp3 = std::find_if(availability.begin(), availability.end(), [](const auto& entry) {
      return toLower(entry.format) == "mp3" && entry.available;
    }) != availability.end();

    if (hasAvailableMp3) {
      settings.outputFormat = "mp3";
      settings.lossyBitrateKbps = 320;
      settings.lossyQuality = 8;
      settings.mp3UseVbr = true;
      settings.mp3VbrQuality = 0;
    }
  }

  switch (gpuProviderBox_.getSelectedId()) {
    case 2:
      settings.gpuExecutionProvider = "cpu";
      break;
    case 3:
      settings.gpuExecutionProvider = "directml";
      break;
    case 4:
      settings.gpuExecutionProvider = "coreml";
      break;
    case 5:
      settings.gpuExecutionProvider = "cuda";
      break;
    default:
      settings.gpuExecutionProvider = "auto";
      break;
  }

  if (!outputPath.empty()) {
    std::filesystem::path normalizedPath(outputPath);
    const auto requiredExtension = extensionForFormat(settings.outputFormat);
    if (toLower(normalizedPath.extension().string()) != requiredExtension) {
      normalizedPath.replace_extension(requiredExtension);
    }
    settings.outputPath = normalizedPath.string();
  }

  settings.rendererName = "BuiltIn";
  const auto rendererIt = rendererIdByComboId_.find(rendererBox_.getSelectedId());
  if (rendererIt != rendererIdByComboId_.end()) {
    settings.rendererName = rendererIt->second;
    for (const auto& info : rendererInfos_) {
      if (info.id == settings.rendererName) {
        settings.externalRendererPath = info.binaryPath.string();
        break;
      }
    }
  }

  return settings;
}

void MainComponent::onCancel() {
  cancelRender_.store(true);
  statusLabel_.setText("Cancelling...", juce::dontSendNotification);
  appendTaskHistory("Cancellation requested");
}

void MainComponent::onImport() {
  if (taskRunning_.load()) {
    statusLabel_.setText("A render task is already running", juce::dontSendNotification);
    return;
  }

  importChooser_ =
      std::make_unique<juce::FileChooser>("Select stem files", juce::File(), "*.wav;*.aiff;*.aif;*.flac;*.mp3;*.ogg");
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
    for (int i = 0; i < files.size(); ++i) {
      selectedFiles.push_back(files.getReference(i));
    }

    const bool useSeparation = separatedStemsToggle_.getToggleState();
    const int preferredStemCount = session_.preferredStemCount;
    statusLabel_.setText("Importing files...", juce::dontSendNotification);
    appendTaskHistory("Import started");

    juce::Component::SafePointer<MainComponent> safeThis(this);
    std::thread([safeThis, selectedFiles = std::move(selectedFiles), useSeparation, preferredStemCount]() mutable {
      std::vector<domain::Stem> importedStems;
      std::vector<std::string> importLines;

      if (selectedFiles.size() == 1 && useSeparation) {
        try {
          const auto mixPath = std::filesystem::path(selectedFiles.front().getFullPathName().toStdString());
          const auto outputDir = mixPath.parent_path() / (mixPath.stem().string() + "_separated");

          ai::StemSeparator separator;
          ai::StemSeparator::SeparationOptions separationOptions;
          separationOptions.targetStemCount = preferredStemCount;
          const auto separationResult = separator.separate(mixPath, outputDir, separationOptions);
          if (separationResult.success) {
            importedStems = separationResult.stems;
            importLines.push_back("Separated import from: " + mixPath.string());
            importLines.push_back("Variant stems: " + std::to_string(separationResult.stemVariantCount));
            for (const auto& stem : separationResult.stems) {
              std::string line = "  stem -> " + stem.filePath + " role=" + domain::toString(stem.role);
              if (stem.separationConfidence.has_value()) {
                line += " confidence=" + std::to_string(stem.separationConfidence.value());
              }
              if (stem.separationArtifactRisk.has_value()) {
                line += " artifactRisk=" + std::to_string(stem.separationArtifactRisk.value());
              }
              importLines.push_back(line);
            }
            if (!separationResult.qaReportPath.empty()) {
              importLines.push_back("Separation QA report: " + separationResult.qaReportPath.string());
            }
            importLines.push_back("QA energyLeakage=" + std::to_string(separationResult.qaMetrics.energyLeakage) +
                                  " residualDistortion=" + std::to_string(separationResult.qaMetrics.residualDistortion) +
                                  " transientRetention=" + std::to_string(separationResult.qaMetrics.transientRetention));
            importLines.push_back(separationResult.logMessage);
          } else {
            importLines.push_back("Separation failed, importing original mix file as stem.");
            importLines.push_back(separationResult.logMessage);
          }
        } catch (const std::exception& error) {
          importLines.push_back("Separation error: " + std::string(error.what()));
        } catch (...) {
          importLines.push_back("Separation error: unknown failure");
        }
      }

      if (importedStems.empty()) {
        for (size_t i = 0; i < selectedFiles.size(); ++i) {
          const auto& file = selectedFiles[i];

          domain::Stem stem;
          stem.id = "stem_" + std::to_string(i + 1);
          stem.name = file.getFileNameWithoutExtension().toStdString();
          stem.filePath = file.getFullPathName().toStdString();
          stem.origin = useSeparation ? domain::StemOrigin::Separated : domain::StemOrigin::Recorded;
          stem.enabled = true;
          importedStems.push_back(stem);

          importLines.push_back(stem.name + " -> " + stem.filePath);
        }
      }

      juce::MessageManager::callAsync(
          [safeThis, importedStems = std::move(importedStems), importLines = std::move(importLines)]() mutable {
            if (safeThis == nullptr) {
              return;
            }

            safeThis->session_.stems = std::move(importedStems);
            safeThis->statusLabel_.setText(
                "Imported " + juce::String(static_cast<int>(safeThis->session_.stems.size())) + " stems",
                juce::dontSendNotification);
            safeThis->appendTaskHistory("Imported " + juce::String(static_cast<int>(safeThis->session_.stems.size())) + " stems");

            safeThis->analysisEntries_.clear();
            safeThis->analysisTableModel_.setEntries(&safeThis->analysisEntries_);
            safeThis->analysisTable_.updateContent();

            safeThis->refreshStemRoutingSelectors();
            safeThis->reportEditor_.setText(juce::String("Imported files:\n") + toJuceText(importLines));
            safeThis->rebuildPreviewBuffersAsync();
          });
    }).detach();

    importChooser_.reset();
  });
}

void MainComponent::onImportOriginalMix() {
  originalMixChooser_ =
      std::make_unique<juce::FileChooser>("Select original stereo mix",
                                          juce::File(),
                                          "*.wav;*.aiff;*.aif;*.flac;*.mp3;*.ogg");
  constexpr int flags = juce::FileBrowserComponent::openMode |
                        juce::FileBrowserComponent::canSelectFiles;

  originalMixChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      originalMixChooser_.reset();
      return;
    }

    session_.originalMixPath = selected.getFullPathName().toStdString();
    clearOriginalMixButton_.setEnabled(true);
    statusLabel_.setText("Original mix loaded", juce::dontSendNotification);
    reportEditor_.setText(reportEditor_.getText() + "\nOriginal mix: " + selected.getFullPathName());
    appendTaskHistory("Original mix loaded: " + selected.getFileName());
    originalMixChooser_.reset();
  });
}

void MainComponent::onClearOriginalMix() {
  if (!session_.originalMixPath.has_value()) {
    statusLabel_.setText("No original mix is configured", juce::dontSendNotification);
    clearOriginalMixButton_.setEnabled(false);
    return;
  }

  session_.originalMixPath.reset();
  clearOriginalMixButton_.setEnabled(false);
  statusLabel_.setText("Original mix cleared", juce::dontSendNotification);
  reportEditor_.setText(reportEditor_.getText() + "\nOriginal mix cleared");
  appendTaskHistory("Original mix configuration cleared");
}

void MainComponent::onRegenerateCachedRenders() {
  engine::OfflineRenderPipeline::clearCaches();
  {
    std::scoped_lock lock(exportHealthCacheMutex());
    exportHealthCache().reset();
  }

  statusLabel_.setText("Render caches cleared", juce::dontSendNotification);
  appendTaskHistory("Render caches cleared; next render will regenerate intermediates");

  if (!session_.stems.empty()) {
    rebuildPreviewBuffersAsync();
  }
}

void MainComponent::onSaveSession() {
  saveSessionChooser_ = std::make_unique<juce::FileChooser>(
      "Save session", juce::File::getSpecialLocation(juce::File::userDocumentsDirectory), "*.json");
  constexpr int flags = juce::FileBrowserComponent::saveMode |
                        juce::FileBrowserComponent::canSelectFiles |
                        juce::FileBrowserComponent::warnAboutOverwriting;

  saveSessionChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      saveSessionChooser_.reset();
      return;
    }

    try {
      session_.renderSettings = buildCurrentRenderSettings(session_.renderSettings.outputPath);
      session_.timeline.zoom = zoomSlider_.getValue();
      session_.timeline.fineScrub = fineScrubToggle_.getToggleState();
      session_.timeline.loopEnabled = transportController_.loopEnabled();
      session_.timeline.loopInSeconds = transportController_.loopInSeconds();
      session_.timeline.loopOutSeconds = transportController_.loopOutSeconds();
      sessionRepository_.save(selected.getFullPathName().toStdString(), session_);
      statusLabel_.setText("Session saved", juce::dontSendNotification);
      appendTaskHistory("Session saved: " + selected.getFullPathName());
    } catch (const std::exception& error) {
      statusLabel_.setText("Save failed", juce::dontSendNotification);
      reportEditor_.setText("Session save error:\n" + juce::String(error.what()));
      appendTaskHistory("Session save failed");
    }

    saveSessionChooser_.reset();
  });
}

void MainComponent::onLoadSession() {
  loadSessionChooser_ = std::make_unique<juce::FileChooser>("Load session", juce::File(), "*.json");
  constexpr int flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

  loadSessionChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      loadSessionChooser_.reset();
      return;
    }

    const auto selectedPath = selected.getFullPathName().toStdString();
    statusLabel_.setText("Loading session...", juce::dontSendNotification);
    appendTaskHistory("Session load started: " + selected.getFullPathName());

    juce::Component::SafePointer<MainComponent> safeThis(this);
    std::thread([safeThis, selectedPath]() {
      engine::SessionRepository repository;
      std::optional<domain::Session> loadedSession;
      juce::String errorMessage;

      try {
        loadedSession = repository.load(selectedPath);
      } catch (const std::exception& error) {
        errorMessage = error.what();
      } catch (...) {
        errorMessage = "Unknown session load error";
      }

      juce::MessageManager::callAsync([safeThis, selectedPath, loadedSession = std::move(loadedSession), errorMessage]() mutable {
        if (safeThis == nullptr) {
          return;
        }

        if (loadedSession.has_value()) {
          safeThis->applyLoadedSession(std::move(loadedSession.value()), selectedPath);
        } else {
          safeThis->statusLabel_.setText("Load failed", juce::dontSendNotification);
          safeThis->reportEditor_.setText("Session load error:\n" + errorMessage);
          safeThis->appendTaskHistory("Session load failed");
        }
      });
    }).detach();

    loadSessionChooser_.reset();
  });
}

void MainComponent::onPreviewOriginal() {
  if (transportController_.totalSamples() == 0) {
    rebuildPreviewBuffersAsync();
    statusLabel_.setText("Building preview...", juce::dontSendNotification);
    return;
  }

  const auto progress = transportController_.progress();
  previewEngine_.setSource(engine::PreviewSource::OriginalMix);
  const auto preview = previewEngine_.buildCrossfadedPreview(1024);
  updateTransportFromBuffer(preview);
  transportController_.seekToFraction(progress);
  playbackCursorSamples_.store(transportController_.positionSamples());
  transportController_.play();
  previewEngine_.play();

  statusLabel_.setText("Preview A selected", juce::dontSendNotification);
  appendTaskHistory("Preview source switched to Original (A)");
}

void MainComponent::onPreviewRendered() {
  if (transportController_.totalSamples() == 0) {
    rebuildPreviewBuffersAsync();
    statusLabel_.setText("Building preview...", juce::dontSendNotification);
    return;
  }

  const auto progress = transportController_.progress();
  previewEngine_.setSource(engine::PreviewSource::RenderedMix);
  const auto preview = previewEngine_.buildCrossfadedPreview(1024);
  updateTransportFromBuffer(preview);
  transportController_.seekToFraction(progress);
  playbackCursorSamples_.store(transportController_.positionSamples());
  transportController_.play();
  previewEngine_.play();

  statusLabel_.setText("Preview B selected", juce::dontSendNotification);
  appendTaskHistory("Preview source switched to Rendered (B)");
}

void MainComponent::onAddExternalRenderer() {
  externalRendererChooser_ = std::make_unique<juce::FileChooser>("Select external limiter binary");
  constexpr int flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

  externalRendererChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      externalRendererChooser_.reset();
      return;
    }

    const auto selectedPath = selected.getFullPathName().toStdString();
    const auto selectedName = selected.getFileName().toStdString();
    statusLabel_.setText("Validating external renderer...", juce::dontSendNotification);
    appendTaskHistory("External renderer validation started: " + juce::String(selectedName));

    juce::Component::SafePointer<MainComponent> safeThis(this);
    std::thread([safeThis, selectedPath, selectedName]() {
      const auto validation = renderers::ExternalLimiterRenderer::validateBinary(selectedPath);
      juce::MessageManager::callAsync([safeThis, selectedPath, selectedName, validation]() {
        if (safeThis == nullptr) {
          return;
        }

        renderers::ExternalRendererConfig config;
        config.id = "ExternalUserUI" + std::to_string(safeThis->userExternalRendererConfigs_.size() + 1);
        config.name = selectedName;
        config.binaryPath = selectedPath;
        config.licenseId = "User-supplied";
        safeThis->userExternalRendererConfigs_.push_back(config);
        safeThis->refreshRenderers();

        const juce::String statusText =
            validation.valid ? "External renderer added" : "External renderer added (validation failed)";
        safeThis->statusLabel_.setText(statusText, juce::dontSendNotification);
        safeThis->appendTaskHistory(statusText);
        safeThis->reportEditor_.setText(safeThis->reportEditor_.getText() +
                                        "\nAdded external renderer: " + juce::String(selectedPath) +
                                        "\nValidation: " + juce::String(validation.valid ? "passed" : "failed") +
                                        " (" + juce::String(validation.diagnostics) + ")" +
                                        "\nLicense note: user-supplied tool is not distributed by this app.");
      });
    }).detach();

    externalRendererChooser_.reset();
  });
}

void MainComponent::onPrefetchLame() {
  if (!util::LameDownloader::isSupportedOnCurrentPlatform()) {
    statusLabel_.setText("LAME downloader is not supported on this platform", juce::dontSendNotification);
    return;
  }

  prefetchLameButton_.setEnabled(false);
  statusLabel_.setText("Prefetching LAME...", juce::dontSendNotification);

  juce::Component::SafePointer<MainComponent> safeThis(this);
  std::thread([safeThis]() {
    const auto result = util::LameDownloader::ensureAvailable();
    juce::MessageManager::callAsync([safeThis, result]() {
      if (safeThis == nullptr) {
        return;
      }

      safeThis->prefetchLameButton_.setEnabled(true);
      safeThis->refreshCodecAvailability();

      if (result.success) {
        safeThis->statusLabel_.setText("LAME ready for MP3 export", juce::dontSendNotification);
        safeThis->appendTaskHistory("LAME prefetch completed");
      } else {
        safeThis->statusLabel_.setText("LAME prefetch failed", juce::dontSendNotification);
        safeThis->appendTaskHistory("LAME prefetch failed");
      }

      juce::String report = safeThis->reportEditor_.getText();
      if (!report.isEmpty()) {
        report += "\n";
      }
      report += "LAME prefetch: ";
      report += result.success ? "success" : "failed";
      if (!result.executablePath.empty()) {
        report += "\nPath: " + juce::String(result.executablePath.string());
      }
      if (!result.detail.empty()) {
        report += "\nDetail: " + juce::String(result.detail);
      }
      safeThis->reportEditor_.setText(report);
    });
  }).detach();
}

void MainComponent::updateMeterPanel(const automaster::MasteringReport& report) {
  meterLufsLabel_.setText("LUFS: " + juce::String(report.integratedLufs, 2), juce::dontSendNotification);
  meterShortTermLabel_.setText("Short-term: " + juce::String(report.shortTermLufs, 2), juce::dontSendNotification);
  meterTruePeakLabel_.setText("True Peak: " + juce::String(report.truePeakDbtp, 2) + " dBTP",
                              juce::dontSendNotification);
}

void MainComponent::onAutoMix() {
  if (taskRunning_.load()) {
    statusLabel_.setText("A render task is already running", juce::dontSendNotification);
    return;
  }

  if (session_.stems.empty()) {
    statusLabel_.setText("Import stems first", juce::dontSendNotification);
    return;
  }

  cancelRender_.store(false);
  taskRunning_.store(true);
  cancelButton_.setEnabled(true);
  statusLabel_.setText("Auto Mix started", juce::dontSendNotification);
  appendTaskHistory("Auto Mix started");

  const auto sessionSnapshot = session_;
  std::optional<ai::ModelPack> mixPack;
  if (const auto* selected = findPackById(modelManager_, modelManager_.activePackId("mix")); selected != nullptr) {
    mixPack = *selected;
  }

  juce::Component::SafePointer<MainComponent> safeThis(this);
  std::thread([safeThis, sessionSnapshot, mixPack]() mutable {
    std::vector<analysis::StemAnalysisEntry> analysisEntries;
    std::optional<domain::MixPlan> plan;
    juce::String reportText;
    juce::String errorText;
    bool cancelled = false;

    auto updateStatus = [safeThis](const juce::String& text) {
      if (safeThis == nullptr) {
        return;
      }
      juce::MessageManager::callAsync([safeThis, text]() {
        if (safeThis == nullptr) {
          return;
        }
        safeThis->statusLabel_.setText(text, juce::dontSendNotification);
        safeThis->appendTaskHistory(text);
      });
    };

    try {
      updateStatus("Auto Mix: analyzing...");
      analysis::StemAnalyzer analyzer;
      analysisEntries = analyzer.analyzeSession(sessionSnapshot);

      if (safeThis != nullptr && safeThis->cancelRender_.load()) {
        cancelled = true;
      }

      if (!cancelled) {
        updateStatus("Auto Mix: building plan...");
        automix::HeuristicAutoMixStrategy heuristicMix;
        const auto heuristicPlan = heuristicMix.buildPlan(sessionSnapshot, analysisEntries, 1.0);
        plan = heuristicPlan;

        ai::AutoMixStrategyAI aiMix;
        std::string backendDiagnostics;
        std::unique_ptr<ai::IModelInference> inference;
        if (mixPack.has_value()) {
          inference = createInferenceBackend(&mixPack.value(), sessionSnapshot.renderSettings.gpuExecutionProvider, &backendDiagnostics);
        }

        if (inference != nullptr) {
          auto aiPlan = aiMix.buildPlan(sessionSnapshot, analysisEntries, heuristicPlan, inference.get());
          if (mixPack.has_value()) {
            aiPlan.decisionLog.push_back("AI pack: " + mixPack->id + " license=" + mixPack->licenseId);
          }
          if (!backendDiagnostics.empty()) {
            aiPlan.decisionLog.push_back("Inference backend: " + backendDiagnostics);
          }
          plan = std::move(aiPlan);
        }

        reportText = juce::String("Analysis report JSON:\n") + analyzer.toJsonReport(analysisEntries);
        if (plan.has_value()) {
          reportText += juce::String("\n\nMix decisions:\n") + toJuceText(plan->decisionLog);
        }
      }
    } catch (const std::exception& error) {
      errorText = "Auto Mix failed:\n" + juce::String(error.what());
    } catch (...) {
      errorText = "Auto Mix failed:\nUnknown error";
    }

    juce::MessageManager::callAsync(
        [safeThis,
         analysisEntries = std::move(analysisEntries),
         plan = std::move(plan),
         reportText,
         errorText,
         cancelled]() mutable {
          if (safeThis == nullptr) {
            return;
          }

          safeThis->taskRunning_.store(false);
          safeThis->cancelButton_.setEnabled(false);

          if (!errorText.isEmpty()) {
            safeThis->statusLabel_.setText("Auto Mix failed", juce::dontSendNotification);
            safeThis->reportEditor_.setText(errorText);
            safeThis->appendTaskHistory("Auto Mix failed");
            return;
          }

          if (cancelled || safeThis->cancelRender_.load()) {
            safeThis->statusLabel_.setText("Auto Mix cancelled", juce::dontSendNotification);
            safeThis->appendTaskHistory("Auto Mix cancelled");
            return;
          }

          safeThis->analysisEntries_ = std::move(analysisEntries);
          safeThis->analysisTableModel_.setEntries(&safeThis->analysisEntries_);
          safeThis->analysisTable_.updateContent();

          if (plan.has_value()) {
            safeThis->session_.mixPlan = plan.value();
          }

          if (!reportText.isEmpty()) {
            safeThis->reportEditor_.setText(reportText);
          }

          safeThis->statusLabel_.setText("Auto Mix plan generated", juce::dontSendNotification);
          safeThis->appendTaskHistory("Auto Mix completed");
          safeThis->rebuildPreviewBuffersAsync();
        });
  }).detach();
}

void MainComponent::onAutoMaster() {
  if (taskRunning_.load()) {
    statusLabel_.setText("A render task is already running", juce::dontSendNotification);
    return;
  }

  if (session_.stems.empty()) {
    statusLabel_.setText("Import stems first", juce::dontSendNotification);
    return;
  }

  cancelRender_.store(false);
  taskRunning_.store(true);
  cancelButton_.setEnabled(true);
  statusLabel_.setText("Auto Master started", juce::dontSendNotification);
  appendTaskHistory("Auto Master started");

  auto preset = selectedPlatformPreset();
  if (preset == domain::MasterPreset::Custom) {
    preset = selectedMasterPreset();
  }

  const auto settings = buildCurrentRenderSettings("");
  const auto sessionSnapshot = session_;
  std::optional<ai::ModelPack> masterPack;
  if (const auto* selected = findPackById(modelManager_, modelManager_.activePackId("master")); selected != nullptr) {
    masterPack = *selected;
  }

  juce::Component::SafePointer<MainComponent> safeThis(this);
  std::thread([safeThis, sessionSnapshot, settings, preset, masterPack]() mutable {
    domain::MasterPlan masterPlan;
    engine::AudioBuffer rawMixBuffer;
    engine::AudioBuffer previewMaster;
    automaster::MasteringReport previewReport;
    juce::String reportAppend;
    juce::String errorText;
    bool cancelled = false;

    try {
      engine::OfflineRenderPipeline pipeline;
      std::atomic_bool* cancelPtr = nullptr;
      if (safeThis != nullptr) {
        cancelPtr = &safeThis->cancelRender_;
      }
      std::mutex progressMutex;
      auto lastProgressEmit = std::chrono::steady_clock::time_point {};
      double lastProgressFraction = -1.0;
      std::string lastProgressStage;

      const auto rawMix = pipeline.renderRawMix(
          sessionSnapshot,
          settings,
          [safeThis, &progressMutex, &lastProgressEmit, &lastProgressFraction, &lastProgressStage](const engine::RenderProgress& progress) {
            if (safeThis == nullptr) {
              return;
            }
            bool emit = false;
            bool stageChanged = false;
            {
              std::scoped_lock lock(progressMutex);
              const auto now = std::chrono::steady_clock::now();
              stageChanged = progress.stage != lastProgressStage;
              const bool finalProgress = progress.fraction >= 0.999;
              const bool timeGateOpen =
                  lastProgressEmit.time_since_epoch().count() == 0 ||
                  now - lastProgressEmit >= std::chrono::milliseconds(160);
              const bool deltaGateOpen = std::abs(progress.fraction - lastProgressFraction) >= 0.02;
              emit = stageChanged || finalProgress || (timeGateOpen && deltaGateOpen);
              if (emit) {
                lastProgressEmit = now;
                lastProgressFraction = progress.fraction;
                lastProgressStage = progress.stage;
              }
            }
            if (!emit) {
              return;
            }

            juce::MessageManager::callAsync([safeThis, progress]() {
              if (safeThis == nullptr) {
                return;
              }
              const bool cacheHit = progress.stage == "Mix render cache hit";
              if (cacheHit) {
                safeThis->statusLabel_.setText("Auto Master: Using cached mix render (fast path)",
                                               juce::dontSendNotification);
                safeThis->appendTaskHistory("Auto Master using cached mix render");
                return;
              }

              safeThis->statusLabel_.setText("Auto Master: " + juce::String(progress.stage) + " " +
                                                 juce::String(progress.fraction * 100.0, 1) + "%",
                                             juce::dontSendNotification);
              if (progress.fraction >= 0.999 || progress.stage != "Summing stem buses") {
                safeThis->appendTaskHistory("Auto Master " + juce::String(progress.stage) + " " +
                                            juce::String(progress.fraction * 100.0, 1) + "%");
              }
            });
          },
          cancelPtr);

      if (rawMix.cancelled) {
        cancelled = true;
      } else {
        rawMixBuffer = rawMix.mixBuffer;
      }

      if (!cancelled) {
        automaster::HeuristicAutoMasterStrategy autoMasterStrategy;
        analysis::StemAnalyzer analyzer;
        masterPlan = autoMasterStrategy.buildPlan(preset, rawMixBuffer);

        if (sessionSnapshot.originalMixPath.has_value()) {
          try {
            engine::AudioFileIO fileIO;
            engine::AudioResampler resampler;
            auto originalMix = fileIO.readAudioFile(sessionSnapshot.originalMixPath.value());
            if (originalMix.getSampleRate() != rawMixBuffer.getSampleRate()) {
              originalMix = resampler.resampleLinear(originalMix, rawMixBuffer.getSampleRate());
            }

            automaster::OriginalMixReference referenceTarget;
            masterPlan = referenceTarget.applySoftTarget(masterPlan,
                                                         rawMixBuffer,
                                                         originalMix,
                                                         autoMasterStrategy,
                                                         analyzer);
          } catch (const std::exception& error) {
            reportAppend += "\nOriginal mix target skipped: " + juce::String(error.what());
          }
        }

        std::string backendDiagnostics;
        std::unique_ptr<ai::IModelInference> masterInference;
        if (masterPack.has_value()) {
          masterInference = createInferenceBackend(&masterPack.value(), settings.gpuExecutionProvider, &backendDiagnostics);
        }

        ai::AutoMasterStrategyAI aiMaster;
        if (masterInference != nullptr) {
          const auto mixMetrics = analyzer.analyzeBuffer(rawMixBuffer);
          masterPlan = aiMaster.buildPlan(mixMetrics, masterPlan, masterInference.get());
          if (masterPack.has_value()) {
            masterPlan.decisionLog.push_back("AI pack: " + masterPack->id + " license=" + masterPack->licenseId);
          }
          if (!backendDiagnostics.empty()) {
            masterPlan.decisionLog.push_back("Inference backend: " + backendDiagnostics);
          }
        }

        if (masterInference != nullptr) {
          previewMaster = aiMaster.applyPlan(rawMixBuffer, masterPlan, autoMasterStrategy, &previewReport);
        } else {
          previewMaster = autoMasterStrategy.applyPlan(rawMixBuffer, masterPlan, &previewReport);
        }

        reportAppend += "\nMaster decisions:\n" + toJuceText(masterPlan.decisionLog);
      }
    } catch (const std::exception& error) {
      errorText = "Auto Master failed:\n" + juce::String(error.what());
    } catch (...) {
      errorText = "Auto Master failed:\nUnknown error";
    }

    juce::MessageManager::callAsync([safeThis,
                                     masterPlan = std::move(masterPlan),
                                     rawMixBuffer = std::move(rawMixBuffer),
                                     previewMaster = std::move(previewMaster),
                                     previewReport,
                                     reportAppend,
                                     errorText,
                                     cancelled]() mutable {
      if (safeThis == nullptr) {
        return;
      }

      safeThis->taskRunning_.store(false);
      safeThis->cancelButton_.setEnabled(false);

      if (!errorText.isEmpty()) {
        safeThis->statusLabel_.setText("Auto Master failed", juce::dontSendNotification);
        safeThis->reportEditor_.setText(errorText);
        safeThis->appendTaskHistory("Auto Master failed");
        return;
      }

      if (cancelled) {
        safeThis->statusLabel_.setText("Auto Master cancelled", juce::dontSendNotification);
        safeThis->appendTaskHistory("Auto Master cancelled");
        return;
      }

      safeThis->session_.masterPlan = std::move(masterPlan);
      safeThis->previewEngine_.setBuffers(rawMixBuffer, previewMaster);
      safeThis->previewEngine_.setSource(engine::PreviewSource::OriginalMix);
      safeThis->previewEngine_.stop();
      safeThis->updateTransportFromBuffer(safeThis->previewEngine_.buildCrossfadedPreview(1024));
      safeThis->updateMeterPanel(previewReport);

      safeThis->statusLabel_.setText("Auto Master plan generated", juce::dontSendNotification);
      safeThis->appendTaskHistory("Auto Master completed");
      if (!reportAppend.isEmpty()) {
        safeThis->reportEditor_.setText(safeThis->reportEditor_.getText() + reportAppend);
      }
    });
  }).detach();
}

void MainComponent::onBatchImport() {
  if (taskRunning_.load()) {
    statusLabel_.setText("A render task is already running", juce::dontSendNotification);
    return;
  }

  batchImportChooser_ = std::make_unique<juce::FileChooser>("Select folder for batch mastering");
  constexpr int flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories;

  batchImportChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto folder = chooser.getResult();
    if (folder == juce::File()) {
      batchImportChooser_.reset();
      return;
    }

    cancelRender_.store(false);
    cancelButton_.setEnabled(true);
    taskRunning_.store(true);
    statusLabel_.setText("Batch preparing...", juce::dontSendNotification);
    appendTaskHistory("Batch started: " + folder.getFullPathName());

    const std::filesystem::path inputFolder(folder.getFullPathName().toStdString());
    const std::filesystem::path outputFolder = inputFolder / "automix_batch_exports";
    const auto baseRenderSettings = buildCurrentRenderSettings("");
    juce::Component::SafePointer<MainComponent> safeThis(this);

    std::thread([safeThis, inputFolder, outputFolder, baseRenderSettings]() mutable {
      std::vector<domain::BatchItem> items;
      juce::String prepError;
      try {
        std::filesystem::create_directories(outputFolder);
        engine::BatchQueueRunner batchQueueRunner;
        items = batchQueueRunner.buildItemsFromFolder(inputFolder, outputFolder);
      } catch (const std::exception& error) {
        prepError = error.what();
      } catch (...) {
        prepError = "Unknown batch preparation error";
      }

      if (safeThis == nullptr) {
        return;
      }

      if (!prepError.isEmpty() || items.empty()) {
        juce::MessageManager::callAsync([safeThis, prepError]() {
          if (safeThis == nullptr) {
            return;
          }
          safeThis->taskRunning_.store(false);
          safeThis->cancelButton_.setEnabled(false);
          if (!prepError.isEmpty()) {
            safeThis->statusLabel_.setText("Batch preparation failed", juce::dontSendNotification);
            safeThis->reportEditor_.setText("Batch preparation error:\n" + prepError);
            safeThis->appendTaskHistory("Batch preparation failed");
          } else {
            safeThis->statusLabel_.setText("Batch folder has no supported audio files", juce::dontSendNotification);
            safeThis->appendTaskHistory("Batch preparation found no supported files");
          }
        });
        return;
      }

      juce::MessageManager::callAsync([safeThis]() {
        if (safeThis == nullptr) {
          return;
        }
        safeThis->statusLabel_.setText("Batch started", juce::dontSendNotification);
      });

      domain::BatchJob job;
      job.items = std::move(items);
      job.settings.outputFolder = outputFolder;
      job.settings.parallelAnalysis = true;
      const int hardwareThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
      job.settings.analysisThreads = std::max(1, hardwareThreads / 2);
      job.settings.renderParallelism = std::max(1, hardwareThreads / 2);
      job.settings.renderSettings = baseRenderSettings;

      engine::BatchQueueRunner runner;
      std::atomic_bool* cancelPtr = nullptr;
      if (safeThis != nullptr) {
        cancelPtr = &safeThis->cancelRender_;
      }
      std::mutex progressMutex;
      auto lastProgressEmit = std::chrono::steady_clock::time_point {};
      size_t lastItemIndex = std::numeric_limits<size_t>::max();
      double lastProgress = -1.0;
      std::string lastStage;

      const auto result = runner.process(
          job,
          [safeThis, &progressMutex, &lastProgressEmit, &lastItemIndex, &lastProgress, &lastStage](const size_t itemIndex,
                                                                                                     const double progress,
                                                                                                     const std::string& stage) {
            if (safeThis == nullptr) {
              return;
            }
            bool emit = false;
            {
              std::scoped_lock lock(progressMutex);
              const auto now = std::chrono::steady_clock::now();
              const bool itemChanged = itemIndex != lastItemIndex;
              const bool stageChanged = stage != lastStage;
              const bool finalProgress = progress >= 0.999;
              const bool timeGateOpen =
                  lastProgressEmit.time_since_epoch().count() == 0 ||
                  now - lastProgressEmit >= std::chrono::milliseconds(220);
              const bool deltaGateOpen = std::abs(progress - lastProgress) >= 0.03;
              emit = itemChanged || stageChanged || finalProgress || (timeGateOpen && deltaGateOpen);
              if (emit) {
                lastProgressEmit = now;
                lastItemIndex = itemIndex;
                lastProgress = progress;
                lastStage = stage;
              }
            }
            if (!emit) {
              return;
            }

            juce::MessageManager::callAsync([safeThis, itemIndex, progress, stage]() {
              if (safeThis == nullptr) {
                return;
              }
              safeThis->statusLabel_.setText("Batch item " + juce::String(static_cast<int>(itemIndex + 1)) +
                                                 " " + stage + " (" + juce::String(progress * 100.0, 1) + "%)",
                                             juce::dontSendNotification);
              safeThis->appendTaskHistory("Batch item " + juce::String(static_cast<int>(itemIndex + 1)) +
                                          " " + stage + " " + juce::String(progress * 100.0, 1) + "%");
            });
          },
          cancelPtr);

      if (safeThis == nullptr) {
        return;
      }

      juce::String summary;
      summary << "Batch completed\n";
      summary << "Completed: " << result.completed << "\n";
      summary << "Failed: " << result.failed << "\n";
      summary << "Cancelled: " << result.cancelled << "\n";

      for (const auto& item : job.items) {
        summary << item.session.sessionName << " -> " << juce::String(item.outputPath.string()) << " ["
                << juce::String(domain::toString(item.status)) << "]";
        if (!item.error.empty()) {
          summary << " error=" << juce::String(item.error);
        }
        summary << "\n";
      }

      juce::MessageManager::callAsync([safeThis, summary]() {
        if (safeThis == nullptr) {
          return;
        }
        safeThis->taskRunning_.store(false);
        safeThis->cancelButton_.setEnabled(false);
        safeThis->statusLabel_.setText("Batch complete", juce::dontSendNotification);
        safeThis->reportEditor_.setText(summary);
        safeThis->appendTaskHistory("Batch completed");
      });
    }).detach();

    batchImportChooser_.reset();
  });
}

void MainComponent::onExport() {
  if (taskRunning_.load()) {
    statusLabel_.setText("A render task is already running", juce::dontSendNotification);
    return;
  }

  if (session_.stems.empty()) {
    statusLabel_.setText("Import stems first", juce::dontSendNotification);
    return;
  }

  const auto rendererSelection = rendererIdByComboId_.find(rendererBox_.getSelectedId());
  const auto selectedRendererId = rendererSelection != rendererIdByComboId_.end() ? rendererSelection->second : std::string("BuiltIn");
  if (const auto profile = domain::findProjectProfile(projectProfiles_, session_.projectProfileId); profile.has_value()) {
    if (!profile->pinnedRendererIds.empty()) {
      const bool pinned = std::find(profile->pinnedRendererIds.begin(),
                                    profile->pinnedRendererIds.end(),
                                    selectedRendererId) != profile->pinnedRendererIds.end();
      if (!pinned) {
        const bool strictPolicy = toLower(session_.safetyPolicyId) == "strict";
        appendTaskHistory("Renderer " + juce::String(selectedRendererId) + " not pinned for profile " + profile->id);
        if (strictPolicy) {
          statusLabel_.setText("Export blocked by strict safety policy: renderer not pinned for selected profile",
                               juce::dontSendNotification);
          return;
        }
      }
    }
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
    const auto sessionCopy = session_;
    const auto analysisSnapshot = analysisEntries_;
    const bool quickExportMode = toLower(settings.exportSpeedMode) == kExportSpeedModeQuick;

    cancelRender_.store(false);
    taskRunning_.store(true);
    cancelButton_.setEnabled(true);
    statusLabel_.setText("Export started", juce::dontSendNotification);
    appendTaskHistory("Export started: " + selected.getFullPathName());

    juce::Component::SafePointer<MainComponent> safeThis(this);
    std::thread([safeThis, sessionCopy, settings, quickExportMode, analysisSnapshot = std::move(analysisSnapshot)]() mutable {
      renderers::RenderResult renderResult;
      juce::String crashMessage;
      juce::String healthText;
      std::vector<analysis::StemAnalysisEntry> analysisEntriesLocal = std::move(analysisSnapshot);
      bool healthHasCriticalIssues = false;
      size_t healthIssueCount = 0;
      try {
        if (quickExportMode) {
          healthText = "Quick export mode: stem-health preflight skipped for faster turnaround.";
        } else {
          if (analysisEntriesLocal.empty()) {
            if (safeThis != nullptr) {
              juce::MessageManager::callAsync([safeThis]() {
                if (safeThis == nullptr) {
                  return;
                }
                safeThis->statusLabel_.setText("Export: analyzing stems", juce::dontSendNotification);
              });
            }

            analysis::StemAnalyzer analyzer;
            analysisEntriesLocal = analyzer.analyzeSession(sessionCopy);
          }
          const auto healthCacheKey = buildExportHealthCacheKey(sessionCopy, analysisEntriesLocal);
          bool healthCacheHit = false;
          {
            std::scoped_lock lock(exportHealthCacheMutex());
            const auto& cached = exportHealthCache();
            if (cached.has_value() && cached->key == healthCacheKey) {
              healthText = cached->text;
              healthHasCriticalIssues = cached->hasCriticalIssues;
              healthIssueCount = cached->issueCount;
              healthCacheHit = true;
            }
          }

          if (!healthCacheHit) {
            analysis::StemHealthAssistant healthAssistant;
            const auto healthReport = healthAssistant.analyze(sessionCopy, analysisEntriesLocal);
            healthText = juce::String(healthAssistant.toText(healthReport));
            healthHasCriticalIssues = healthReport.hasCriticalIssues;
            healthIssueCount = healthReport.issues.size();

            std::scoped_lock lock(exportHealthCacheMutex());
            exportHealthCache() = ExportHealthCacheEntry{
                .key = healthCacheKey,
                .text = healthText,
                .hasCriticalIssues = healthHasCriticalIssues,
                .issueCount = healthIssueCount,
            };
          }
        }

        auto renderer = renderers::createRenderer(settings.rendererName);
        std::atomic_bool* cancelPtr = nullptr;
        if (safeThis != nullptr) {
          cancelPtr = &safeThis->cancelRender_;
        }

        std::mutex progressMutex;
        auto lastProgressEmit = std::chrono::steady_clock::time_point {};
        double lastProgressFraction = -1.0;
        std::string lastProgressStage;

        renderResult = renderer->render(
            sessionCopy,
            settings,
            [safeThis, &progressMutex, &lastProgressEmit, &lastProgressFraction, &lastProgressStage](const double progress,
                                                                                                      const std::string& stage) {
              if (safeThis == nullptr) {
                return;
              }
              bool emit = false;
              {
                std::scoped_lock lock(progressMutex);
                const auto now = std::chrono::steady_clock::now();
                const bool stageChanged = stage != lastProgressStage;
                const bool finalProgress = progress >= 0.999;
                const bool timeGateOpen =
                    lastProgressEmit.time_since_epoch().count() == 0 ||
                    now - lastProgressEmit >= std::chrono::milliseconds(180);
                const bool deltaGateOpen = std::abs(progress - lastProgressFraction) >= 0.02;
                emit = stageChanged || finalProgress || (timeGateOpen && deltaGateOpen);
                if (emit) {
                  lastProgressEmit = now;
                  lastProgressFraction = progress;
                  lastProgressStage = stage;
                }
              }
              if (!emit) {
                return;
              }

              juce::MessageManager::callAsync([safeThis, progress, stage]() {
                if (safeThis == nullptr) {
                  return;
                }
                if (stage == "Mix render cache hit") {
                  safeThis->statusLabel_.setText("Export: Using cached mix render (fast path)",
                                                 juce::dontSendNotification);
                  safeThis->appendTaskHistory("Export using cached mix render");
                  return;
                }
                safeThis->statusLabel_.setText("Export: " + juce::String(stage) + " (" + juce::String(progress * 100.0, 1) + "%)",
                                               juce::dontSendNotification);
                if (progress >= 0.999 || stage != "Summing stem buses") {
                  safeThis->appendTaskHistory("Export " + juce::String(stage) + " " +
                                              juce::String(progress * 100.0, 1) + "%");
                }
              });
            },
            cancelPtr);
      } catch (const std::exception& error) {
        crashMessage = "Export exception:\n" + juce::String(error.what());
      } catch (...) {
        crashMessage = "Export exception:\nUnknown error";
      }

      const auto exportSpeedMode = settings.exportSpeedMode;
      juce::MessageManager::callAsync([safeThis,
                                       renderResult,
                                       crashMessage,
                                       analysisEntriesLocal = std::move(analysisEntriesLocal),
                                       quickExportMode,
                                       exportSpeedMode,
                                       healthText,
                                       healthHasCriticalIssues,
                                       healthIssueCount]() mutable {
        if (safeThis == nullptr) {
          return;
        }

        safeThis->taskRunning_.store(false);
        safeThis->cancelButton_.setEnabled(false);
        if (!analysisEntriesLocal.empty()) {
          safeThis->analysisEntries_ = std::move(analysisEntriesLocal);
          safeThis->analysisTableModel_.setEntries(&safeThis->analysisEntries_);
          safeThis->analysisTable_.updateContent();
        }

        if (!crashMessage.isEmpty()) {
          safeThis->statusLabel_.setText("Export crashed", juce::dontSendNotification);
          safeThis->reportEditor_.setText(crashMessage);
          safeThis->appendTaskHistory("Export crashed");
          return;
        }

        if (renderResult.cancelled) {
          safeThis->statusLabel_.setText("Export cancelled", juce::dontSendNotification);
          safeThis->appendTaskHistory("Export cancelled");
          return;
        }

        if (quickExportMode) {
          safeThis->appendTaskHistory("Quick export mode active: stem-health preflight skipped");
        } else if (healthIssueCount > 0) {
          safeThis->appendTaskHistory("Stem health check found " + juce::String(static_cast<int>(healthIssueCount)) + " issue(s)");
        } else {
          safeThis->appendTaskHistory("Stem health check passed");
        }

        safeThis->statusLabel_.setText(renderResult.success ? "Export complete" : "Export failed",
                                       juce::dontSendNotification);
        if (healthHasCriticalIssues && renderResult.success) {
          safeThis->statusLabel_.setText("Export complete with critical stem health warnings", juce::dontSendNotification);
        }
        safeThis->appendTaskHistory(renderResult.success ? "Export completed" : "Export failed");
        juce::String report = juce::String("Renderer: ") + juce::String(renderResult.rendererName) +
                              juce::String("\nExport mode: ") + juce::String(exportSpeedMode) +
                              juce::String("\nOutput: ") + juce::String(renderResult.outputAudioPath) +
                              juce::String("\nReport: ") + juce::String(renderResult.reportPath) +
                              juce::String("\n\nLogs:\n") + toJuceText(renderResult.logs);
        if (!healthText.isEmpty()) {
          report += "\n\n";
          report += healthText;
        }
        safeThis->reportEditor_.setText(report);
      });
    }).detach();

    exportChooser_.reset();
  });
}

} // namespace automix::app
