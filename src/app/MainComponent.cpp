#include "app/MainComponent.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <sstream>
#include <thread>

#include "ai/AutoMasterStrategyAI.h"
#include "ai/AutoMixStrategyAI.h"
#include "ai/IModelInference.h"
#include "ai/OnnxModelInference.h"
#include "ai/RtNeuralInference.h"
#include "ai/StemSeparator.h"
#include "automaster/OriginalMixReference.h"
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
  addAndMakeVisible(saveSessionButton_);
  addAndMakeVisible(loadSessionButton_);
  addAndMakeVisible(autoMixButton_);
  addAndMakeVisible(autoMasterButton_);
  addAndMakeVisible(batchImportButton_);
  addAndMakeVisible(previewOriginalButton_);
  addAndMakeVisible(previewRenderedButton_);
  addAndMakeVisible(playPauseButton_);
  addAndMakeVisible(stopButton_);
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
  addAndMakeVisible(exportBitrateLabel_);
  addAndMakeVisible(exportBitrateSlider_);
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

  importButton_.addListener(this);
  originalMixButton_.addListener(this);
  saveSessionButton_.addListener(this);
  loadSessionButton_.addListener(this);
  autoMixButton_.addListener(this);
  autoMasterButton_.addListener(this);
  batchImportButton_.addListener(this);
  previewOriginalButton_.addListener(this);
  previewRenderedButton_.addListener(this);
  playPauseButton_.addListener(this);
  stopButton_.addListener(this);
  addExternalRendererButton_.addListener(this);
  prefetchLameButton_.addListener(this);
  exportButton_.addListener(this);
  cancelButton_.addListener(this);

  rendererBox_.addListener(this);
  exportFormatBox_.addListener(this);
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
  transportSlider_.addListener(this);

  residualBlendLabel_.setJustificationType(juce::Justification::centredLeft);
  residualBlendSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  residualBlendSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
  residualBlendSlider_.setRange(0.0, 10.0, 0.1);
  residualBlendSlider_.setValue(0.0, juce::dontSendNotification);

  exportFormatLabel_.setJustificationType(juce::Justification::centredLeft);

  exportBitrateLabel_.setJustificationType(juce::Justification::centredLeft);
  exportBitrateSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  exportBitrateSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
  exportBitrateSlider_.setRange(64.0, 320.0, 1.0);
  exportBitrateSlider_.setValue(192.0, juce::dontSendNotification);

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

  cancelButton_.setEnabled(false);
  session_.sessionName = "Untitled Session";

  refreshRenderers();
  refreshCodecAvailability();
  refreshModelPacks();
  populateMasterPresetSelectors();
  refreshStemRoutingSelectors();

  transportController_.addChangeListener(this);
  startTimerHz(20);
  updateTransportDisplay();
}

MainComponent::~MainComponent() {
  stopTimer();
  transportController_.removeChangeListener(this);

  importButton_.removeListener(this);
  originalMixButton_.removeListener(this);
  saveSessionButton_.removeListener(this);
  loadSessionButton_.removeListener(this);
  autoMixButton_.removeListener(this);
  autoMasterButton_.removeListener(this);
  batchImportButton_.removeListener(this);
  previewOriginalButton_.removeListener(this);
  previewRenderedButton_.removeListener(this);
  playPauseButton_.removeListener(this);
  stopButton_.removeListener(this);
  addExternalRendererButton_.removeListener(this);
  prefetchLameButton_.removeListener(this);
  exportButton_.removeListener(this);
  cancelButton_.removeListener(this);

  rendererBox_.removeListener(this);
  exportFormatBox_.removeListener(this);
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
  transportSlider_.removeListener(this);
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
  auto transportRow = area.removeFromTop(24);

  importButton_.setBounds(top.removeFromLeft(98).reduced(2));
  originalMixButton_.setBounds(top.removeFromLeft(112).reduced(2));
  saveSessionButton_.setBounds(top.removeFromLeft(112).reduced(2));
  loadSessionButton_.setBounds(top.removeFromLeft(112).reduced(2));
  autoMixButton_.setBounds(top.removeFromLeft(95).reduced(2));
  autoMasterButton_.setBounds(top.removeFromLeft(110).reduced(2));
  batchImportButton_.setBounds(top.removeFromLeft(112).reduced(2));
  exportButton_.setBounds(top.removeFromLeft(90).reduced(2));
  cancelButton_.setBounds(top.removeFromLeft(85).reduced(2));

  previewOriginalButton_.setBounds(toolsRow.removeFromLeft(100).reduced(2));
  previewRenderedButton_.setBounds(toolsRow.removeFromLeft(100).reduced(2));
  playPauseButton_.setBounds(toolsRow.removeFromLeft(96).reduced(2));
  stopButton_.setBounds(toolsRow.removeFromLeft(70).reduced(2));
  separatedStemsToggle_.setBounds(toolsRow.removeFromLeft(160).reduced(2));
  rendererBox_.setBounds(toolsRow.removeFromLeft(220).reduced(2));
  addExternalRendererButton_.setBounds(toolsRow.removeFromLeft(170).reduced(2));
  prefetchLameButton_.setBounds(toolsRow.removeFromLeft(130).reduced(2));

  meterLufsLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));
  meterShortTermLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));
  meterTruePeakLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));

  residualBlendLabel_.setBounds(blendRow.removeFromLeft(132).reduced(2));
  residualBlendSlider_.setBounds(blendRow.removeFromLeft(250).reduced(2));

  exportFormatLabel_.setBounds(exportRow.removeFromLeft(52).reduced(2));
  exportFormatBox_.setBounds(exportRow.removeFromLeft(190).reduced(2));
  exportBitrateLabel_.setBounds(exportRow.removeFromLeft(90).reduced(2));
  exportBitrateSlider_.setBounds(exportRow.removeFromLeft(220).reduced(2));
  gpuProviderLabel_.setBounds(exportRow.removeFromLeft(84).reduced(2));
  gpuProviderBox_.setBounds(exportRow.removeFromLeft(150).reduced(2));

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
  transportSlider_.setBounds(transportRow.reduced(2));

  statusLabel_.setBounds(area.removeFromTop(24).reduced(2));
  analysisTable_.setBounds(area.removeFromTop(180));
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
      transportController_.play();
      previewEngine_.play();
    }
    updateTransportDisplay();
    return;
  }

  if (button == &stopButton_) {
    transportController_.stop();
    previewEngine_.stop();
    updateTransportDisplay();
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

    const bool lossy = util::WavWriter::isLossyFormat(format);
    exportBitrateSlider_.setEnabled(lossy);
    exportBitrateLabel_.setEnabled(lossy);

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

  if (slider == &transportSlider_ && !ignoreTransportSliderChange_) {
    transportController_.seekToFraction(transportSlider_.getValue());
    return;
  }
}

void MainComponent::timerCallback() {
  if (transportController_.isPlaying()) {
    const auto totalSamples = transportController_.totalSamples();
    const auto totalSeconds = transportController_.totalSeconds();
    if (totalSamples > 0 && totalSeconds > 0.0) {
      const auto sampleRate = static_cast<double>(totalSamples) / totalSeconds;
      const int increment = std::max(1, static_cast<int>(std::lround(sampleRate / 20.0)));
      transportController_.advance(increment);
    }
  }

  updateTransportDisplay();
}

void MainComponent::changeListenerCallback(juce::ChangeBroadcaster* source) {
  if (source == &transportController_) {
    updateTransportDisplay();
  }
}

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
  exportFormatBox_.setTooltip(toJuceText(tooltipLines));

  const auto formatIt = codecFormatByComboId_.find(exportFormatBox_.getSelectedId());
  const std::string selectedFormat = formatIt != codecFormatByComboId_.end() ? formatIt->second : "wav";
  const bool lossy = util::WavWriter::isLossyFormat(selectedFormat);
  exportBitrateSlider_.setEnabled(lossy);
  exportBitrateLabel_.setEnabled(lossy);
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
  if (session_.stems.empty()) {
    waveformPreview_.setBuffer(engine::AudioBuffer{});
    transportController_.setTimeline(0, 44100.0);
    return;
  }

  try {
    auto previewSession = session_;

    const auto soloIt = stemIdBySoloComboId_.find(soloStemBox_.getSelectedId());
    const auto muteIt = stemIdByMuteComboId_.find(muteStemBox_.getSelectedId());
    const auto soloStemId = soloIt != stemIdBySoloComboId_.end() ? soloIt->second : std::string();
    const auto muteStemId = muteIt != stemIdByMuteComboId_.end() ? muteIt->second : std::string();

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

    const auto previousProgress = transportController_.progress();
    const auto settings = buildCurrentRenderSettings("");

    engine::OfflineRenderPipeline pipeline;
    const auto raw = pipeline.renderRawMix(previewSession, settings, {}, nullptr);
    if (raw.cancelled || raw.mixBuffer.getNumSamples() == 0) {
      return;
    }

    auto mastered = raw.mixBuffer;
    if (session_.masterPlan.has_value()) {
      mastered = autoMasterStrategy_.applyPlan(raw.mixBuffer, session_.masterPlan.value(), nullptr);
    }

    previewEngine_.setBuffers(raw.mixBuffer, mastered);
    const auto preview = previewEngine_.buildCrossfadedPreview(1024);
    updateTransportFromBuffer(preview);
    transportController_.seekToFraction(previousProgress);
  } catch (const std::exception& error) {
    reportEditor_.setText(reportEditor_.getText() + "\nPreview rebuild skipped: " + juce::String(error.what()));
  }
}

void MainComponent::rebuildPreviewBuffersAsync() {
  if (session_.stems.empty()) {
    waveformPreview_.setBuffer(engine::AudioBuffer{});
    transportController_.setTimeline(0, 44100.0);
    return;
  }

  auto previewSession = session_;
  const auto soloIt = stemIdBySoloComboId_.find(soloStemBox_.getSelectedId());
  const auto muteIt = stemIdByMuteComboId_.find(muteStemBox_.getSelectedId());
  const auto soloStemId = soloIt != stemIdBySoloComboId_.end() ? soloIt->second : std::string();
  const auto muteStemId = muteIt != stemIdByMuteComboId_.end() ? muteIt->second : std::string();
  const auto previousProgress = transportController_.progress();

  juce::Component::SafePointer<MainComponent> safeThis(this);
  std::thread([safeThis,
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

      engine::AudioPreviewEngine previewEngine;
      previewEngine.setBuffers(raw.mixBuffer, mastered);
      const auto preview = previewEngine.buildCrossfadedPreview(1024);

      juce::MessageManager::callAsync(
          [safeThis, rawMix = raw.mixBuffer, mastered = std::move(mastered), preview = std::move(preview), previousProgress]() mutable {
            if (safeThis == nullptr) {
              return;
            }

            safeThis->previewEngine_.setBuffers(rawMix, mastered);
            safeThis->updateTransportFromBuffer(preview);
            safeThis->transportController_.seekToFraction(previousProgress);
          });
    } catch (const std::exception& error) {
      const auto message = juce::String(error.what());
      juce::MessageManager::callAsync([safeThis, message]() {
        if (safeThis == nullptr) {
          return;
        }
        safeThis->reportEditor_.setText(safeThis->reportEditor_.getText() + "\nPreview rebuild skipped: " + message);
      });
    }
  }).detach();
}

void MainComponent::updateTransportFromBuffer(const engine::AudioBuffer& buffer) {
  waveformPreview_.setBuffer(buffer);
  waveformPreview_.setPlayheadProgress(0.0);
  transportController_.setTimeline(buffer.getNumSamples(), buffer.getSampleRate());
  transportController_.stop();

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

  if (transportController_.state() == engine::TransportController::State::Playing) {
    playPauseButton_.setButtonText("Pause");
  } else {
    playPauseButton_.setButtonText("Play");
  }

  const auto positionText = formatDuration(transportController_.positionSeconds());
  const auto totalText = formatDuration(transportController_.totalSeconds());
  transportSlider_.setTooltip(positionText + " / " + totalText);
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
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;
  settings.outputBitDepth = 24;
  settings.processingThreads = 0;
  settings.preferHardwareAcceleration = true;

  const auto formatIt = codecFormatByComboId_.find(exportFormatBox_.getSelectedId());
  settings.outputFormat = formatIt != codecFormatByComboId_.end() ? formatIt->second : "wav";
  if (!util::WavWriter::isFormatAvailable(settings.outputFormat)) {
    settings.outputFormat = "wav";
  }

  settings.lossyBitrateKbps = std::clamp(static_cast<int>(std::lround(exportBitrateSlider_.getValue())), 64, 320);
  settings.lossyQuality =
      std::clamp(static_cast<int>(std::lround((static_cast<double>(settings.lossyBitrateKbps) - 64.0) / 25.6)), 0, 10);

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
}

void MainComponent::onImport() {
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

    session_.stems.clear();
    std::vector<std::string> importLines;

    if (files.size() == 1 && separatedStemsToggle_.getToggleState()) {
      try {
        const auto selected = files.getReference(0);
        const auto mixPath = std::filesystem::path(selected.getFullPathName().toStdString());
        const auto outputDir = mixPath.parent_path() / (mixPath.stem().string() + "_separated");

        ai::StemSeparator separator;
        const auto separationResult = separator.separate(mixPath, outputDir);
        if (separationResult.success) {
          session_.stems = separationResult.stems;
          importLines.push_back("Separated import from: " + mixPath.string());
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
          importLines.push_back(separationResult.logMessage);
        } else {
          importLines.push_back("Separation failed, importing original mix file as stem.");
          importLines.push_back(separationResult.logMessage);
        }
      } catch (const std::exception& error) {
        importLines.push_back("Separation error: " + std::string(error.what()));
      }
    }

    if (session_.stems.empty()) {
      for (int i = 0; i < files.size(); ++i) {
        const auto& file = files.getReference(i);

        domain::Stem stem;
        stem.id = "stem_" + std::to_string(i + 1);
        stem.name = file.getFileNameWithoutExtension().toStdString();
        stem.filePath = file.getFullPathName().toStdString();
        stem.origin = separatedStemsToggle_.getToggleState() ? domain::StemOrigin::Separated : domain::StemOrigin::Recorded;
        stem.enabled = true;
        session_.stems.push_back(stem);

        importLines.push_back(stem.name + " -> " + stem.filePath);
      }
    }

    statusLabel_.setText("Imported " + juce::String(static_cast<int>(session_.stems.size())) + " stems",
                         juce::dontSendNotification);

    analysisEntries_.clear();
    analysisTableModel_.setEntries(&analysisEntries_);
    analysisTable_.updateContent();

    refreshStemRoutingSelectors();
    reportEditor_.setText(juce::String("Imported files:\n") + toJuceText(importLines));
    rebuildPreviewBuffers();

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
    statusLabel_.setText("Original mix loaded", juce::dontSendNotification);
    reportEditor_.setText(reportEditor_.getText() + "\nOriginal mix: " + selected.getFullPathName());
    originalMixChooser_.reset();
  });
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
      sessionRepository_.save(selected.getFullPathName().toStdString(), session_);
      statusLabel_.setText("Session saved", juce::dontSendNotification);
    } catch (const std::exception& error) {
      statusLabel_.setText("Save failed", juce::dontSendNotification);
      reportEditor_.setText("Session save error:\n" + juce::String(error.what()));
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

    try {
      session_ = sessionRepository_.load(selected.getFullPathName().toStdString());
      residualBlendSlider_.setValue(session_.residualBlend, juce::dontSendNotification);
      exportBitrateSlider_.setValue(session_.renderSettings.lossyBitrateKbps, juce::dontSendNotification);

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

      refreshRenderers();
      refreshCodecAvailability();
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

      analysisEntries_.clear();
      analysisTableModel_.setEntries(&analysisEntries_);
      analysisTable_.updateContent();

      statusLabel_.setText("Session loaded", juce::dontSendNotification);
      reportEditor_.setText("Loaded session: " + selected.getFullPathName());
      rebuildPreviewBuffers();
    } catch (const std::exception& error) {
      statusLabel_.setText("Load failed", juce::dontSendNotification);
      reportEditor_.setText("Session load error:\n" + juce::String(error.what()));
    }

    loadSessionChooser_.reset();
  });
}

void MainComponent::onPreviewOriginal() {
  if (transportController_.totalSamples() == 0) {
    rebuildPreviewBuffers();
  }

  const auto progress = transportController_.progress();
  previewEngine_.setSource(engine::PreviewSource::OriginalMix);
  const auto preview = previewEngine_.buildCrossfadedPreview(1024);
  updateTransportFromBuffer(preview);
  transportController_.seekToFraction(progress);
  transportController_.play();
  previewEngine_.play();

  statusLabel_.setText("Preview A selected", juce::dontSendNotification);
}

void MainComponent::onPreviewRendered() {
  if (transportController_.totalSamples() == 0) {
    rebuildPreviewBuffers();
  }

  const auto progress = transportController_.progress();
  previewEngine_.setSource(engine::PreviewSource::RenderedMix);
  const auto preview = previewEngine_.buildCrossfadedPreview(1024);
  updateTransportFromBuffer(preview);
  transportController_.seekToFraction(progress);
  transportController_.play();
  previewEngine_.play();

  statusLabel_.setText("Preview B selected", juce::dontSendNotification);
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

    renderers::ExternalRendererConfig config;
    config.id = "ExternalUserUI" + std::to_string(userExternalRendererConfigs_.size() + 1);
    config.name = selected.getFileName().toStdString();
    config.binaryPath = selected.getFullPathName().toStdString();
    config.licenseId = "User-supplied";
    const auto validation = renderers::ExternalLimiterRenderer::validateBinary(config.binaryPath);
    userExternalRendererConfigs_.push_back(config);

    refreshRenderers();

    const juce::String statusText = validation.valid ? "External renderer added" : "External renderer added (validation failed)";
    statusLabel_.setText(statusText, juce::dontSendNotification);
    reportEditor_.setText(reportEditor_.getText() +
                          "\nAdded external renderer: " + selected.getFullPathName() +
                          "\nValidation: " + juce::String(validation.valid ? "passed" : "failed") +
                          " (" + juce::String(validation.diagnostics) + ")" +
                          "\nLicense note: user-supplied tool is not distributed by this app.");
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
      } else {
        safeThis->statusLabel_.setText("LAME prefetch failed", juce::dontSendNotification);
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
            return;
          }

          if (cancelled || safeThis->cancelRender_.load()) {
            safeThis->statusLabel_.setText("Auto Mix cancelled", juce::dontSendNotification);
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

      const auto rawMix = pipeline.renderRawMix(
          sessionSnapshot,
          settings,
          [safeThis](const engine::RenderProgress& progress) {
            if (safeThis == nullptr) {
              return;
            }
            juce::MessageManager::callAsync([safeThis, progress]() {
              if (safeThis == nullptr) {
                return;
              }
              safeThis->statusLabel_.setText("Auto Master: " + juce::String(progress.stage) + " " +
                                                 juce::String(progress.fraction * 100.0, 1) + "%",
                                             juce::dontSendNotification);
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

        previewMaster = autoMasterStrategy.applyPlan(rawMixBuffer, masterPlan, &previewReport);
        if (masterInference != nullptr) {
          previewMaster = aiMaster.applyPlan(rawMixBuffer, masterPlan, autoMasterStrategy, &previewReport);
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
        return;
      }

      if (cancelled) {
        safeThis->statusLabel_.setText("Auto Master cancelled", juce::dontSendNotification);
        return;
      }

      safeThis->session_.masterPlan = std::move(masterPlan);
      safeThis->previewEngine_.setBuffers(rawMixBuffer, previewMaster);
      safeThis->previewEngine_.setSource(engine::PreviewSource::OriginalMix);
      safeThis->previewEngine_.stop();
      safeThis->updateTransportFromBuffer(safeThis->previewEngine_.buildCrossfadedPreview(1024));
      safeThis->updateMeterPanel(previewReport);

      safeThis->statusLabel_.setText("Auto Master plan generated", juce::dontSendNotification);
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

    const std::filesystem::path inputFolder(folder.getFullPathName().toStdString());
    const std::filesystem::path outputFolder = inputFolder / "automix_batch_exports";
    std::filesystem::create_directories(outputFolder);
    const auto baseRenderSettings = buildCurrentRenderSettings("");

    cancelRender_.store(false);
    cancelButton_.setEnabled(true);
    taskRunning_.store(true);

    engine::BatchQueueRunner batchQueueRunner;
    auto items = batchQueueRunner.buildItemsFromFolder(inputFolder, outputFolder);
    if (items.empty()) {
      statusLabel_.setText("Batch folder has no supported audio files", juce::dontSendNotification);
      taskRunning_.store(false);
      cancelButton_.setEnabled(false);
      batchImportChooser_.reset();
      return;
    }

    statusLabel_.setText("Batch started", juce::dontSendNotification);
    juce::Component::SafePointer<MainComponent> safeThis(this);

    std::thread([safeThis, items = std::move(items), outputFolder, baseRenderSettings]() mutable {
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

      const auto result = runner.process(
          job,
          [safeThis](const size_t itemIndex, const double progress, const std::string& stage) {
            if (safeThis == nullptr) {
              return;
            }
            juce::MessageManager::callAsync([safeThis, itemIndex, progress, stage]() {
              if (safeThis == nullptr) {
                return;
              }
              safeThis->statusLabel_.setText("Batch item " + juce::String(static_cast<int>(itemIndex + 1)) +
                                                 " " + stage + " (" + juce::String(progress * 100.0, 1) + "%)",
                                             juce::dontSendNotification);
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

    cancelRender_.store(false);
    taskRunning_.store(true);
    cancelButton_.setEnabled(true);
    statusLabel_.setText("Export started", juce::dontSendNotification);

    juce::Component::SafePointer<MainComponent> safeThis(this);
    std::thread([safeThis, sessionCopy, settings]() mutable {
      renderers::RenderResult renderResult;
      juce::String crashMessage;
      try {
        auto renderer = renderers::createRenderer(settings.rendererName);
        std::atomic_bool* cancelPtr = nullptr;
        if (safeThis != nullptr) {
          cancelPtr = &safeThis->cancelRender_;
        }
        renderResult = renderer->render(
            sessionCopy,
            settings,
            [safeThis](const double progress, const std::string& stage) {
              if (safeThis == nullptr) {
                return;
              }
              juce::MessageManager::callAsync([safeThis, progress, stage]() {
                if (safeThis == nullptr) {
                  return;
                }
                safeThis->statusLabel_.setText("Export: " + juce::String(stage) + " (" + juce::String(progress * 100.0, 1) + "%)",
                                               juce::dontSendNotification);
              });
            },
            cancelPtr);
      } catch (const std::exception& error) {
        crashMessage = "Export exception:\n" + juce::String(error.what());
      } catch (...) {
        crashMessage = "Export exception:\nUnknown error";
      }

      juce::MessageManager::callAsync([safeThis, renderResult, crashMessage]() {
        if (safeThis == nullptr) {
          return;
        }

        safeThis->taskRunning_.store(false);
        safeThis->cancelButton_.setEnabled(false);

        if (!crashMessage.isEmpty()) {
          safeThis->statusLabel_.setText("Export crashed", juce::dontSendNotification);
          safeThis->reportEditor_.setText(crashMessage);
          return;
        }

        if (renderResult.cancelled) {
          safeThis->statusLabel_.setText("Export cancelled", juce::dontSendNotification);
          return;
        }

        safeThis->statusLabel_.setText(renderResult.success ? "Export complete" : "Export failed",
                                       juce::dontSendNotification);
        safeThis->reportEditor_.setText(juce::String("Renderer: ") + juce::String(renderResult.rendererName) +
                                        juce::String("\nOutput: ") + juce::String(renderResult.outputAudioPath) +
                                        juce::String("\nReport: ") + juce::String(renderResult.reportPath) +
                                        juce::String("\n\nLogs:\n") + toJuceText(renderResult.logs));
      });
    }).detach();

    exportChooser_.reset();
  });
}

} // namespace automix::app
