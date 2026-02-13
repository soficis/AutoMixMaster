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
#include "automaster/OriginalMixReference.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/BatchQueueRunner.h"
#include "engine/OfflineRenderPipeline.h"
#include "renderers/RendererFactory.h"

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

std::unique_ptr<ai::IModelInference> createInferenceBackend(const ai::ModelPack* pack) {
  if (pack == nullptr) {
    return nullptr;
  }

  std::unique_ptr<ai::IModelInference> backend;
  const auto engine = pack->engine;
  if (engine.find("onnx") != std::string::npos || engine == "unknown") {
    backend = std::make_unique<ai::OnnxModelInference>();
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
  return backend;
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
  addAndMakeVisible(addExternalRendererButton_);
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
  addAndMakeVisible(aiModelsLabel_);
  addAndMakeVisible(roleModelBox_);
  addAndMakeVisible(mixModelBox_);
  addAndMakeVisible(masterModelBox_);
  addAndMakeVisible(statusLabel_);
  addAndMakeVisible(meterLufsLabel_);
  addAndMakeVisible(meterShortTermLabel_);
  addAndMakeVisible(meterTruePeakLabel_);
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
  addExternalRendererButton_.addListener(this);
  exportButton_.addListener(this);
  cancelButton_.addListener(this);
  exportFormatBox_.addListener(this);
  roleModelBox_.addListener(this);
  mixModelBox_.addListener(this);
  masterModelBox_.addListener(this);
  residualBlendSlider_.addListener(this);

  refreshRenderers();
  separatedStemsToggle_.setToggleState(false, juce::dontSendNotification);
  residualBlendLabel_.setJustificationType(juce::Justification::centredLeft);
  residualBlendSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  residualBlendSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
  residualBlendSlider_.setRange(0.0, 10.0, 0.1);
  residualBlendSlider_.setValue(0.0, juce::dontSendNotification);
  exportFormatLabel_.setJustificationType(juce::Justification::centredLeft);
  exportFormatBox_.addItem("WAV", 1);
  exportFormatBox_.addItem("FLAC", 2);
  exportFormatBox_.addItem("AIFF", 3);
  exportFormatBox_.addItem("OGG (lossy)", 4);
  exportFormatBox_.addItem("MP3 (lossy)", 5);
  exportFormatBox_.setSelectedId(1, juce::dontSendNotification);
  exportBitrateLabel_.setJustificationType(juce::Justification::centredLeft);
  exportBitrateSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  exportBitrateSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
  exportBitrateSlider_.setRange(64.0, 320.0, 1.0);
  exportBitrateSlider_.setValue(192.0, juce::dontSendNotification);
  exportBitrateLabel_.setEnabled(false);
  exportBitrateSlider_.setEnabled(false);
  cancelButton_.setEnabled(false);
  aiModelsLabel_.setJustificationType(juce::Justification::centredLeft);
  roleModelBox_.setTextWhenNothingSelected("Role: none");
  mixModelBox_.setTextWhenNothingSelected("Mix: none");
  masterModelBox_.setTextWhenNothingSelected("Master: none");
  refreshModelPacks();

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

  session_.sessionName = "Untitled Session";
}

MainComponent::~MainComponent() {
  importButton_.removeListener(this);
  originalMixButton_.removeListener(this);
  saveSessionButton_.removeListener(this);
  loadSessionButton_.removeListener(this);
  autoMixButton_.removeListener(this);
  autoMasterButton_.removeListener(this);
  batchImportButton_.removeListener(this);
  previewOriginalButton_.removeListener(this);
  previewRenderedButton_.removeListener(this);
  addExternalRendererButton_.removeListener(this);
  exportButton_.removeListener(this);
  cancelButton_.removeListener(this);
  exportFormatBox_.removeListener(this);
  roleModelBox_.removeListener(this);
  mixModelBox_.removeListener(this);
  masterModelBox_.removeListener(this);
  residualBlendSlider_.removeListener(this);
}

void MainComponent::resized() {
  auto area = getLocalBounds().reduced(8);
  auto top = area.removeFromTop(36);
  auto toolsRow = area.removeFromTop(30);
  auto meterRow = area.removeFromTop(24);
  auto blendRow = area.removeFromTop(28);
  auto exportRow = area.removeFromTop(28);
  auto modelRow = area.removeFromTop(30);

  importButton_.setBounds(top.removeFromLeft(110).reduced(2));
  originalMixButton_.setBounds(top.removeFromLeft(120).reduced(2));
  saveSessionButton_.setBounds(top.removeFromLeft(120).reduced(2));
  loadSessionButton_.setBounds(top.removeFromLeft(120).reduced(2));
  autoMixButton_.setBounds(top.removeFromLeft(110).reduced(2));
  autoMasterButton_.setBounds(top.removeFromLeft(120).reduced(2));
  batchImportButton_.setBounds(top.removeFromLeft(120).reduced(2));
  exportButton_.setBounds(top.removeFromLeft(110).reduced(2));
  cancelButton_.setBounds(top.removeFromLeft(90).reduced(2));
  previewOriginalButton_.setBounds(toolsRow.removeFromLeft(110).reduced(2));
  previewRenderedButton_.setBounds(toolsRow.removeFromLeft(110).reduced(2));
  separatedStemsToggle_.setBounds(toolsRow.removeFromLeft(170).reduced(2));
  rendererBox_.setBounds(toolsRow.removeFromLeft(240).reduced(2));
  addExternalRendererButton_.setBounds(toolsRow.removeFromLeft(180).reduced(2));

  meterLufsLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));
  meterShortTermLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));
  meterTruePeakLabel_.setBounds(meterRow.removeFromLeft(220).reduced(2));

  residualBlendLabel_.setBounds(blendRow.removeFromLeft(140).reduced(2));
  residualBlendSlider_.setBounds(blendRow.removeFromLeft(260).reduced(2));

  exportFormatLabel_.setBounds(exportRow.removeFromLeft(60).reduced(2));
  exportFormatBox_.setBounds(exportRow.removeFromLeft(190).reduced(2));
  exportBitrateLabel_.setBounds(exportRow.removeFromLeft(90).reduced(2));
  exportBitrateSlider_.setBounds(exportRow.removeFromLeft(220).reduced(2));

  aiModelsLabel_.setBounds(modelRow.removeFromLeft(70).reduced(2));
  roleModelBox_.setBounds(modelRow.removeFromLeft(250).reduced(2));
  mixModelBox_.setBounds(modelRow.removeFromLeft(250).reduced(2));
  masterModelBox_.setBounds(modelRow.removeFromLeft(250).reduced(2));

  statusLabel_.setBounds(area.removeFromTop(24));
  analysisTable_.setBounds(area.removeFromTop(200));
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

  if (button == &addExternalRendererButton_) {
    onAddExternalRenderer();
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
  if (comboBoxThatHasChanged == &exportFormatBox_) {
    const int selected = exportFormatBox_.getSelectedId();
    const bool lossy = (selected == 4 || selected == 5);
    exportBitrateSlider_.setEnabled(lossy);
    exportBitrateLabel_.setEnabled(lossy);
  }
}

void MainComponent::sliderValueChanged(juce::Slider* slider) {
  if (slider == &residualBlendSlider_) {
    session_.residualBlend = residualBlendSlider_.getValue();
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

  roleModelBox_.setSelectedId(1);
  mixModelBox_.setSelectedId(1);
  masterModelBox_.setSelectedId(1);
  modelManager_.setActivePackId("role", "none");
  modelManager_.setActivePackId("mix", "none");
  modelManager_.setActivePackId("master", "none");
}

std::vector<renderers::ExternalRendererConfig> MainComponent::loadConfiguredExternalRenderers() const {
  std::vector<renderers::ExternalRendererConfig> configs;
#ifdef ENABLE_EXTERNAL_TOOL_SUPPORT
  const char* rawValue = std::getenv("AUTOMIX_EXTERNAL_RENDERERS");
  if (rawValue == nullptr || *rawValue == '\0') {
    return configs;
  }

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
    ++comboId;
  }

  rendererBox_.setSelectedId(1);
}

domain::RenderSettings MainComponent::buildCurrentRenderSettings(const std::string& outputPath) const {
  domain::RenderSettings settings;
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;
  settings.outputBitDepth = 24;
  settings.processingThreads = 0;
  settings.preferHardwareAcceleration = true;

  switch (exportFormatBox_.getSelectedId()) {
    case 2:
      settings.outputFormat = "flac";
      break;
    case 3:
      settings.outputFormat = "aiff";
      break;
    case 4:
      settings.outputFormat = "ogg";
      break;
    case 5:
      settings.outputFormat = "mp3";
      break;
    default:
      settings.outputFormat = "wav";
      break;
  }

  settings.lossyBitrateKbps = std::clamp(static_cast<int>(std::lround(exportBitrateSlider_.getValue())), 64, 320);
  settings.lossyQuality =
      std::clamp(static_cast<int>(std::lround((static_cast<double>(settings.lossyBitrateKbps) - 64.0) / 25.6)), 0, 10);
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
    session_.stems.clear();
    const auto files = chooser.getResults();

    for (int i = 0; i < files.size(); ++i) {
      const auto& file = files.getReference(i);

      domain::Stem stem;
      stem.id = "stem_" + std::to_string(i + 1);
      stem.name = file.getFileNameWithoutExtension().toStdString();
      stem.filePath = file.getFullPathName().toStdString();
      stem.origin = separatedStemsToggle_.getToggleState() ? domain::StemOrigin::Separated : domain::StemOrigin::Recorded;
      session_.stems.push_back(stem);
    }

    statusLabel_.setText("Imported " + juce::String(static_cast<int>(session_.stems.size())) + " stems",
                         juce::dontSendNotification);
    analysisEntries_.clear();
    analysisTableModel_.setEntries(&analysisEntries_);
    analysisTable_.updateContent();
    reportEditor_.setText(juce::String("Imported files:\n") +
                          toJuceText([&]() {
                            std::vector<std::string> lines;
                            lines.reserve(session_.stems.size());
                            for (const auto& stem : session_.stems) {
                              lines.push_back(stem.name + " -> " + stem.filePath);
                            }
                            return lines;
                          }()));

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
      analysisEntries_.clear();
      analysisTableModel_.setEntries(&analysisEntries_);
      analysisTable_.updateContent();
      statusLabel_.setText("Session loaded", juce::dontSendNotification);
      reportEditor_.setText("Loaded session: " + selected.getFullPathName());
    } catch (const std::exception& error) {
      statusLabel_.setText("Load failed", juce::dontSendNotification);
      reportEditor_.setText("Session load error:\n" + juce::String(error.what()));
    }

    loadSessionChooser_.reset();
  });
}

void MainComponent::onPreviewOriginal() {
  previewEngine_.setSource(engine::PreviewSource::OriginalMix);
  previewEngine_.play();
  const auto preview = previewEngine_.buildCrossfadedPreview(1024);
  statusLabel_.setText("Preview A selected", juce::dontSendNotification);
  reportEditor_.setText(reportEditor_.getText() +
                        "\nPreview source: Original (" + juce::String(preview.getNumSamples()) + " samples)");
}

void MainComponent::onPreviewRendered() {
  previewEngine_.setSource(engine::PreviewSource::RenderedMix);
  previewEngine_.play();
  const auto preview = previewEngine_.buildCrossfadedPreview(1024);
  statusLabel_.setText("Preview B selected", juce::dontSendNotification);
  reportEditor_.setText(reportEditor_.getText() +
                        "\nPreview source: Rendered (" + juce::String(preview.getNumSamples()) + " samples)");
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
    userExternalRendererConfigs_.push_back(config);
    refreshRenderers();

    statusLabel_.setText("External renderer added", juce::dontSendNotification);
    reportEditor_.setText(reportEditor_.getText() +
                          "\nAdded external renderer: " + selected.getFullPathName() +
                          "\nLicense note: user-supplied tool is not distributed by this app.");
    externalRendererChooser_.reset();
  });
}

void MainComponent::updateMeterPanel(const automaster::MasteringReport& report) {
  meterLufsLabel_.setText("LUFS: " + juce::String(report.integratedLufs, 2), juce::dontSendNotification);
  meterShortTermLabel_.setText("Short-term: " + juce::String(report.shortTermLufs, 2), juce::dontSendNotification);
  meterTruePeakLabel_.setText("True Peak: " + juce::String(report.truePeakDbtp, 2) + " dBTP",
                              juce::dontSendNotification);
}

void MainComponent::onAutoMix() {
  if (session_.stems.empty()) {
    statusLabel_.setText("Import stems first", juce::dontSendNotification);
    return;
  }

  analysisEntries_ = analyzer_.analyzeSession(session_);
  analysisTableModel_.setEntries(&analysisEntries_);
  analysisTable_.updateContent();

  const auto heuristicPlan = autoMixStrategy_.buildPlan(session_, analysisEntries_, 1.0);
  session_.mixPlan = heuristicPlan;

  ai::AutoMixStrategyAI aiMix;
  const auto* mixPack = findPackById(modelManager_, modelManager_.activePackId("mix"));
  auto inference = createInferenceBackend(mixPack);
  if (inference != nullptr) {
    session_.mixPlan = aiMix.buildPlan(session_, analysisEntries_, heuristicPlan, inference.get());
    session_.mixPlan->decisionLog.push_back("AI pack: " + mixPack->id + " license=" + mixPack->licenseId);
  }

  statusLabel_.setText("Auto Mix plan generated", juce::dontSendNotification);
  const juce::String analysisJson = analyzer_.toJsonReport(analysisEntries_);
  reportEditor_.setText(juce::String("Analysis report JSON:\n") + analysisJson +
                        juce::String("\n\nMix decisions:\n") + toJuceText(session_.mixPlan->decisionLog));
}

void MainComponent::onAutoMaster() {
  if (session_.stems.empty()) {
    statusLabel_.setText("Import stems first", juce::dontSendNotification);
    return;
  }

  cancelRender_.store(false);

  domain::RenderSettings settings;
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;

  engine::OfflineRenderPipeline pipeline;
  auto rawMix = pipeline.renderRawMix(session_, settings, {}, &cancelRender_);
  if (rawMix.cancelled) {
    statusLabel_.setText("Auto Master cancelled", juce::dontSendNotification);
    return;
  }

  session_.masterPlan = autoMasterStrategy_.buildPlan(domain::MasterPreset::DefaultStreaming, rawMix.mixBuffer);
  if (session_.originalMixPath.has_value()) {
    try {
      engine::AudioFileIO fileIO;
      engine::AudioResampler resampler;
      auto originalMix = fileIO.readAudioFile(session_.originalMixPath.value());
      if (originalMix.getSampleRate() != rawMix.mixBuffer.getSampleRate()) {
        originalMix = resampler.resampleLinear(originalMix, rawMix.mixBuffer.getSampleRate());
      }

      automaster::OriginalMixReference referenceTarget;
      session_.masterPlan = referenceTarget.applySoftTarget(session_.masterPlan.value(),
                                                            rawMix.mixBuffer,
                                                            originalMix,
                                                            autoMasterStrategy_,
                                                            analyzer_);
    } catch (const std::exception& error) {
      reportEditor_.setText(reportEditor_.getText() + "\nOriginal mix target skipped: " + juce::String(error.what()));
    }
  }

  const auto* masterPack = findPackById(modelManager_, modelManager_.activePackId("master"));
  auto masterInference = createInferenceBackend(masterPack);
  ai::AutoMasterStrategyAI aiMaster;
  if (masterInference != nullptr) {
    const auto mixMetrics = analyzer_.analyzeBuffer(rawMix.mixBuffer);
    session_.masterPlan =
        aiMaster.buildPlan(mixMetrics, session_.masterPlan.value(), masterInference.get());
    session_.masterPlan->decisionLog.push_back("AI pack: " + masterPack->id + " license=" + masterPack->licenseId);
  }

  automaster::MasteringReport previewReport;
  auto previewMaster = autoMasterStrategy_.applyPlan(rawMix.mixBuffer, session_.masterPlan.value(), &previewReport);
  if (masterInference != nullptr) {
    previewMaster = aiMaster.applyPlan(rawMix.mixBuffer, session_.masterPlan.value(), autoMasterStrategy_, &previewReport);
  }
  previewEngine_.setBuffers(rawMix.mixBuffer, previewMaster);
  previewEngine_.setSource(engine::PreviewSource::OriginalMix);
  previewEngine_.stop();
  updateMeterPanel(previewReport);

  statusLabel_.setText("Auto Master plan generated", juce::dontSendNotification);
  reportEditor_.setText(reportEditor_.getText() + "\nMaster decisions:\n" + toJuceText(session_.masterPlan->decisionLog));
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
        renderResult = renderer->render(sessionCopy, settings, {}, cancelPtr);
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
