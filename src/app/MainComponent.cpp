#include "app/MainComponent.h"

#include <exception>
#include <filesystem>

#include "automaster/OriginalMixReference.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/OfflineRenderPipeline.h"
#include "renderers/PhaseLimiterRenderer.h"
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
  addAndMakeVisible(autoMixButton_);
  addAndMakeVisible(autoMasterButton_);
  addAndMakeVisible(exportButton_);
  addAndMakeVisible(separatedStemsToggle_);
  addAndMakeVisible(residualBlendLabel_);
  addAndMakeVisible(residualBlendSlider_);
  addAndMakeVisible(rendererBox_);
  addAndMakeVisible(aiModelsLabel_);
  addAndMakeVisible(roleModelBox_);
  addAndMakeVisible(mixModelBox_);
  addAndMakeVisible(masterModelBox_);
  addAndMakeVisible(statusLabel_);
  addAndMakeVisible(analysisTable_);
  addAndMakeVisible(reportEditor_);

  importButton_.addListener(this);
  originalMixButton_.addListener(this);
  autoMixButton_.addListener(this);
  autoMasterButton_.addListener(this);
  exportButton_.addListener(this);
  roleModelBox_.addListener(this);
  mixModelBox_.addListener(this);
  masterModelBox_.addListener(this);
  residualBlendSlider_.addListener(this);

  rendererBox_.addItem("BuiltIn", 1);
  renderers::PhaseLimiterRenderer phaseLimiterProbe;
  const juce::String phaseLimiterLabel =
      phaseLimiterProbe.isAvailable() ? "PhaseLimiter" : "PhaseLimiter (not found in assets)";
  rendererBox_.addItem(phaseLimiterLabel, 2);
  rendererBox_.setSelectedId(1);
  separatedStemsToggle_.setToggleState(false, juce::dontSendNotification);
  residualBlendLabel_.setJustificationType(juce::Justification::centredLeft);
  residualBlendSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  residualBlendSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
  residualBlendSlider_.setRange(0.0, 10.0, 0.1);
  residualBlendSlider_.setValue(0.0, juce::dontSendNotification);
  aiModelsLabel_.setJustificationType(juce::Justification::centredLeft);
  roleModelBox_.setTextWhenNothingSelected("Role: none");
  mixModelBox_.setTextWhenNothingSelected("Mix: none");
  masterModelBox_.setTextWhenNothingSelected("Master: none");
  refreshModelPacks();

  statusLabel_.setText("Ready", juce::dontSendNotification);

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
  autoMixButton_.removeListener(this);
  autoMasterButton_.removeListener(this);
  exportButton_.removeListener(this);
  roleModelBox_.removeListener(this);
  mixModelBox_.removeListener(this);
  masterModelBox_.removeListener(this);
  residualBlendSlider_.removeListener(this);
}

void MainComponent::resized() {
  auto area = getLocalBounds().reduced(8);
  auto top = area.removeFromTop(36);
  auto blendRow = area.removeFromTop(28);
  auto modelRow = area.removeFromTop(30);

  importButton_.setBounds(top.removeFromLeft(110).reduced(2));
  originalMixButton_.setBounds(top.removeFromLeft(120).reduced(2));
  autoMixButton_.setBounds(top.removeFromLeft(110).reduced(2));
  autoMasterButton_.setBounds(top.removeFromLeft(120).reduced(2));
  exportButton_.setBounds(top.removeFromLeft(110).reduced(2));
  separatedStemsToggle_.setBounds(top.removeFromLeft(170).reduced(2));
  rendererBox_.setBounds(top.removeFromLeft(160).reduced(2));

  residualBlendLabel_.setBounds(blendRow.removeFromLeft(140).reduced(2));
  residualBlendSlider_.setBounds(blendRow.removeFromLeft(260).reduced(2));

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

  if (button == &autoMixButton_) {
    onAutoMix();
    return;
  }

  if (button == &autoMasterButton_) {
    onAutoMaster();
    return;
  }

  if (button == &exportButton_) {
    onExport();
  }
}

void MainComponent::comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) {
  if (comboBoxThatHasChanged == &roleModelBox_) {
    modelManager_.setActivePackId("role", roleModelBox_.getText().toStdString());
    return;
  }
  if (comboBoxThatHasChanged == &mixModelBox_) {
    modelManager_.setActivePackId("mix", mixModelBox_.getText().toStdString());
    return;
  }
  if (comboBoxThatHasChanged == &masterModelBox_) {
    modelManager_.setActivePackId("master", masterModelBox_.getText().toStdString());
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

  roleModelBox_.addItem("none", 1);
  mixModelBox_.addItem("none", 1);
  masterModelBox_.addItem("none", 1);

  int itemId = 2;
  for (const auto& pack : packs) {
    const juce::String label = pack.id + " (" + pack.type + ")";
    if (pack.type == "role_classifier") {
      roleModelBox_.addItem(label, itemId++);
    } else if (pack.type == "mix_parameters") {
      mixModelBox_.addItem(label, itemId++);
    } else if (pack.type == "master_parameters") {
      masterModelBox_.addItem(label, itemId++);
    } else {
      roleModelBox_.addItem(label, itemId++);
      mixModelBox_.addItem(label, itemId++);
      masterModelBox_.addItem(label, itemId++);
    }
  }

  roleModelBox_.setSelectedId(1);
  mixModelBox_.setSelectedId(1);
  masterModelBox_.setSelectedId(1);
}

void MainComponent::onImport() {
  importChooser_ = std::make_unique<juce::FileChooser>("Select stem files", juce::File(), "*.wav;*.aiff;*.aif;*.flac");
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
      std::make_unique<juce::FileChooser>("Select original stereo mix", juce::File(), "*.wav;*.aiff;*.aif;*.flac");
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

void MainComponent::onAutoMix() {
  if (session_.stems.empty()) {
    statusLabel_.setText("Import stems first", juce::dontSendNotification);
    return;
  }

  analysisEntries_ = analyzer_.analyzeSession(session_);
  analysisTableModel_.setEntries(&analysisEntries_);
  analysisTable_.updateContent();
  session_.mixPlan = autoMixStrategy_.buildPlan(session_, analysisEntries_, 1.0);

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

  statusLabel_.setText("Auto Master plan generated", juce::dontSendNotification);
  reportEditor_.setText(reportEditor_.getText() + "\nMaster decisions:\n" + toJuceText(session_.masterPlan->decisionLog));
}

void MainComponent::onExport() {
  if (session_.stems.empty()) {
    statusLabel_.setText("Import stems first", juce::dontSendNotification);
    return;
  }

  exportChooser_ = std::make_unique<juce::FileChooser>(
      "Export master", juce::File::getSpecialLocation(juce::File::userDocumentsDirectory), "*.wav");
  constexpr int flags = juce::FileBrowserComponent::saveMode |
                        juce::FileBrowserComponent::canSelectFiles |
                        juce::FileBrowserComponent::warnAboutOverwriting;

  exportChooser_->launchAsync(flags, [this](const juce::FileChooser& chooser) {
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      exportChooser_.reset();
      return;
    }

    domain::RenderSettings settings;
    settings.outputSampleRate = 44100;
    settings.blockSize = 1024;
    settings.outputBitDepth = 24;
    settings.outputPath = selected.getFullPathName().toStdString();
    settings.rendererName = rendererBox_.getSelectedId() == 2 ? "PhaseLimiter" : "BuiltIn";

    renderers::RenderResult renderResult;
    try {
      auto renderer = renderers::createRenderer(settings.rendererName);
      renderResult = renderer->render(session_, settings, {}, &cancelRender_);
    } catch (const std::exception& error) {
      statusLabel_.setText("Export crashed", juce::dontSendNotification);
      reportEditor_.setText(juce::String("Export exception:\n") + juce::String(error.what()));
      exportChooser_.reset();
      return;
    } catch (...) {
      statusLabel_.setText("Export crashed", juce::dontSendNotification);
      reportEditor_.setText("Export exception:\nUnknown error");
      exportChooser_.reset();
      return;
    }

    if (renderResult.cancelled) {
      statusLabel_.setText("Export cancelled", juce::dontSendNotification);
      exportChooser_.reset();
      return;
    }

    statusLabel_.setText(renderResult.success ? "Export complete" : "Export failed", juce::dontSendNotification);
    reportEditor_.setText(juce::String("Renderer: ") + juce::String(renderResult.rendererName) +
                          juce::String("\nOutput: ") + juce::String(renderResult.outputAudioPath) +
                          juce::String("\nReport: ") + juce::String(renderResult.reportPath) +
                          juce::String("\n\nLogs:\n") + toJuceText(renderResult.logs));

    exportChooser_.reset();
  });
}

} // namespace automix::app
