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
#include "renderers/ExternalLimiterRenderer.h"
#include "util/LameDownloader.h"
#include "util/StringUtils.h"
#include "util/WavWriter.h"

namespace automix::app {
namespace {

using ::automix::util::toLower;
using ::automix::util::extensionForFormat;

class BackgroundJob final : public juce::ThreadPoolJob {
 public:
  explicit BackgroundJob(std::function<void()> task)
      : juce::ThreadPoolJob("BackgroundJob"), task_(std::move(task)) {}

  JobStatus runJob() override {
    task_();
    return jobHasFinished;
  }

 private:
  std::function<void()> task_;
};

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

juce::String modelLabel(const ai::HubModelInfo& model) {
  juce::String label = model.repoId + " [" + model.useCase + "]";
  label += " dls=" + juce::String(model.downloads);
  if (model.recommended) {
    label += " *";
  }
  return label;
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
  refreshCodecAvailability();
  refreshModelPacks();
  populateMasterPresetSelectors();
  refreshProjectProfiles();
  refreshStemRoutingSelectors();

  {
    ModelController::Callbacks modelCallbacks;
    modelCallbacks.onStatus = [this](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), msg]() {
        if (safeThis != nullptr) {
          safeThis->statusLabel_.setText(juce::String(msg), juce::dontSendNotification);
        }
      });
    };
    modelCallbacks.onTaskHistory = [this](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), msg]() {
        if (safeThis != nullptr) {
          safeThis->appendTaskHistory(juce::String(msg));
        }
      });
    };
    modelCallbacks.onReport = [this](const std::string& text) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), text]() {
        if (safeThis != nullptr) {
          safeThis->reportEditor_.setText(juce::String(text));
        }
      });
    };
    modelCallbacks.onModelPacksChanged = [this]() {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this)]() {
        if (safeThis != nullptr) {
          safeThis->refreshModelPacks();
        }
      });
    };
    modelCallbacks.onCatalogReady = [this]() {
      const auto& models = modelController_->discoveredModels();
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

      juce::Component::SafePointer<MainComponent> safeThis(this);
      modelMenu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&modelsMenuButton_),
                              [safeThis](const int selection) {
                                if (safeThis == nullptr) {
                                  return;
                                }
                                if (selection == 1900) {
                                  safeThis->modelController_->fetchCatalog();
                                  return;
                                }
                                const auto& discovered = safeThis->modelController_->discoveredModels();
                                if (selection < 1000 ||
                                    selection >= 1000 + static_cast<int>(discovered.size())) {
                                  return;
                                }
                                const auto& model = discovered[static_cast<size_t>(selection - 1000)];
                                safeThis->modelController_->installModel(model.repoId);
                              });
    };
    modelController_ = std::make_unique<ModelController>(modelManager_, backgroundPool_, std::move(modelCallbacks));
  }

  {
    ImportController::Callbacks importCallbacks;
    importCallbacks.onStatus = [this](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), msg]() {
        if (safeThis != nullptr) {
          safeThis->statusLabel_.setText(juce::String(msg), juce::dontSendNotification);
        }
      });
    };
    importCallbacks.onTaskHistory = [this](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), msg]() {
        if (safeThis != nullptr) {
          safeThis->appendTaskHistory(juce::String(msg));
        }
      });
    };
    importCallbacks.onImportComplete = [this](ImportResult result) {
      session_.stems = std::move(result.stems);
      statusLabel_.setText(
          "Imported " + juce::String(static_cast<int>(session_.stems.size())) + " stems",
          juce::dontSendNotification);
      appendTaskHistory("Imported " + juce::String(static_cast<int>(session_.stems.size())) + " stems");

      analysisEntries_.clear();
      analysisTableModel_.setEntries(&analysisEntries_);
      analysisTable_.updateContent();

      refreshStemRoutingSelectors();
      reportEditor_.setText(juce::String("Imported files:\n") + toJuceText(result.logLines));
      rebuildPreviewBuffersAsync();
    };
    importController_ = std::make_unique<ImportController>(backgroundPool_, std::move(importCallbacks));
  }

  {
    ExportController::Callbacks exportCallbacks;
    exportCallbacks.onStatus = [this](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), msg]() {
        if (safeThis != nullptr) {
          safeThis->statusLabel_.setText(juce::String(msg), juce::dontSendNotification);
        }
      });
    };
    exportCallbacks.onTaskHistory = [this](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), msg]() {
        if (safeThis != nullptr) {
          safeThis->appendTaskHistory(juce::String(msg));
        }
      });
    };
    exportCallbacks.onExportComplete = [this](ExportResult result) {
      taskRunning_.store(false);
      cancelButton_.setEnabled(false);

      if (!result.analysisEntries.empty()) {
        analysisEntries_ = std::move(result.analysisEntries);
        analysisTableModel_.setEntries(&analysisEntries_);
        analysisTable_.updateContent();
      }

      if (!result.crashMessage.isEmpty()) {
        statusLabel_.setText("Export crashed", juce::dontSendNotification);
        reportEditor_.setText(result.crashMessage);
        appendTaskHistory("Export crashed");
        return;
      }

      if (result.cancelled) {
        statusLabel_.setText("Export cancelled", juce::dontSendNotification);
        appendTaskHistory("Export cancelled");
        return;
      }

      const bool quickExportMode = toLower(result.exportSpeedMode) == "quick";
      if (quickExportMode) {
        appendTaskHistory("Quick export mode active: stem-health preflight skipped");
      } else if (result.healthIssueCount > 0) {
        appendTaskHistory("Stem health check found " + juce::String(static_cast<int>(result.healthIssueCount)) + " issue(s)");
      } else {
        appendTaskHistory("Stem health check passed");
      }

      statusLabel_.setText(result.success ? "Export complete" : "Export failed",
                           juce::dontSendNotification);
      if (result.healthHasCriticalIssues && result.success) {
        statusLabel_.setText("Export complete with critical stem health warnings", juce::dontSendNotification);
      }
      appendTaskHistory(result.success ? "Export completed" : "Export failed");
      juce::String report = juce::String("Renderer: ") + juce::String(result.rendererName) +
                            juce::String("\nExport mode: ") + juce::String(result.exportSpeedMode) +
                            juce::String("\nOutput: ") + juce::String(result.outputAudioPath) +
                            juce::String("\nReport: ") + juce::String(result.reportPath) +
                            juce::String("\n\nLogs:\n") + toJuceText(result.logs);
      if (!result.healthText.isEmpty()) {
        report += "\n\n";
        report += result.healthText;
      }
      reportEditor_.setText(report);
    };
    exportController_ = std::make_unique<ExportController>(backgroundPool_, std::move(exportCallbacks));
  }

  {
    ProcessingController::Callbacks processingCallbacks;
    processingCallbacks.onStatus = [this](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), msg]() {
        if (safeThis != nullptr) {
          safeThis->statusLabel_.setText(juce::String(msg), juce::dontSendNotification);
        }
      });
    };
    processingCallbacks.onTaskHistory = [this](const std::string& msg) {
      juce::MessageManager::callAsync([safeThis = juce::Component::SafePointer<MainComponent>(this), msg]() {
        if (safeThis != nullptr) {
          safeThis->appendTaskHistory(juce::String(msg));
        }
      });
    };
    processingCallbacks.onAutoMixComplete = [this](AutoMixResult result) {
      taskRunning_.store(false);
      cancelButton_.setEnabled(false);

      if (!result.errorText.isEmpty()) {
        statusLabel_.setText("Auto Mix failed", juce::dontSendNotification);
        reportEditor_.setText(result.errorText);
        appendTaskHistory("Auto Mix failed");
        return;
      }

      if (result.cancelled) {
        statusLabel_.setText("Auto Mix cancelled", juce::dontSendNotification);
        appendTaskHistory("Auto Mix cancelled");
        return;
      }

      analysisEntries_ = std::move(result.analysisEntries);
      analysisTableModel_.setEntries(&analysisEntries_);
      analysisTable_.updateContent();

      if (result.mixPlan.has_value()) {
        session_.mixPlan = result.mixPlan.value();
      }

      if (!result.reportText.isEmpty()) {
        reportEditor_.setText(result.reportText);
      }

      statusLabel_.setText("Auto Mix plan generated", juce::dontSendNotification);
      appendTaskHistory("Auto Mix completed");
      rebuildPreviewBuffersAsync();
    };
    processingCallbacks.onAutoMasterComplete = [this](AutoMasterResult result) {
      taskRunning_.store(false);
      cancelButton_.setEnabled(false);

      if (!result.errorText.isEmpty()) {
        statusLabel_.setText("Auto Master failed", juce::dontSendNotification);
        reportEditor_.setText(result.errorText);
        appendTaskHistory("Auto Master failed");
        return;
      }

      if (result.cancelled) {
        statusLabel_.setText("Auto Master cancelled", juce::dontSendNotification);
        appendTaskHistory("Auto Master cancelled");
        return;
      }

      session_.masterPlan = std::move(result.masterPlan);
      previewEngine_.setBuffers(result.rawMixBuffer, result.previewMaster);
      previewEngine_.setSource(engine::PreviewSource::OriginalMix);
      previewEngine_.stop();
      updateTransportFromBuffer(previewEngine_.buildCrossfadedPreview(1024));
      updateMeterPanel(result.previewReport);

      statusLabel_.setText("Auto Master plan generated", juce::dontSendNotification);
      appendTaskHistory("Auto Master completed");
      if (!result.reportAppend.isEmpty()) {
        reportEditor_.setText(reportEditor_.getText() + result.reportAppend);
      }
    };
    processingCallbacks.onBatchComplete = [this](BatchResult result) {
      taskRunning_.store(false);
      cancelButton_.setEnabled(false);

      if (!result.errorText.isEmpty()) {
        if (result.summary.isEmpty()) {
          statusLabel_.setText("Batch preparation failed", juce::dontSendNotification);
          reportEditor_.setText(result.errorText);
          appendTaskHistory("Batch preparation failed");
        } else {
          statusLabel_.setText("Batch folder has no supported audio files", juce::dontSendNotification);
          appendTaskHistory("Batch preparation found no supported files");
        }
        return;
      }

      statusLabel_.setText("Batch complete", juce::dontSendNotification);
      reportEditor_.setText(result.summary);
      appendTaskHistory("Batch completed");
    };
    processingController_ = std::make_unique<ProcessingController>(backgroundPool_, std::move(processingCallbacks));
  }

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
  cancelRender_.store(true);
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
  session_.renderSettings.metadataPolicy = profile.metadataPolicy;
  session_.renderSettings.metadataTemplate = profile.metadataTemplate;
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
  backgroundPool_.addJob(new BackgroundJob([safeThis,
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
  }), true);
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
  settings.metadataPolicy = session_.renderSettings.metadataPolicy.empty() ? "copy_all" : session_.renderSettings.metadataPolicy;
  settings.metadataTemplate = session_.renderSettings.metadataTemplate;

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

    importController_->importFiles(std::move(selectedFiles),
                                   separatedStemsToggle_.getToggleState(),
                                   session_.preferredStemCount);
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
  ExportController::clearHealthCache();

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
    backgroundPool_.addJob(new BackgroundJob([safeThis, selectedPath]() {
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
    }), true);

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
    backgroundPool_.addJob(new BackgroundJob([safeThis, selectedPath, selectedName]() {
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
    }), true);

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
  backgroundPool_.addJob(new BackgroundJob([safeThis]() {
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
  }), true);
}

void MainComponent::onModelsMenu() {
  juce::PopupMenu menu;
  menu.addItem(1, "Browse & Download Models");
  menu.addItem(2, "Installed Models");
  menu.addItem(3, "Check Updates");
  menu.addItem(4, "Integrity & Licenses");
  menu.addSeparator();
  menu.addItem(5, "Open Model Hub Folder");

  juce::Component::SafePointer<MainComponent> safeThis(this);
  menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&modelsMenuButton_),
                     [safeThis](const int result) {
                       if (safeThis == nullptr) {
                         return;
                       }
                       switch (result) {
                         case 1:
                           safeThis->modelController_->fetchCatalog();
                           break;
                         case 2:
                           safeThis->modelController_->showInstalled();
                           break;
                         case 3:
                           safeThis->modelController_->checkUpdates();
                           break;
                         case 4:
                           safeThis->modelController_->verifyIntegrity();
                           break;
                         case 5: {
                           const auto folder = juce::File(
                               (std::filesystem::path("assets") / "modelhub").string());
                           folder.createDirectory();
                           folder.revealToUser();
                           break;
                         }
                         default:
                           break;
                       }
                     });
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

  std::optional<ai::ModelPack> mixPack;
  if (const auto* selected = findPackById(modelManager_, modelManager_.activePackId("mix")); selected != nullptr) {
    mixPack = *selected;
  }

  processingController_->runAutoMix(session_, mixPack, cancelRender_);
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
  std::optional<ai::ModelPack> masterPack;
  if (const auto* selected = findPackById(modelManager_, modelManager_.activePackId("master")); selected != nullptr) {
    masterPack = *selected;
  }

  processingController_->runAutoMaster(session_, settings, preset, masterPack, cancelRender_);
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
    const auto baseRenderSettings = buildCurrentRenderSettings("");

    processingController_->runBatch(inputFolder, baseRenderSettings, cancelRender_);

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

    cancelRender_.store(false);
    taskRunning_.store(true);
    cancelButton_.setEnabled(true);
    statusLabel_.setText("Export started", juce::dontSendNotification);
    appendTaskHistory("Export started: " + selected.getFullPathName());

    exportController_->runExport(session_, settings, analysisEntries_, cancelRender_);

    exportChooser_.reset();
  });
}

} // namespace automix::app
