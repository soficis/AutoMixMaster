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
#include "ai/FeatureSchema.h"
#include "ai/OnnxModelInference.h"
#include "analysis/StemAnalyzer.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/BatchQueueRunner.h"
#include "engine/LoudnessMeter.h"
#include "engine/OfflineRenderPipeline.h"
#include "renderers/RendererPipeline.h"
#include "util/StringUtils.h"
#include "util/WavWriter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

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

std::map<std::string, std::string> activePackMapForUi(const ai::ModelManager& modelManager) {
  std::map<std::string, std::string> active;
  for (const auto* scope : {"mix", "master", "analysis", "separation"}) {
    const auto id = modelManager.activePackId(scope);
    if (!id.empty()) {
      active[scope] = id;
    }
  }
  return active;
}

void configureInferenceBackend(ai::OnnxModelInference& inference,
                               const ai::ModelPack& pack,
                               const std::string& providerPreference) {
  auto resolvedProvider = providerPreference;
  if ((resolvedProvider.empty() || util::toLower(resolvedProvider) == "auto") && !pack.providerAffinity.empty()) {
    resolvedProvider = pack.providerAffinity.front();
  }
  inference.setExecutionProviderPreference(resolvedProvider);
  inference.setGraphOptimizationEnabled(true);
  inference.setWarmupEnabled(true);
  inference.setPreferQuantizedVariants(util::toLower(pack.preferredPrecision) != "fp32");
  inference.setPreferredPrecision(pack.preferredPrecision.empty() ? "auto" : pack.preferredPrecision);
  inference.setThreadConfiguration(pack.defaultIntraOpThreads.value_or(0), pack.defaultInterOpThreads.value_or(0));
  inference.setProfilingEnabled(pack.enableProfiling);
}

std::string summarizeInferenceOutputs(const ai::InferenceResult& inferenceResult, const size_t maxEntries = 6) {
  if (inferenceResult.outputs.empty()) {
    return "(no outputs)";
  }

  std::vector<std::pair<std::string, double>> entries;
  entries.reserve(inferenceResult.outputs.size());
  for (const auto& [key, value] : inferenceResult.outputs) {
    entries.emplace_back(key, value);
  }
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

  std::ostringstream summary;
  summary << std::fixed << std::setprecision(3);
  const auto limit = std::min(maxEntries, entries.size());
  for (size_t index = 0; index < limit; ++index) {
    if (index > 0) {
      summary << ", ";
    }
    summary << entries[index].first << "=" << entries[index].second;
  }
  if (entries.size() > limit) {
    summary << ", ...";
  }
  return summary.str();
}

std::string summarizeInferenceDelta(const ai::InferenceResult& beforeResult,
                                    const ai::InferenceResult& afterResult,
                                    const size_t maxEntries = 6) {
  std::vector<std::tuple<std::string, double, double>> deltas;
  for (const auto& [key, afterValue] : afterResult.outputs) {
    const auto beforeIt = beforeResult.outputs.find(key);
    if (beforeIt == beforeResult.outputs.end()) {
      continue;
    }
    deltas.emplace_back(key, beforeIt->second, afterValue - beforeIt->second);
  }

  if (deltas.empty()) {
    return "(no shared output keys)";
  }

  std::sort(deltas.begin(), deltas.end(), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

  std::ostringstream summary;
  summary << std::fixed << std::setprecision(3);
  const auto limit = std::min(maxEntries, deltas.size());
  for (size_t index = 0; index < limit; ++index) {
    if (index > 0) {
      summary << ", ";
    }
    summary << std::get<0>(deltas[index]) << "_delta=" << std::get<2>(deltas[index]);
  }
  if (deltas.size() > limit) {
    summary << ", ...";
  }
  return summary.str();
}

juce::String toChainPreviewText(const std::vector<std::string>& chain) {
  juce::String text("Active chain: ");
  if (chain.empty()) {
    text += "BuiltIn";
    return text;
  }

  for (size_t i = 0; i < chain.size(); ++i) {
    if (i > 0) {
      text += " -> ";
    }
    text += chain[i].c_str();
  }
  return text;
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

std::filesystem::path uiPreferencesPath() {
  const auto appDataDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
  return std::filesystem::path(appDataDir.getFullPathName().toStdString()) / "AutoMixMaster" / "ui_preferences.json";
}

constexpr const char* kBatchRecursivePreferenceKey = "batchRecursiveScan";
constexpr const char* kExportReportSidecarPreferenceKey = "writePerExportReportJson";

nlohmann::json loadUiPreferences() {
  try {
    std::ifstream input(uiPreferencesPath());
    if (!input.is_open()) {
      return nlohmann::json::object();
    }
    nlohmann::json json;
    input >> json;
    if (!json.is_object()) {
      return nlohmann::json::object();
    }
    return json;
  } catch (...) {
    return nlohmann::json::object();
  }
}

void saveUiPreferences(nlohmann::json preferences) {
  try {
    const auto path = uiPreferencesPath();
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
      return;
    }

    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
      return;
    }
    if (!preferences.is_object()) {
      preferences = nlohmann::json::object();
    }
    output << preferences.dump(2);
  } catch (...) {
  }
}

bool loadBatchRecursivePreference() {
  const auto json = loadUiPreferences();
  return json.value(kBatchRecursivePreferenceKey, false);
}

void saveBatchRecursivePreference(const bool enabled) {
  auto json = loadUiPreferences();
  json[kBatchRecursivePreferenceKey] = enabled;
  saveUiPreferences(std::move(json));
}

bool loadExportReportSidecarPreference() {
  const auto json = loadUiPreferences();
  return json.value(kExportReportSidecarPreferenceKey, true);
}

void saveExportReportSidecarPreference(const bool enabled) {
  auto json = loadUiPreferences();
  json[kExportReportSidecarPreferenceKey] = enabled;
  saveUiPreferences(std::move(json));
}

void setBatchRecursiveEnvironment(const bool enabled) {
#if defined(_WIN32)
  _putenv_s("AUTOMIX_BATCH_RECURSIVE", enabled ? "1" : "0");
#else
  setenv("AUTOMIX_BATCH_RECURSIVE", enabled ? "1" : "0", 1);
#endif
}

class SettingsPanel final : public juce::Component {
 public:
  SettingsPanel(juce::AudioDeviceManager& audioDeviceManager,
                const bool writeReportJsonSidecar,
                std::function<void(bool)> onWriteReportSidecarChanged)
      : audioSelector_(audioDeviceManager, 0, 0, 0, 2, false, false, true, false),
        onWriteReportSidecarChanged_(std::move(onWriteReportSidecarChanged)) {
    reportSidecarToggle_.setButtonText("Write .report.json sidecar next to each exported file");
    reportSidecarToggle_.setTooltip("Disable to export only audio files without per-file JSON report sidecars.");
    reportSidecarToggle_.setToggleState(writeReportJsonSidecar, juce::dontSendNotification);
    reportSidecarToggle_.onClick = [this] {
      if (onWriteReportSidecarChanged_) {
        onWriteReportSidecarChanged_(reportSidecarToggle_.getToggleState());
      }
    };

    addAndMakeVisible(reportSidecarToggle_);
    addAndMakeVisible(audioSelector_);
  }

  void resized() override {
    auto area = getLocalBounds().reduced(10);
    reportSidecarToggle_.setBounds(area.removeFromTop(28));
    area.removeFromTop(10);
    audioSelector_.setBounds(area);
  }

 private:
  juce::AudioDeviceSelectorComponent audioSelector_;
  juce::ToggleButton reportSidecarToggle_;
  std::function<void(bool)> onWriteReportSidecarChanged_;
};

bool isStemTrimSeparator(const char value) {
  return std::isspace(static_cast<unsigned char>(value)) != 0 || value == '_' || value == '-' || value == '.';
}

std::string trimStemTokenSeparators(std::string value) {
  value = util::trim(std::move(value));
  while (!value.empty() && isStemTrimSeparator(value.front())) {
    value.erase(value.begin());
  }
  while (!value.empty() && isStemTrimSeparator(value.back())) {
    value.pop_back();
  }
  return value;
}

std::string sanitizeFileStem(std::string value) {
  static constexpr size_t kMaxStemLength = 80;
  value = trimStemTokenSeparators(std::move(value));
  if (value.empty()) {
    return "song";
  }

  for (char& ch : value) {
    switch (ch) {
      case '<':
      case '>':
      case ':':
      case '"':
      case '/':
      case '\\':
      case '|':
      case '?':
      case '*':
        ch = '_';
        break;
      default:
        break;
    }
  }

  if (value.size() > kMaxStemLength) {
    value.resize(kMaxStemLength);
    value = trimStemTokenSeparators(std::move(value));
  }

  if (value.empty()) {
    return "song";
  }

  return value;
}

std::string stripStemRoleSuffix(std::string value) {
  value = trimStemTokenSeparators(std::move(value));
  if (value.empty()) {
    return value;
  }

  const auto lower = util::toLower(value);
  static const std::vector<std::string> roleTokens = {
      "vocals", "vocal", "vox", "bass", "drums", "drum", "kick", "snare",
      "guitar", "gtr", "piano", "keys", "key", "synth", "fx", "effects", "sfx",
      "other", "music", "mix"};

  if (lower.size() > 3 && lower.back() == ')') {
    const auto openPos = lower.find_last_of('(');
    if (openPos != std::string::npos && openPos > 0 && openPos + 1 < lower.size() - 1) {
      const auto role = trimStemTokenSeparators(lower.substr(openPos + 1, lower.size() - openPos - 2));
      if (std::find(roleTokens.begin(), roleTokens.end(), role) != roleTokens.end()) {
        return trimStemTokenSeparators(value.substr(0, openPos));
      }
    }
  }

  static constexpr char separators[] = {'_', '-', ' '};
  for (const auto& role : roleTokens) {
    for (const auto sep : separators) {
      const auto suffix = std::string(1, sep) + role;
      if (lower.size() > suffix.size() && lower.ends_with(suffix)) {
        return trimStemTokenSeparators(value.substr(0, value.size() - suffix.size()));
      }
    }
  }

  return value;
}

std::string deriveSongTitleFromSession(const domain::Session& session) {
  if (session.originalMixPath.has_value() && !session.originalMixPath->empty()) {
    const auto path = std::filesystem::path(*session.originalMixPath);
    const auto stem = stripStemRoleSuffix(path.stem().string());
    if (!stem.empty()) {
      return sanitizeFileStem(stem);
    }
  }

  const auto sessionName = trimStemTokenSeparators(session.sessionName);
  if (!sessionName.empty() && util::toLower(sessionName) != "untitled session") {
    const auto stem = stripStemRoleSuffix(sessionName);
    if (!stem.empty()) {
      return sanitizeFileStem(stem);
    }
  }

  if (!session.stems.empty()) {
    const auto stem = stripStemRoleSuffix(session.stems.front().name);
    if (!stem.empty()) {
      return sanitizeFileStem(stem);
    }
  }

  return "song";
}

juce::File buildUniqueDatedExportFile(const juce::File& folder,
                                      const std::string& title,
                                      const juce::String& ext) {
  const auto dateStamp = juce::Time::getCurrentTime().formatted("%Y%m%d");
  const juce::String safeTitle(title);

  for (int index = 1; index <= 9999; ++index) {
    const auto sequence = juce::String(index).paddedLeft('0', 2);
    const auto fileName = safeTitle + "_AutoMixMaster_" + dateStamp + "_" + sequence + "." + ext;
    const auto candidate = folder.getChildFile(fileName);
    if (!candidate.existsAsFile()) {
      return candidate;
    }
  }

  return folder.getNonexistentChildFile(safeTitle + "_AutoMixMaster_" + dateStamp, ext, false);
}

double linearToDbFs(const double linear) {
  constexpr double minValue = 1.0e-12;
  return 20.0 * std::log10(std::max(linear, minValue));
}

struct DifferenceMetrics {
  double referenceRmsDbfs = -120.0;
  double outputRmsDbfs = -120.0;
  double residualRmsDbfs = -120.0;
  double residualRelativeDb = -120.0;
  bool changed = false;
  bool audiblyDifferent = false;
};

DifferenceMetrics analyzeDifference(const engine::AudioBuffer& reference, const engine::AudioBuffer& output) {
  const int channels = std::min(reference.getNumChannels(), output.getNumChannels());
  const int samples = std::min(reference.getNumSamples(), output.getNumSamples());
  if (channels <= 0 || samples <= 0) {
    throw std::runtime_error("Unable to compare buffers: no overlapping channels or samples.");
  }

  double referenceEnergy = 0.0;
  double outputEnergy = 0.0;
  double residualEnergy = 0.0;
  const double normalization = static_cast<double>(channels * samples);

  for (int ch = 0; ch < channels; ++ch) {
    for (int i = 0; i < samples; ++i) {
      const double refSample = static_cast<double>(reference.getSample(ch, i));
      const double outSample = static_cast<double>(output.getSample(ch, i));
      const double residual = outSample - refSample;
      referenceEnergy += refSample * refSample;
      outputEnergy += outSample * outSample;
      residualEnergy += residual * residual;
    }
  }

  const double referenceRms = std::sqrt(referenceEnergy / normalization);
  const double outputRms = std::sqrt(outputEnergy / normalization);
  const double residualRms = std::sqrt(residualEnergy / normalization);

  DifferenceMetrics metrics;
  metrics.referenceRmsDbfs = linearToDbFs(referenceRms);
  metrics.outputRmsDbfs = linearToDbFs(outputRms);
  metrics.residualRmsDbfs = linearToDbFs(residualRms);

  const double baseline = std::max(referenceRms, outputRms);
  metrics.residualRelativeDb = linearToDbFs(residualRms / std::max(baseline, 1.0e-12));
  metrics.changed = metrics.residualRelativeDb > -80.0;
  metrics.audiblyDifferent = metrics.residualRelativeDb > -42.0;
  return metrics;
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
  applySessionUiSelections();
  syncSessionUiSelections();

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
    cb.onProgress = [safe](const double progress) {
      juce::MessageManager::callAsync([safe, progress]() {
        if (safe && safe->taskOrchestrator_->activeTask() == ActiveTask::Model)
          safe->taskOrchestrator_->setProgress(progress);
      });
    };
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
    cb.onProgress = [safe](const double progress) {
      juce::MessageManager::callAsync([safe, progress]() {
        if (safe && safe->taskOrchestrator_->activeTask() == ActiveTask::Import)
          safe->taskOrchestrator_->setProgress(progress);
      });
    };
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
    cb.onProgress = [safe](const double progress) {
      juce::MessageManager::callAsync([safe, progress]() {
        if (safe && safe->taskOrchestrator_->activeTask() == ActiveTask::Export)
          safe->taskOrchestrator_->setProgress(progress);
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
          // Pipeline mode: continue to Auto Master automatically
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

        safe->previewManager_->setBuffer(result.previewMaster);
        safe->heroWaveform_->setBuffer(result.previewMaster);
        safe->updateTransportFromBuffer(result.previewMaster);

        safe->taskOrchestrator_->finishTaskCompleted(ActiveTask::AutoMaster, "Auto Master complete");

        if (!safe->pendingPipelineExportFolder_.empty()) {
          // Pipeline mode: continue to export automatically
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
    cb.onProgress = [safe](const double progress) {
      juce::MessageManager::callAsync([safe, progress]() {
        if (safe && safe->taskOrchestrator_->activeTask() == ActiveTask::Session)
          safe->taskOrchestrator_->setProgress(progress);
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
  fb.items.add(juce::FlexItem(*heroWaveform_).withFlex(1.5f).withMinHeight(120.0f));
  fb.items.add(juce::FlexItem(*transportBar_).withHeight(static_cast<float>(kTransportHeight)));
  fb.items.add(juce::FlexItem(*controlDeck_).withFlex(2.5f).withMinHeight(180.0f));
  fb.items.add(juce::FlexItem(*taskCenter_).withFlex(1.2f).withMinHeight(160.0f));

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
  if (key == juce::KeyPress('m', ctrl | juce::ModifierKeys::shiftModifier, 0)) {
    onAutoMixMaster();
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
  controlDeck_->onAutoMixMaster = [this] { onAutoMixMaster(); };
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
    batchVerificationInputFolder_ = std::filesystem::path(selected.getFullPathName().toStdString());
    batchVerificationRecursiveScan_ = controlDeck_->getBatchRecursiveToggle().getToggleState();

    auto settings = buildCurrentRenderSettings("");
    batchVerificationSettings_ = settings;
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

void MainLayout::startExportVerification(const std::string& outputAudioPath) {
  if (outputAudioPath.empty()) {
    return;
  }

  if (!exportVerificationSession_.has_value() || !exportVerificationSettings_.has_value()) {
    taskOrchestrator_->appendHistory("Export verification skipped: missing export context.");
    return;
  }

  auto sessionSnapshot = exportVerificationSession_.value();
  auto settingsSnapshot = exportVerificationSettings_.value();
  const auto analysisPack = resolveActiveModelPackForTask("analysis");
  exportVerificationSession_.reset();
  exportVerificationSettings_.reset();

  auto safe = safeAsync(this);

  struct VerifyExportJob final : juce::ThreadPoolJob {
    domain::Session session;
    domain::RenderSettings settings;
    std::string exportPath;
    std::optional<ai::ModelPack> analysisPack;
    juce::Component::SafePointer<MainLayout> safeLayout;

    VerifyExportJob(domain::Session sessionSnapshot,
                    domain::RenderSettings settingsSnapshot,
                    std::string outputPath,
                    std::optional<ai::ModelPack> analysisPackSnapshot,
                    juce::Component::SafePointer<MainLayout> safePtr)
        : juce::ThreadPoolJob("VerifyExportJob"),
          session(std::move(sessionSnapshot)),
          settings(std::move(settingsSnapshot)),
          exportPath(std::move(outputPath)),
          analysisPack(std::move(analysisPackSnapshot)),
          safeLayout(std::move(safePtr)) {}

    JobStatus runJob() override {
      juce::String reportText;
      try {
        engine::AudioFileIO fileIO;
        auto exported = fileIO.readAudioFile(exportPath);
        if (exported.getNumSamples() <= 0 || exported.getNumChannels() <= 0) {
          throw std::runtime_error("Exported file contained no decodable audio samples.");
        }

        engine::OfflineRenderPipeline pipeline;
        auto mixedRawResult = pipeline.renderRawMix(session, settings, {}, nullptr);
        if (mixedRawResult.cancelled || mixedRawResult.mixBuffer.getNumSamples() <= 0) {
          throw std::runtime_error("Unable to render pre-master reference mix for verification.");
        }

        auto exportedComparable = exported;
        if (std::abs(exportedComparable.getSampleRate() - mixedRawResult.mixBuffer.getSampleRate()) > 1.0e-6) {
          engine::AudioResampler resampler;
          exportedComparable = resampler.resampleLinear(exportedComparable, mixedRawResult.mixBuffer.getSampleRate());
        }

        analysis::StemAnalyzer analyzer;
        const auto masteringDiff = analyzeDifference(mixedRawResult.mixBuffer, exportedComparable);

        std::optional<DifferenceMetrics> mixDiff;
        if (session.mixPlan.has_value()) {
          auto baselineSession = session;
          baselineSession.mixPlan.reset();
          auto baselineRawResult = pipeline.renderRawMix(baselineSession, settings, {}, nullptr);
          if (!baselineRawResult.cancelled && baselineRawResult.mixBuffer.getNumSamples() > 0) {
            mixDiff = analyzeDifference(baselineRawResult.mixBuffer, mixedRawResult.mixBuffer);
          }
        }

        engine::LoudnessMeter loudnessMeter;
        const auto preMasterMetrics = loudnessMeter.analyze(mixedRawResult.mixBuffer);
        const auto postMasterMetrics = loudnessMeter.analyze(exportedComparable);
        const auto preMasterAnalysisMetrics = analyzer.analyzeBuffer(mixedRawResult.mixBuffer);
        const auto postMasterAnalysisMetrics = analyzer.analyzeBuffer(exportedComparable);

        std::optional<ai::InferenceResult> preMasterAnalysis;
        std::optional<ai::InferenceResult> postMasterAnalysis;
        std::string analysisDetails;
        if (analysisPack.has_value()) {
          const auto modelPath = analysisPack->rootPath / analysisPack->modelFile;
          if (util::toLower(modelPath.extension().string()) != ".onnx") {
            analysisDetails = "Analysis pack '" + analysisPack->id +
                              "' skipped: only ONNX analysis packs are supported in verification.";
          } else {
            ai::OnnxModelInference inference;
            configureInferenceBackend(inference, analysisPack.value(), settings.gpuExecutionProvider);
            if (!inference.loadModel(modelPath)) {
              analysisDetails = "Analysis pack '" + analysisPack->id + "' failed to load: " + modelPath.string();
            } else {
              const auto task = analysisPack->type.empty() ? std::string("analysis_model") : analysisPack->type;
              preMasterAnalysis = inference.run(ai::InferenceRequest{
                  .task = task,
                  .features = ai::FeatureSchemaV1::extract(preMasterAnalysisMetrics),
              });
              postMasterAnalysis = inference.run(ai::InferenceRequest{
                  .task = task,
                  .features = ai::FeatureSchemaV1::extract(postMasterAnalysisMetrics),
              });
              analysisDetails = inference.backendDiagnostics();
            }
          }
        }

        std::ostringstream report;
        report << std::fixed << std::setprecision(2);
        report << "Verification report for " << exportPath << "\n";
        report << "Mastering applied: " << (masteringDiff.changed ? "yes" : "no") << "\n";
        report << "Audible difference (proxy): " << (masteringDiff.audiblyDifferent ? "likely yes" : "subtle/none") << "\n";
        report << "Residual vs pre-master: " << masteringDiff.residualRelativeDb << " dB\n";
        report << "Pre-master LUFS: " << preMasterMetrics.integratedLufs
               << " | Post-master LUFS: " << postMasterMetrics.integratedLufs << "\n";
        report << "Pre-master peak: " << preMasterMetrics.samplePeakDbfs
               << " dBFS | Post-master peak: " << postMasterMetrics.samplePeakDbfs << " dBFS\n";
        if (mixDiff.has_value()) {
          report << "Mixing applied: " << (mixDiff->changed ? "yes" : "no") << "\n";
          report << "Mix residual vs baseline: " << mixDiff->residualRelativeDb << " dB\n";
        } else {
          report << "Mixing applied: skipped (no mix plan available)\n";
        }
        if (analysisPack.has_value()) {
          report << "Analysis pack: " << analysisPack->id << "\n";
          if (!analysisDetails.empty()) {
            report << "Analysis backend: " << analysisDetails << "\n";
          }
          if (preMasterAnalysis.has_value() && postMasterAnalysis.has_value() &&
              preMasterAnalysis->usedModel && postMasterAnalysis->usedModel) {
            report << "Analysis outputs (pre-master): "
                   << summarizeInferenceOutputs(preMasterAnalysis.value()) << "\n";
            report << "Analysis outputs (post-master): "
                   << summarizeInferenceOutputs(postMasterAnalysis.value()) << "\n";
            report << "Analysis output deltas (post - pre): "
                   << summarizeInferenceDelta(preMasterAnalysis.value(), postMasterAnalysis.value()) << "\n";
          } else {
            report << "Analysis outputs: unavailable (model rejected task or returned no usable outputs)\n";
          }
        }
        report << "Note: audible difference uses an objective residual-energy proxy, not a psychoacoustic AB test.";
        reportText = report.str();
      } catch (const std::exception& error) {
        reportText = juce::String("Verification failed: ") + error.what();
      } catch (...) {
        reportText = "Verification failed: unknown error";
      }

      juce::MessageManager::callAsync([safeLayout = safeLayout, reportText]() {
        if (!safeLayout) {
          return;
        }
        safeLayout->taskOrchestrator_->appendHistory(reportText);
      });

      return jobHasFinished;
    }
  };

  backgroundPool_.addJob(
      new VerifyExportJob(std::move(sessionSnapshot),
                          std::move(settingsSnapshot),
                          outputAudioPath,
                          analysisPack,
                          safe),
      true);
}

void MainLayout::startBatchVerification(const std::string& outputFolder) {
  if (outputFolder.empty()) {
    return;
  }
  if (!batchVerificationInputFolder_.has_value() || !batchVerificationSettings_.has_value()) {
    taskOrchestrator_->appendHistory("Batch verification skipped: missing batch context.");
    return;
  }

  const auto inputFolderSnapshot = batchVerificationInputFolder_.value();
  const auto settingsSnapshot = batchVerificationSettings_.value();
  const auto analysisPack = resolveActiveModelPackForTask("analysis");
  const bool recursiveSnapshot = batchVerificationRecursiveScan_;
  batchVerificationInputFolder_.reset();
  batchVerificationSettings_.reset();

  auto safe = safeAsync(this);

  struct VerifyBatchJob final : juce::ThreadPoolJob {
    std::filesystem::path inputFolder;
    std::filesystem::path outputFolder;
    domain::RenderSettings settings;
    std::optional<ai::ModelPack> analysisPack;
    bool recursiveScan = false;
    juce::Component::SafePointer<MainLayout> safeLayout;

    VerifyBatchJob(std::filesystem::path inputPath,
                   std::filesystem::path outputPath,
                   domain::RenderSettings renderSettings,
                   std::optional<ai::ModelPack> analysisPackSnapshot,
                   const bool recursive,
                   juce::Component::SafePointer<MainLayout> safePtr)
        : juce::ThreadPoolJob("VerifyBatchJob"),
          inputFolder(std::move(inputPath)),
          outputFolder(std::move(outputPath)),
          settings(std::move(renderSettings)),
          analysisPack(std::move(analysisPackSnapshot)),
          recursiveScan(recursive),
          safeLayout(std::move(safePtr)) {}

    JobStatus runJob() override {
      juce::String reportText;
      try {
        engine::BatchQueueRunner runner;
        auto items = runner.buildItemsFromFolder(inputFolder, outputFolder, recursiveScan);
        if (items.empty()) {
          throw std::runtime_error("No batch items available for verification.");
        }

        const auto resolvedFormat = util::WavWriter::resolveFormat(std::filesystem::path{}, settings.outputFormat);
        const auto requiredExtension = util::extensionForFormat(resolvedFormat);
        for (auto& item : items) {
          if (util::toLower(item.outputPath.extension().string()) != requiredExtension) {
            item.outputPath.replace_extension(requiredExtension);
          }
        }

        analysis::StemAnalyzer analyzer;
        automix::HeuristicAutoMixStrategy autoMix;
        engine::OfflineRenderPipeline pipeline;
        engine::AudioFileIO fileIO;
        engine::AudioResampler resampler;
        engine::LoudnessMeter meter;

        int verified = 0;
        int missingOutputs = 0;
        int masteringApplied = 0;
        int masteringAudible = 0;
        int mixingApplied = 0;
        int mixingAudible = 0;
        double masteringResidualSumDb = 0.0;
        double mixingResidualSumDb = 0.0;
        double loudnessDeltaSum = 0.0;
        int analysisEvaluated = 0;
        int analysisConfidenceCount = 0;
        double analysisConfidencePreSum = 0.0;
        double analysisConfidencePostSum = 0.0;
        std::string analysisPackDiagnostics;
        std::optional<std::string> firstAnalysisPreview;
        ai::OnnxModelInference analysisInference;
        bool analysisInferenceReady = false;
        std::string analysisTask = "analysis_model";

        if (analysisPack.has_value()) {
          const auto modelPath = analysisPack->rootPath / analysisPack->modelFile;
          if (util::toLower(modelPath.extension().string()) == ".onnx") {
            configureInferenceBackend(analysisInference, analysisPack.value(), settings.gpuExecutionProvider);
            if (analysisInference.loadModel(modelPath)) {
              analysisInferenceReady = true;
              analysisTask = analysisPack->type.empty() ? std::string("analysis_model") : analysisPack->type;
              analysisPackDiagnostics = analysisInference.backendDiagnostics();
            } else {
              analysisPackDiagnostics = "failed to load analysis model at " + modelPath.string();
            }
          } else {
            analysisPackDiagnostics = "analysis pack is not ONNX and was skipped";
          }
        }

        for (auto& item : items) {
          try {
            if (!std::filesystem::exists(item.outputPath)) {
              ++missingOutputs;
              continue;
            }

            auto outputBuffer = fileIO.readAudioFile(item.outputPath.string());
            if (outputBuffer.getNumSamples() <= 0 || outputBuffer.getNumChannels() <= 0) {
              ++missingOutputs;
              continue;
            }

            auto mixedSession = item.session;
            const auto analysisEntries = analyzer.analyzeSession(mixedSession);
            mixedSession.mixPlan = autoMix.buildPlan(mixedSession, analysisEntries, 1.0);

            auto mixedRawResult = pipeline.renderRawMix(mixedSession, settings, {}, nullptr);
            if (mixedRawResult.cancelled || mixedRawResult.mixBuffer.getNumSamples() <= 0) {
              continue;
            }

            auto baselineSession = mixedSession;
            baselineSession.mixPlan.reset();
            auto baselineRawResult = pipeline.renderRawMix(baselineSession, settings, {}, nullptr);
            if (baselineRawResult.cancelled || baselineRawResult.mixBuffer.getNumSamples() <= 0) {
              continue;
            }

            auto comparableOutput = outputBuffer;
            if (std::abs(comparableOutput.getSampleRate() - mixedRawResult.mixBuffer.getSampleRate()) > 1.0e-6) {
              comparableOutput = resampler.resampleLinear(comparableOutput, mixedRawResult.mixBuffer.getSampleRate());
            }

            const auto masteringDiff = analyzeDifference(mixedRawResult.mixBuffer, comparableOutput);
            const auto mixingDiff = analyzeDifference(baselineRawResult.mixBuffer, mixedRawResult.mixBuffer);
            const auto preMasterMetrics = meter.analyze(mixedRawResult.mixBuffer);
            const auto postMasterMetrics = meter.analyze(comparableOutput);
            const auto preMasterAnalysisMetrics = analyzer.analyzeBuffer(mixedRawResult.mixBuffer);
            const auto postMasterAnalysisMetrics = analyzer.analyzeBuffer(comparableOutput);

            ++verified;
            masteringApplied += masteringDiff.changed ? 1 : 0;
            masteringAudible += masteringDiff.audiblyDifferent ? 1 : 0;
            mixingApplied += mixingDiff.changed ? 1 : 0;
            mixingAudible += mixingDiff.audiblyDifferent ? 1 : 0;
            masteringResidualSumDb += masteringDiff.residualRelativeDb;
            mixingResidualSumDb += mixingDiff.residualRelativeDb;
            loudnessDeltaSum += (postMasterMetrics.integratedLufs - preMasterMetrics.integratedLufs);

            if (analysisInferenceReady) {
              const auto preAnalysis = analysisInference.run(ai::InferenceRequest{
                  .task = analysisTask,
                  .features = ai::FeatureSchemaV1::extract(preMasterAnalysisMetrics),
              });
              const auto postAnalysis = analysisInference.run(ai::InferenceRequest{
                  .task = analysisTask,
                  .features = ai::FeatureSchemaV1::extract(postMasterAnalysisMetrics),
              });
              if (preAnalysis.usedModel && postAnalysis.usedModel) {
                ++analysisEvaluated;
                const auto preConfidenceIt = preAnalysis.outputs.find("confidence");
                const auto postConfidenceIt = postAnalysis.outputs.find("confidence");
                if (preConfidenceIt != preAnalysis.outputs.end() && postConfidenceIt != postAnalysis.outputs.end()) {
                  analysisConfidencePreSum += preConfidenceIt->second;
                  analysisConfidencePostSum += postConfidenceIt->second;
                  ++analysisConfidenceCount;
                }
                if (!firstAnalysisPreview.has_value()) {
                  firstAnalysisPreview = "pre: " + summarizeInferenceOutputs(preAnalysis) +
                                         " | post: " + summarizeInferenceOutputs(postAnalysis);
                }
              }
            }
          } catch (...) {
            ++missingOutputs;
          }
        }

        std::ostringstream report;
        report << std::fixed << std::setprecision(2);
        report << "Batch verification summary\n";
        report << "Input folder: " << inputFolder.string() << "\n";
        report << "Output folder: " << outputFolder.string() << "\n";
        report << "Items discovered: " << items.size() << "\n";
        report << "Items verified: " << verified << "\n";
        report << "Missing/undecodable outputs: " << missingOutputs << "\n";
        if (analysisPack.has_value()) {
          report << "Analysis pack configured: " << analysisPack->id << "\n";
          if (!analysisPackDiagnostics.empty()) {
            report << "Analysis backend: " << analysisPackDiagnostics << "\n";
          }
        }

        if (verified > 0) {
          const auto count = static_cast<double>(verified);
          report << "Mixing applied: " << mixingApplied << "/" << verified
                 << " (audible proxy: " << mixingAudible << "/" << verified << ")\n";
          report << "Mastering applied: " << masteringApplied << "/" << verified
                 << " (audible proxy: " << masteringAudible << "/" << verified << ")\n";
          report << "Average mix residual vs baseline: " << (mixingResidualSumDb / count) << " dB\n";
          report << "Average master residual vs pre-master: " << (masteringResidualSumDb / count) << " dB\n";
          report << "Average LUFS delta (post - pre): " << (loudnessDeltaSum / count) << " LU\n";
          if (analysisPack.has_value()) {
            report << "Analysis evaluations: " << analysisEvaluated << "/" << verified << "\n";
            if (analysisConfidenceCount > 0) {
              const double confidenceCount = static_cast<double>(analysisConfidenceCount);
              report << "Average analysis confidence pre: " << (analysisConfidencePreSum / confidenceCount) << "\n";
              report << "Average analysis confidence post: " << (analysisConfidencePostSum / confidenceCount) << "\n";
            }
            if (firstAnalysisPreview.has_value()) {
              report << "Analysis output sample: " << firstAnalysisPreview.value() << "\n";
            }
          }
        } else {
          report << "No outputs were verified.\n";
        }

        report << "Note: audible difference uses residual-energy proxy, not a psychoacoustic AB test.";
        reportText = report.str();
      } catch (const std::exception& error) {
        reportText = juce::String("Batch verification failed: ") + error.what();
      } catch (...) {
        reportText = "Batch verification failed: unknown error";
      }

      juce::MessageManager::callAsync([safeLayout = safeLayout, reportText]() {
        if (!safeLayout) {
          return;
        }
        safeLayout->taskOrchestrator_->appendHistory(reportText);
      });

      return jobHasFinished;
    }
  };

  backgroundPool_.addJob(
      new VerifyBatchJob(inputFolderSnapshot,
                         outputFolder,
                         settingsSnapshot,
                         analysisPack,
                         recursiveSnapshot,
                         safe),
      true);
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
