#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <nlohmann/json.hpp>

#include "ai/ModelManager.h"
#include "ai/OnnxModelInference.h"
#include "automaster/IAutoMasterStrategy.h"
#include "domain/Session.h"
#include "engine/AudioBuffer.h"
#include "renderers/RendererRegistry.h"
#include "util/StringUtils.h"

#include "app/ui/TaskLifecycle.h"  // for ActiveTask (used by cbFactory templates)

namespace automix::app::detail {

// ── Safe pointer helper ────────────────────────────────────────

template <typename Comp>
auto safeAsync(Comp* comp) {
  return juce::Component::SafePointer<Comp>(comp);
}

// ── Model pack helpers ─────────────────────────────────────────

inline std::map<std::string, std::string> activePackMapForUi(const ai::ModelManager& modelManager) {
  std::map<std::string, std::string> active;
  for (const auto* scope : {"mix", "master", "analysis", "separation"}) {
    const auto id = modelManager.activePackId(scope);
    if (!id.empty()) {
      active[scope] = id;
    }
  }
  return active;
}

inline void configureInferenceBackend(ai::OnnxModelInference& inference,
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

// ── Inference summary helpers ──────────────────────────────────

inline std::string summarizeInferenceOutputs(const ai::InferenceResult& inferenceResult, const size_t maxEntries = 6) {
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

inline std::string summarizeInferenceDelta(const ai::InferenceResult& beforeResult,
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

// ── Chain preview ──────────────────────────────────────────────

inline juce::String toChainPreviewText(const std::vector<std::string>& chain) {
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

// ── Stem panel helpers (body in MainLayout.cpp — needs StemPanel full type) ──

void updateStemPanelFromSession(class StemPanel& panel, const domain::Session& session);

// ── UI preferences ─────────────────────────────────────────────

inline std::filesystem::path uiPreferencesPath() {
  const auto appDataDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
  return std::filesystem::path(appDataDir.getFullPathName().toStdString()) / "AutoMixMaster" / "ui_preferences.json";
}

inline constexpr const char* kBatchRecursivePreferenceKey = "batchRecursiveScan";
inline constexpr const char* kExportReportSidecarPreferenceKey = "writePerExportReportJson";

inline nlohmann::json loadUiPreferences() {
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

inline void saveUiPreferences(nlohmann::json preferences) {
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

inline bool loadBatchRecursivePreference() {
  const auto json = loadUiPreferences();
  return json.value(kBatchRecursivePreferenceKey, false);
}

inline void saveBatchRecursivePreference(const bool enabled) {
  auto json = loadUiPreferences();
  json[kBatchRecursivePreferenceKey] = enabled;
  saveUiPreferences(std::move(json));
}

inline bool loadExportReportSidecarPreference() {
  const auto json = loadUiPreferences();
  return json.value(kExportReportSidecarPreferenceKey, true);
}

inline void saveExportReportSidecarPreference(const bool enabled) {
  auto json = loadUiPreferences();
  json[kExportReportSidecarPreferenceKey] = enabled;
  saveUiPreferences(std::move(json));
}

inline void setBatchRecursiveEnvironment(const bool enabled) {
#if defined(_WIN32)
  _putenv_s("AUTOMIX_BATCH_RECURSIVE", enabled ? "1" : "0");
#else
  setenv("AUTOMIX_BATCH_RECURSIVE", enabled ? "1" : "0", 1);
#endif
}

// ── Settings panel (small JUCE Component) ──────────────────────

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

// ── Stem string helpers ────────────────────────────────────────

inline bool isStemTrimSeparator(const char value) {
  return std::isspace(static_cast<unsigned char>(value)) != 0 || value == '_' || value == '-' || value == '.';
}

inline std::string trimStemTokenSeparators(std::string value) {
  value = util::trim(std::move(value));
  while (!value.empty() && isStemTrimSeparator(value.front())) {
    value.erase(value.begin());
  }
  while (!value.empty() && isStemTrimSeparator(value.back())) {
    value.pop_back();
  }
  return value;
}

inline std::string sanitizeFileStem(std::string value) {
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

inline std::string stripStemRoleSuffix(std::string value) {
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

inline std::string deriveSongTitleFromSession(const domain::Session& session) {
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

// ── File naming helpers ────────────────────────────────────────

inline juce::File buildUniqueDatedExportFile(const juce::File& folder,
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

// ── Audio comparison ───────────────────────────────────────────

inline double linearToDbFs(const double linear) {
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

inline DifferenceMetrics analyzeDifference(const engine::AudioBuffer& reference, const engine::AudioBuffer& output) {
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

// ── External renderer config loading ───────────────────────────

inline std::vector<renderers::ExternalRendererConfig> loadConfiguredExternalRenderers(
    std::function<void(const juce::String&)> onError) {
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
      if (onError)
        onError("External renderer config parse failed: " + juce::String(path.string()) +
                " (" + juce::String(error.what()) + ")");
      continue;
    } catch (...) {
      if (onError)
        onError("External renderer config parse failed: " + juce::String(path.string()) +
                " (unknown error)");
      continue;
    }
  }

  return configs;
}

// ── Callback factory helpers for controller creation ───────────
// These reduce boilerplate in MainLayout::initControllers().
// Usage: cb.onStatus = cbFactory::onStatus(safe);

namespace cbFactory {

template <typename SafePtr>
auto onStatus(SafePtr safe) {
  return [safe](const std::string& msg) {
    juce::MessageManager::callAsync([safe, msg]() {
      if (safe && safe->getTaskOrchestrator()) safe->getTaskOrchestrator()->setStatus(juce::String(msg), "");
    });
  };
}

template <typename SafePtr>
auto onHistory(SafePtr safe) {
  return [safe](const std::string& msg) {
    juce::MessageManager::callAsync([safe, msg]() {
      if (safe && safe->getTaskOrchestrator()) safe->getTaskOrchestrator()->appendHistory(juce::String(msg));
    });
  };
}

template <typename SafePtr>
auto onProgressForTask(SafePtr safe, ActiveTask task) {
  return [safe, task](const double progress) {
    juce::MessageManager::callAsync([safe, task, progress]() {
      if (safe && safe->getTaskOrchestrator() && safe->getTaskOrchestrator()->activeTask() == task)
        safe->getTaskOrchestrator()->setProgress(progress);
    });
  };
}

} // namespace cbFactory

} // namespace automix::app::detail
