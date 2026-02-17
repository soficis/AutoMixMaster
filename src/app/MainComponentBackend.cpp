#include "app/MainComponent.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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

  const auto availability = exportController_ != nullptr
                                ? exportController_->listCodecAvailability()
                                : util::WavWriter::getAvailableFormats();
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
  if (exportController_ != nullptr) {
    return exportController_->selectedExportSpeedMode(exportSpeedModeBox_.getSelectedId(), exportSpeedModeByComboId_);
  }
  const auto it = exportSpeedModeByComboId_.find(exportSpeedModeBox_.getSelectedId());
  return it == exportSpeedModeByComboId_.end() ? kExportSpeedModeFinal : it->second;
}

bool MainComponent::isQuickExportModeSelected() const {
  if (exportController_ != nullptr) {
    return exportController_->isQuickExportMode(selectedExportSpeedMode());
  }
  return selectedExportSpeedMode() == kExportSpeedModeQuick;
}

void MainComponent::applyQuickExportDefaults() {
  if (exportController_ == nullptr) {
    return;
  }

  const auto defaults = exportController_->quickExportDefaults(codecFormatByComboId_);
  for (const auto& [comboId, formatName] : codecFormatByComboId_) {
    if (toLower(formatName) == toLower(defaults.outputFormat)) {
      exportFormatBox_.setSelectedId(comboId, juce::dontSendNotification);
      break;
    }
  }
  if (defaults.usedFallbackCodec) {
    statusLabel_.setText("Quick mode: MP3 unavailable, using fallback codec", juce::dontSendNotification);
    appendTaskHistory("Quick mode fallback codec selected (MP3 unavailable)");
  }

  exportBitrateSlider_.setValue(static_cast<double>(defaults.lossyBitrateKbps), juce::dontSendNotification);
  mp3ModeBox_.setSelectedId(defaults.mp3UseVbr ? 2 : 1, juce::dontSendNotification);
  mp3VbrSlider_.setValue(static_cast<double>(defaults.mp3VbrQuality), juce::dontSendNotification);
  session_.renderSettings.outputFormat = defaults.outputFormat;
  session_.renderSettings.lossyBitrateKbps = defaults.lossyBitrateKbps;
  session_.renderSettings.lossyQuality = defaults.lossyQuality;
  session_.renderSettings.mp3UseVbr = defaults.mp3UseVbr;
  session_.renderSettings.mp3VbrQuality = defaults.mp3VbrQuality;
}

void MainComponent::updateExportCodecControls() {
  if (exportController_ == nullptr) {
    return;
  }
  const auto formatIt = codecFormatByComboId_.find(exportFormatBox_.getSelectedId());
  const std::string selectedFormat = formatIt != codecFormatByComboId_.end() ? formatIt->second : "wav";
  const bool mp3UseVbr = mp3ModeBox_.getSelectedId() == 2;
  const auto controls =
      exportController_->codecControlsFor(selectedFormat, mp3UseVbr, selectedExportSpeedMode());

  exportFormatBox_.setEnabled(controls.formatEnabled);
  exportFormatLabel_.setEnabled(controls.formatEnabled);
  exportBitrateSlider_.setEnabled(controls.bitrateEnabled);
  exportBitrateLabel_.setEnabled(controls.bitrateEnabled);
  mp3ModeBox_.setEnabled(controls.mp3ModeEnabled);
  mp3ModeLabel_.setEnabled(controls.mp3ModeEnabled);
  mp3VbrSlider_.setEnabled(controls.mp3VbrEnabled);
  mp3VbrLabel_.setEnabled(controls.mp3VbrEnabled);
}

void MainComponent::refreshProjectProfiles() {
  if (profileController_ != nullptr) {
    projectProfiles_ = profileController_->loadProfiles(std::filesystem::current_path());
  } else {
    projectProfiles_ = domain::loadProjectProfiles(std::filesystem::current_path());
  }
  projectProfileBox_.clear(juce::dontSendNotification);
  projectProfileIdByComboId_.clear();

  std::optional<domain::ProjectProfile> selectedProfile;
  if (profileController_ != nullptr) {
    selectedProfile = profileController_->selectedProfile(projectProfiles_, session_.projectProfileId);
  } else {
    selectedProfile = domain::findProjectProfile(projectProfiles_, session_.projectProfileId);
  }

  int selectedId = 0;
  int comboId = 1;
  for (const auto& profile : projectProfiles_) {
    projectProfileBox_.addItem(profile.name + " [" + profile.id + "]", comboId);
    projectProfileIdByComboId_[comboId] = profile.id;
    if (selectedProfile.has_value() && profile.id == selectedProfile->id) {
      selectedId = comboId;
    }
    ++comboId;
  }

  if (selectedId == 0 && !projectProfileIdByComboId_.empty()) {
    selectedId = projectProfileIdByComboId_.begin()->first;
  }

  if (selectedId > 0) {
    projectProfileBox_.setSelectedId(selectedId, juce::dontSendNotification);
    if (selectedProfile.has_value()) {
      applyProjectProfile(selectedProfile.value());
    } else {
      const auto it = projectProfileIdByComboId_.find(selectedId);
      if (it != projectProfileIdByComboId_.end()) {
        if (const auto fallbackProfile = domain::findProjectProfile(projectProfiles_, it->second);
            fallbackProfile.has_value()) {
          applyProjectProfile(fallbackProfile.value());
        }
      }
    }
  }
}

void MainComponent::applyProjectProfile(const domain::ProjectProfile& profile) {
  const auto applied = profileController_ != nullptr
                           ? profileController_->applyProfile(session_, profile)
                           : AppliedProfileSettings {};
  if (profileController_ == nullptr) {
    session_.projectProfileId = profile.id;
    session_.safetyPolicyId = profile.safetyPolicyId;
    session_.preferredStemCount = profile.preferredStemCount;
    session_.renderSettings.gpuExecutionProvider = profile.gpuProvider;
    session_.renderSettings.outputFormat = profile.outputFormat;
    session_.renderSettings.lossyBitrateKbps = profile.lossyBitrateKbps;
    session_.renderSettings.mp3UseVbr = profile.mp3UseVbr;
    session_.renderSettings.mp3VbrQuality = profile.mp3VbrQuality;
    session_.renderSettings.metadataPolicy = profile.metadataPolicy;
    session_.renderSettings.metadataTemplate = profile.metadataTemplate;
    session_.renderSettings.rendererName = profile.rendererName;
  }

  const auto gpuProvider = applied.gpuProvider.empty() ? profile.gpuProvider : applied.gpuProvider;
  if (gpuProvider == "cpu") {
    gpuProviderBox_.setSelectedId(2, juce::dontSendNotification);
  } else if (gpuProvider == "directml") {
    gpuProviderBox_.setSelectedId(3, juce::dontSendNotification);
  } else if (gpuProvider == "coreml") {
    gpuProviderBox_.setSelectedId(4, juce::dontSendNotification);
  } else if (gpuProvider == "cuda") {
    gpuProviderBox_.setSelectedId(5, juce::dontSendNotification);
  } else {
    gpuProviderBox_.setSelectedId(1, juce::dontSendNotification);
  }

  for (const auto& [comboId, format] : codecFormatByComboId_) {
    if (toLower(format) == toLower(session_.renderSettings.outputFormat)) {
      exportFormatBox_.setSelectedId(comboId, juce::dontSendNotification);
      break;
    }
  }
  exportBitrateSlider_.setValue(session_.renderSettings.lossyBitrateKbps, juce::dontSendNotification);
  mp3ModeBox_.setSelectedId(session_.renderSettings.mp3UseVbr ? 2 : 1, juce::dontSendNotification);
  mp3VbrSlider_.setValue(session_.renderSettings.mp3VbrQuality, juce::dontSendNotification);
  session_.renderSettings.exportSpeedMode = selectedExportSpeedMode();
  if (isQuickExportModeSelected()) {
    applyQuickExportDefaults();
  }
  updateExportCodecControls();

  const auto selectedRendererId = applied.rendererName.empty() ? profile.rendererName : applied.rendererName;
  for (const auto& [comboId, rendererId] : rendererIdByComboId_) {
    if (rendererId == selectedRendererId) {
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

  const auto roleModelId = applied.roleModelPackId.empty() ? profile.roleModelPackId : applied.roleModelPackId;
  const auto mixModelId = applied.mixModelPackId.empty() ? profile.mixModelPackId : applied.mixModelPackId;
  const auto masterModelId = applied.masterModelPackId.empty() ? profile.masterModelPackId : applied.masterModelPackId;
  selectModelComboById(roleModelBox_, roleModelIdByComboId_, roleModelId);
  selectModelComboById(mixModelBox_, mixModelIdByComboId_, mixModelId);
  selectModelComboById(masterModelBox_, masterModelIdByComboId_, masterModelId);

  modelManager_.setActivePackId("role", roleModelId);
  modelManager_.setActivePackId("mix", mixModelId);
  modelManager_.setActivePackId("master", masterModelId);

  const auto platformPreset = applied.platformPreset.empty() ? profile.platformPreset : applied.platformPreset;
  const auto normalizedPlatform = toLower(platformPreset);
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
    PreviewController::applyTransportBuffer(engine::AudioBuffer{},
                                            session_.timeline,
                                            transportController_,
                                            playbackCursorSamples_);
    return;
  }

  const auto generation = ++previewBuildGeneration_;
  const auto soloIt = stemIdBySoloComboId_.find(soloStemBox_.getSelectedId());
  const auto muteIt = stemIdByMuteComboId_.find(muteStemBox_.getSelectedId());
  const auto soloStemId = soloIt != stemIdBySoloComboId_.end() ? soloIt->second : std::string();
  const auto muteStemId = muteIt != stemIdByMuteComboId_.end() ? muteIt->second : std::string();
  if (previewController_ != nullptr) {
    PreviewBuildRequest request;
    request.session = session_;
    request.soloStemId = soloStemId;
    request.muteStemId = muteStemId;
    request.generation = generation;
    request.previousProgress = transportController_.progress();
    previewController_->rebuildPreview(std::move(request));
  }
}

void MainComponent::updateTransportFromBuffer(const engine::AudioBuffer& buffer) {
  {
    std::scoped_lock lock(playbackBufferMutex_);
    playbackBuffer_ = buffer;
  }
  waveformPreview_.setBuffer(buffer);
  waveformPreview_.setPlayheadProgress(0.0);
  PreviewController::applyTransportBuffer(buffer,
                                          session_.timeline,
                                          transportController_,
                                          playbackCursorSamples_);
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
  if (exportController_ == nullptr) {
    return session_.renderSettings;
  }

  BuildRenderSettingsRequest request;
  request.outputPath = outputPath;
  request.exportSpeedMode = selectedExportSpeedMode();
  request.metadataPolicy = session_.renderSettings.metadataPolicy;
  request.metadataTemplate = session_.renderSettings.metadataTemplate;
  const auto formatIt = codecFormatByComboId_.find(exportFormatBox_.getSelectedId());
  request.outputFormat = formatIt != codecFormatByComboId_.end() ? formatIt->second : "wav";
  request.lossyBitrateKbps = static_cast<int>(std::lround(exportBitrateSlider_.getValue()));
  request.mp3UseVbr = mp3ModeBox_.getSelectedId() == 2;
  request.mp3VbrQuality = static_cast<int>(std::lround(mp3VbrSlider_.getValue()));
  request.gpuProviderSelectionId = gpuProviderBox_.getSelectedId();
  request.rendererInfos = rendererInfos_;

  request.selectedRendererId = "BuiltIn";
  const auto rendererIt = rendererIdByComboId_.find(rendererBox_.getSelectedId());
  if (rendererIt != rendererIdByComboId_.end()) {
    request.selectedRendererId = rendererIt->second;
  }
  return exportController_->buildRenderSettings(request);
}

void MainComponent::beginCancelableTask(const juce::String& statusText,
                                        const juce::String& historyText,
                                        const ActiveTask activeTask) {
  cancelImport_.store(false);
  cancelModel_.store(false);
  cancelSession_.store(false);
  cancelMix_.store(false);
  cancelMaster_.store(false);
  cancelBatch_.store(false);
  cancelExport_.store(false);
  activeTask_ = activeTask;
  taskRunning_.store(true);
  cancelButton_.setEnabled(true);
  statusLabel_.setText(statusText, juce::dontSendNotification);
  appendTaskHistory(historyText);
}

void MainComponent::finishCancelableTask() {
  taskRunning_.store(false);
  cancelButton_.setEnabled(false);
  activeTask_ = ActiveTask::None;
}

void MainComponent::requestCancelForActiveTask() {
  switch (activeTask_) {
    case ActiveTask::Import:
      cancelImport_.store(true);
      return;
    case ActiveTask::Model:
      cancelModel_.store(true);
      return;
    case ActiveTask::Session:
      cancelSession_.store(true);
      return;
    case ActiveTask::AutoMix:
      cancelMix_.store(true);
      return;
    case ActiveTask::AutoMaster:
      cancelMaster_.store(true);
      return;
    case ActiveTask::Batch:
      cancelBatch_.store(true);
      return;
    case ActiveTask::Export:
      cancelExport_.store(true);
      return;
    case ActiveTask::None:
      break;
  }

  cancelImport_.store(true);
  cancelModel_.store(true);
  cancelSession_.store(true);
  cancelMix_.store(true);
  cancelMaster_.store(true);
  cancelBatch_.store(true);
  cancelExport_.store(true);
}

void MainComponent::onCancel() {
  requestCancelForActiveTask();
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

  const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
  importChooser_->launchAsync(flags, [safeThis](const juce::FileChooser& chooser) {
    if (safeThis == nullptr) {
      return;
    }
    const auto files = chooser.getResults();
    if (files.isEmpty()) {
      safeThis->importChooser_.reset();
      return;
    }

    std::vector<juce::File> selectedFiles;
    selectedFiles.reserve(static_cast<size_t>(files.size()));
    for (int i = 0; i < files.size(); ++i) {
      selectedFiles.push_back(files.getReference(i));
    }

    safeThis->beginCancelableTask("Import started",
                                  "Import started",
                                  ActiveTask::Import);
    safeThis->importController_->importFiles(std::move(selectedFiles),
                                             safeThis->separatedStemsToggle_.getToggleState(),
                                             safeThis->session_.preferredStemCount,
                                             safeThis->cancelImport_);
    safeThis->importChooser_.reset();
  });
}

void MainComponent::onImportOriginalMix() {
  originalMixChooser_ =
      std::make_unique<juce::FileChooser>("Select original stereo mix",
                                          juce::File(),
                                          "*.wav;*.aiff;*.aif;*.flac;*.mp3;*.ogg");
  constexpr int flags = juce::FileBrowserComponent::openMode |
                        juce::FileBrowserComponent::canSelectFiles;

  const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
  originalMixChooser_->launchAsync(flags, [safeThis](const juce::FileChooser& chooser) {
    if (safeThis == nullptr) {
      return;
    }
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      safeThis->originalMixChooser_.reset();
      return;
    }

    const auto result = safeThis->originalMixController_->applySelectedPath(selected.getFullPathName().toStdString(),
                                                                            selected.getFileName().toStdString());
    if (result.applied) {
      safeThis->session_.originalMixPath = result.path;
      safeThis->clearOriginalMixButton_.setEnabled(true);
      safeThis->statusLabel_.setText(result.statusText, juce::dontSendNotification);
      safeThis->reportEditor_.setText(safeThis->reportEditor_.getText() + "\n" + result.reportLine);
      safeThis->appendTaskHistory(result.taskHistoryLine);
    }
    safeThis->originalMixChooser_.reset();
  });
}

void MainComponent::onClearOriginalMix() {
  const auto result = originalMixController_->clear(session_.originalMixPath);
  if (!result.cleared) {
    statusLabel_.setText(result.statusText, juce::dontSendNotification);
    clearOriginalMixButton_.setEnabled(false);
    return;
  }

  session_.originalMixPath.reset();
  clearOriginalMixButton_.setEnabled(false);
  statusLabel_.setText(result.statusText, juce::dontSendNotification);
  reportEditor_.setText(reportEditor_.getText() + "\n" + result.reportLine);
  appendTaskHistory(result.taskHistoryLine);
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
  if (taskRunning_.load()) {
    statusLabel_.setText("A background task is already running", juce::dontSendNotification);
    return;
  }

  saveSessionChooser_ = std::make_unique<juce::FileChooser>(
      "Save session", juce::File::getSpecialLocation(juce::File::userDocumentsDirectory), "*.json");
  constexpr int flags = juce::FileBrowserComponent::saveMode |
                        juce::FileBrowserComponent::canSelectFiles |
                        juce::FileBrowserComponent::warnAboutOverwriting;

  const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
  saveSessionChooser_->launchAsync(flags, [safeThis](const juce::FileChooser& chooser) {
    if (safeThis == nullptr) {
      return;
    }
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      safeThis->saveSessionChooser_.reset();
      return;
    }

    safeThis->session_.renderSettings =
        safeThis->buildCurrentRenderSettings(safeThis->session_.renderSettings.outputPath);
    safeThis->session_.timeline.zoom = safeThis->zoomSlider_.getValue();
    safeThis->session_.timeline.fineScrub = safeThis->fineScrubToggle_.getToggleState();
    safeThis->session_.timeline.loopEnabled = safeThis->transportController_.loopEnabled();
    safeThis->session_.timeline.loopInSeconds = safeThis->transportController_.loopInSeconds();
    safeThis->session_.timeline.loopOutSeconds = safeThis->transportController_.loopOutSeconds();
    safeThis->beginCancelableTask("Saving session...",
                                  "Session save started: " + selected.getFullPathName(),
                                  ActiveTask::Session);
    safeThis->sessionController_->saveSession(selected.getFullPathName().toStdString(),
                                              safeThis->session_,
                                              safeThis->cancelSession_);
    safeThis->saveSessionChooser_.reset();
  });
}

void MainComponent::onLoadSession() {
  if (taskRunning_.load()) {
    statusLabel_.setText("A background task is already running", juce::dontSendNotification);
    return;
  }

  loadSessionChooser_ = std::make_unique<juce::FileChooser>("Load session", juce::File(), "*.json");
  constexpr int flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

  const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
  loadSessionChooser_->launchAsync(flags, [safeThis](const juce::FileChooser& chooser) {
    if (safeThis == nullptr) {
      return;
    }
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      safeThis->loadSessionChooser_.reset();
      return;
    }

    safeThis->beginCancelableTask("Loading session...",
                                  "Session load started: " + selected.getFullPathName(),
                                  ActiveTask::Session);
    safeThis->sessionController_->loadSession(selected.getFullPathName().toStdString(),
                                              safeThis->cancelSession_);
    safeThis->loadSessionChooser_.reset();
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

  const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
  externalRendererChooser_->launchAsync(flags, [safeThis](const juce::FileChooser& chooser) {
    if (safeThis == nullptr) {
      return;
    }
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      safeThis->externalRendererChooser_.reset();
      return;
    }

    const auto selectedPath = selected.getFullPathName().toStdString();
    const auto selectedName = selected.getFileName().toStdString();
    if (safeThis->exportController_ != nullptr) {
      safeThis->exportController_->validateExternalRenderer(selectedPath, selectedName);
    }

    safeThis->externalRendererChooser_.reset();
  });
}

void MainComponent::onPrefetchLame() {
  if (!util::LameDownloader::isSupportedOnCurrentPlatform()) {
    statusLabel_.setText("LAME downloader is not supported on this platform", juce::dontSendNotification);
    return;
  }

  prefetchLameButton_.setEnabled(false);
  statusLabel_.setText("Prefetching LAME...", juce::dontSendNotification);
  if (exportController_ != nullptr) {
    exportController_->prefetchLame();
  } else {
    prefetchLameButton_.setEnabled(true);
  }
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
                           if (safeThis->taskRunning_.load()) {
                             safeThis->statusLabel_.setText("Another background task is running", juce::dontSendNotification);
                             break;
                           }
                           safeThis->beginCancelableTask("Models: fetching Hugging Face catalog...",
                                                         "Models catalog fetch started",
                                                         ActiveTask::Model);
                           safeThis->modelController_->dispatchMenuAction(ModelMenuAction::BrowseAndDownload,
                                                                          &safeThis->cancelModel_);
                           break;
                         case 2:
                           safeThis->modelController_->dispatchMenuAction(ModelMenuAction::ShowInstalled);
                           break;
                         case 3:
                           if (safeThis->taskRunning_.load()) {
                             safeThis->statusLabel_.setText("Another background task is running", juce::dontSendNotification);
                             break;
                           }
                           safeThis->beginCancelableTask("Models: checking updates...",
                                                         "Model update check started",
                                                         ActiveTask::Model);
                           safeThis->modelController_->dispatchMenuAction(ModelMenuAction::CheckUpdates,
                                                                          &safeThis->cancelModel_);
                           break;
                         case 4:
                           safeThis->modelController_->dispatchMenuAction(ModelMenuAction::VerifyIntegrity);
                           break;
                         case 5:
                           safeThis->modelController_->dispatchMenuAction(ModelMenuAction::OpenHubFolder);
                           break;
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

  beginCancelableTask("Auto Mix started", "Auto Mix started", ActiveTask::AutoMix);

  std::optional<ai::ModelPack> mixPack;
  if (const auto* selected = findPackById(modelManager_, modelManager_.activePackId("mix")); selected != nullptr) {
    mixPack = *selected;
  }

  processingController_->runAutoMix(session_, mixPack, cancelMix_);
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

  beginCancelableTask("Auto Master started", "Auto Master started", ActiveTask::AutoMaster);

  auto preset = selectedPlatformPreset();
  if (preset == domain::MasterPreset::Custom) {
    preset = selectedMasterPreset();
  }

  const auto settings = buildCurrentRenderSettings("");
  std::optional<ai::ModelPack> masterPack;
  if (const auto* selected = findPackById(modelManager_, modelManager_.activePackId("master")); selected != nullptr) {
    masterPack = *selected;
  }

  processingController_->runAutoMaster(session_, settings, preset, masterPack, cancelMaster_);
}

void MainComponent::onBatchImport() {
  if (taskRunning_.load()) {
    statusLabel_.setText("A render task is already running", juce::dontSendNotification);
    return;
  }

  batchImportChooser_ = std::make_unique<juce::FileChooser>("Select folder for batch mastering");
  constexpr int flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories;

  const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
  batchImportChooser_->launchAsync(flags, [safeThis](const juce::FileChooser& chooser) {
    if (safeThis == nullptr) {
      return;
    }
    const auto folder = chooser.getResult();
    if (folder == juce::File()) {
      safeThis->batchImportChooser_.reset();
      return;
    }

    safeThis->beginCancelableTask("Batch preparing...",
                                  "Batch started: " + folder.getFullPathName(),
                                  ActiveTask::Batch);

    const std::filesystem::path inputFolder(folder.getFullPathName().toStdString());
    const auto baseRenderSettings = safeThis->buildCurrentRenderSettings("");

    safeThis->processingController_->runBatch(inputFolder, baseRenderSettings, safeThis->cancelBatch_);

    safeThis->batchImportChooser_.reset();
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
  ExportPreflightRequest preflightRequest;
  preflightRequest.selectedRendererId = selectedRendererId;
  preflightRequest.safetyPolicyId = session_.safetyPolicyId;
  preflightRequest.projectProfileId = session_.projectProfileId;
  preflightRequest.projectProfiles = projectProfiles_;
  const auto preflight = exportController_->preflight(preflightRequest);
  if (preflight.taskHistoryText.isNotEmpty()) {
    appendTaskHistory(preflight.taskHistoryText);
  }
  if (!preflight.allowed) {
    statusLabel_.setText(preflight.statusText, juce::dontSendNotification);
    return;
  }

  exportChooser_ = std::make_unique<juce::FileChooser>(
      "Export master",
      juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
      "*.wav;*.flac;*.aiff;*.ogg;*.mp3");
  constexpr int flags = juce::FileBrowserComponent::saveMode |
                        juce::FileBrowserComponent::canSelectFiles |
                        juce::FileBrowserComponent::warnAboutOverwriting;

  const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
  exportChooser_->launchAsync(flags, [safeThis](const juce::FileChooser& chooser) {
    if (safeThis == nullptr) {
      return;
    }
    const auto selected = chooser.getResult();
    if (selected == juce::File()) {
      safeThis->exportChooser_.reset();
      return;
    }

    auto settings = safeThis->buildCurrentRenderSettings(selected.getFullPathName().toStdString());

    safeThis->beginCancelableTask("Export started",
                                  "Export started: " + selected.getFullPathName(),
                                  ActiveTask::Export);

    safeThis->exportController_->runExport(safeThis->session_,
                                           settings,
                                           safeThis->analysisEntries_,
                                           safeThis->cancelExport_);

    safeThis->exportChooser_.reset();
  });
}


} // namespace automix::app
