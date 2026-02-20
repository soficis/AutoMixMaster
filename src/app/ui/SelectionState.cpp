#include "app/ui/SelectionState.h"

#include <utility>

namespace automix::app {
namespace {

template <typename T>
std::optional<T> findByComboId(const std::map<int, T>& values, const int comboId) {
  const auto it = values.find(comboId);
  if (it == values.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace

void SelectionState::clearRendererIds() {
  rendererIdByComboId_.clear();
}

void SelectionState::bindRendererId(const int comboId, std::string rendererId) {
  rendererIdByComboId_[comboId] = std::move(rendererId);
}

std::optional<std::string> SelectionState::rendererIdForCombo(const int comboId) const {
  return findByComboId(rendererIdByComboId_, comboId);
}

void SelectionState::clearCodecFormats() {
  codecFormatByComboId_.clear();
}

void SelectionState::bindCodecFormat(const int comboId, std::string formatId) {
  codecFormatByComboId_[comboId] = std::move(formatId);
}

std::optional<std::string> SelectionState::codecFormatForCombo(const int comboId) const {
  return findByComboId(codecFormatByComboId_, comboId);
}

void SelectionState::clearExportSpeedModes() {
  exportSpeedModeByComboId_.clear();
}

void SelectionState::bindExportSpeedMode(const int comboId, std::string mode) {
  exportSpeedModeByComboId_[comboId] = std::move(mode);
}

std::optional<std::string> SelectionState::exportSpeedModeForCombo(const int comboId) const {
  return findByComboId(exportSpeedModeByComboId_, comboId);
}

void SelectionState::clearProjectProfileIds() {
  projectProfileIdByComboId_.clear();
}

void SelectionState::bindProjectProfileId(const int comboId, std::string profileId) {
  projectProfileIdByComboId_[comboId] = std::move(profileId);
}

std::optional<std::string> SelectionState::projectProfileIdForCombo(const int comboId) const {
  return findByComboId(projectProfileIdByComboId_, comboId);
}

int SelectionState::firstProjectProfileComboId() const {
  if (projectProfileIdByComboId_.empty()) {
    return 0;
  }
  return projectProfileIdByComboId_.begin()->first;
}

void SelectionState::clearMasterPresets() {
  masterPresetByComboId_.clear();
}

void SelectionState::bindMasterPreset(const int comboId, const domain::MasterPreset preset) {
  masterPresetByComboId_[comboId] = preset;
}

std::optional<domain::MasterPreset> SelectionState::masterPresetForCombo(const int comboId) const {
  return findByComboId(masterPresetByComboId_, comboId);
}

void SelectionState::clearPlatformPresets() {
  platformPresetByComboId_.clear();
}

void SelectionState::bindPlatformPreset(const int comboId, const domain::MasterPreset preset) {
  platformPresetByComboId_[comboId] = preset;
}

std::optional<domain::MasterPreset> SelectionState::platformPresetForCombo(const int comboId) const {
  return findByComboId(platformPresetByComboId_, comboId);
}

} // namespace automix::app
