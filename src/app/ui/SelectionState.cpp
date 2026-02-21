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

template <typename T>
std::optional<int> findComboIdForValue(const std::map<int, T>& values, const T& value) {
  for (const auto& [comboId, stored] : values) {
    if (stored == value) {
      return comboId;
    }
  }
  return std::nullopt;
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

std::optional<int> SelectionState::comboIdForRendererId(const std::string& rendererId) const {
  return findComboIdForValue(rendererIdByComboId_, rendererId);
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

std::optional<int> SelectionState::comboIdForCodecFormat(const std::string& formatId) const {
  return findComboIdForValue(codecFormatByComboId_, formatId);
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

std::optional<int> SelectionState::comboIdForExportSpeedMode(const std::string& mode) const {
  return findComboIdForValue(exportSpeedModeByComboId_, mode);
}

void SelectionState::clearRendererChainModes() {
  rendererChainModeByComboId_.clear();
}

void SelectionState::bindRendererChainMode(const int comboId, std::string mode) {
  rendererChainModeByComboId_[comboId] = std::move(mode);
}

std::optional<std::string> SelectionState::rendererChainModeForCombo(const int comboId) const {
  return findByComboId(rendererChainModeByComboId_, comboId);
}

std::optional<int> SelectionState::comboIdForRendererChainMode(const std::string& mode) const {
  return findComboIdForValue(rendererChainModeByComboId_, mode);
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

std::optional<int> SelectionState::comboIdForProjectProfileId(const std::string& profileId) const {
  return findComboIdForValue(projectProfileIdByComboId_, profileId);
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

std::optional<int> SelectionState::comboIdForMasterPreset(const domain::MasterPreset preset) const {
  return findComboIdForValue(masterPresetByComboId_, preset);
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

std::optional<int> SelectionState::comboIdForPlatformPreset(const domain::MasterPreset preset) const {
  return findComboIdForValue(platformPresetByComboId_, preset);
}

} // namespace automix::app
