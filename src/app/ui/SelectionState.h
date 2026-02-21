#pragma once

#include <map>
#include <optional>
#include <string>

#include "domain/MasterPlan.h"

namespace automix::app {

class SelectionState {
 public:
  void clearRendererIds();
  void bindRendererId(int comboId, std::string rendererId);
  std::optional<std::string> rendererIdForCombo(int comboId) const;
  std::optional<int> comboIdForRendererId(const std::string& rendererId) const;

  void clearCodecFormats();
  void bindCodecFormat(int comboId, std::string formatId);
  std::optional<std::string> codecFormatForCombo(int comboId) const;
  std::optional<int> comboIdForCodecFormat(const std::string& formatId) const;

  void clearExportSpeedModes();
  void bindExportSpeedMode(int comboId, std::string mode);
  std::optional<std::string> exportSpeedModeForCombo(int comboId) const;
  std::optional<int> comboIdForExportSpeedMode(const std::string& mode) const;

  void clearRendererChainModes();
  void bindRendererChainMode(int comboId, std::string mode);
  std::optional<std::string> rendererChainModeForCombo(int comboId) const;
  std::optional<int> comboIdForRendererChainMode(const std::string& mode) const;

  void clearProjectProfileIds();
  void bindProjectProfileId(int comboId, std::string profileId);
  std::optional<std::string> projectProfileIdForCombo(int comboId) const;
  std::optional<int> comboIdForProjectProfileId(const std::string& profileId) const;
  int firstProjectProfileComboId() const;

  void clearMasterPresets();
  void bindMasterPreset(int comboId, domain::MasterPreset preset);
  std::optional<domain::MasterPreset> masterPresetForCombo(int comboId) const;
  std::optional<int> comboIdForMasterPreset(domain::MasterPreset preset) const;

  void clearPlatformPresets();
  void bindPlatformPreset(int comboId, domain::MasterPreset preset);
  std::optional<domain::MasterPreset> platformPresetForCombo(int comboId) const;
  std::optional<int> comboIdForPlatformPreset(domain::MasterPreset preset) const;

 private:
  std::map<int, std::string> rendererIdByComboId_;
  std::map<int, std::string> codecFormatByComboId_;
  std::map<int, std::string> exportSpeedModeByComboId_;
  std::map<int, std::string> rendererChainModeByComboId_;
  std::map<int, std::string> projectProfileIdByComboId_;
  std::map<int, domain::MasterPreset> masterPresetByComboId_;
  std::map<int, domain::MasterPreset> platformPresetByComboId_;
};

} // namespace automix::app
