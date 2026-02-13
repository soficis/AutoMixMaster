#include "domain/MasterPlan.h"

namespace automix::domain {

std::string toString(const MasterPreset preset) {
  switch (preset) {
    case MasterPreset::DefaultStreaming:
      return "default_streaming";
    case MasterPreset::Broadcast:
      return "broadcast";
    case MasterPreset::UdioOptimized:
      return "udio_optimized";
    case MasterPreset::Custom:
      return "custom";
  }
  return "default_streaming";
}

MasterPreset masterPresetFromString(const std::string& value) {
  if (value == "broadcast") {
    return MasterPreset::Broadcast;
  }
  if (value == "udio_optimized") {
    return MasterPreset::UdioOptimized;
  }
  if (value == "custom") {
    return MasterPreset::Custom;
  }
  return MasterPreset::DefaultStreaming;
}

} // namespace automix::domain
