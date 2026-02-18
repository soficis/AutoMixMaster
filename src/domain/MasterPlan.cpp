#include "domain/MasterPlan.h"

#include <algorithm>
#include <cctype>

namespace automix::domain {
namespace {

std::string normalized(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

} // namespace

std::string toString(const MasterPreset preset) {
  switch (preset) {
    case MasterPreset::DefaultStreaming:
      return "default_streaming";
    case MasterPreset::Broadcast:
      return "broadcast";
    case MasterPreset::UdioOptimized:
      return "udio_optimized";
    case MasterPreset::Spotify:
      return "spotify";
    case MasterPreset::AppleMusic:
      return "apple_music";
    case MasterPreset::YouTube:
      return "youtube";
    case MasterPreset::AmazonMusic:
      return "amazon_music";
    case MasterPreset::Tidal:
      return "tidal";
    case MasterPreset::BroadcastEbuR128:
      return "broadcast_ebu_r128";
    case MasterPreset::Custom:
      return "custom";
  }
  return "default_streaming";
}

MasterPreset masterPresetFromString(const std::string& value) {
  const auto text = normalized(value);
  if (text == "broadcast") {
    return MasterPreset::Broadcast;
  }
  if (text == "udio_optimized") {
    return MasterPreset::UdioOptimized;
  }
  if (text == "spotify") {
    return MasterPreset::Spotify;
  }
  if (text == "apple_music") {
    return MasterPreset::AppleMusic;
  }
  if (text == "youtube") {
    return MasterPreset::YouTube;
  }
  if (text == "amazon_music") {
    return MasterPreset::AmazonMusic;
  }
  if (text == "tidal") {
    return MasterPreset::Tidal;
  }
  if (text == "broadcast_ebu_r128") {
    return MasterPreset::BroadcastEbuR128;
  }
  if (text == "custom") {
    return MasterPreset::Custom;
  }
  return MasterPreset::DefaultStreaming;
}

} // namespace automix::domain
