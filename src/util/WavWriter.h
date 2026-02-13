#pragma once

#include <filesystem>

#include "engine/AudioBuffer.h"

namespace automix::util {

class WavWriter {
 public:
  void write(const std::filesystem::path& path,
             const engine::AudioBuffer& buffer,
             int bitDepth,
             const std::string& preferredFormat = "auto",
             int lossyBitrateKbps = 192,
             int lossyQuality = 7) const;

  static std::string resolveFormat(const std::filesystem::path& path, const std::string& preferredFormat);
  static bool isLossyFormat(const std::string& format);
};

} // namespace automix::util
