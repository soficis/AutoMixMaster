#pragma once

#include <filesystem>

#include "engine/AudioBuffer.h"

namespace automix::util {

class WavWriter {
 public:
  void write(const std::filesystem::path& path,
             const engine::AudioBuffer& buffer,
             int bitDepth) const;
};

} // namespace automix::util
