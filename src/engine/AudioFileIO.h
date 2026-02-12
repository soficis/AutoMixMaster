#pragma once

#include <filesystem>

#include "engine/AudioBuffer.h"

namespace automix::engine {

class AudioFileIO {
 public:
  AudioBuffer readAudioFile(const std::filesystem::path& filePath) const;
};

} // namespace automix::engine
