#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "engine/AudioBuffer.h"

namespace automix::engine {

class AudioFileIO {
 public:
  AudioBuffer readAudioFile(const std::filesystem::path& filePath) const;
  std::map<std::string, std::string> readMetadata(const std::filesystem::path& filePath) const;
};

} // namespace automix::engine
