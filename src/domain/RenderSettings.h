#pragma once

#include <string>

namespace automix::domain {

struct RenderSettings {
  int outputSampleRate = 44100;
  int blockSize = 1024;
  int outputBitDepth = 24;
  std::string outputPath;
  std::string rendererName = "BuiltIn";
};

} // namespace automix::domain
