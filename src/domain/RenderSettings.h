#pragma once

#include <string>

namespace automix::domain {

struct RenderSettings {
  int outputSampleRate = 44100;
  int blockSize = 1024;
  int outputBitDepth = 24;
  std::string outputPath;
  std::string outputFormat = "auto";
  std::string exportSpeedMode = "final";
  std::string gpuExecutionProvider = "auto";
  int lossyBitrateKbps = 320;
  int lossyQuality = 7;
  bool mp3UseVbr = false;
  int mp3VbrQuality = 4;
  int processingThreads = 0;
  bool preferHardwareAcceleration = true;
  std::string rendererName = "BuiltIn";
  std::string externalRendererPath;
  int externalRendererTimeoutMs = 300000;
};

} // namespace automix::domain
