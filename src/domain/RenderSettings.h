#pragma once

#include <map>
#include <string>
#include <vector>

namespace automix::domain {

struct RenderSettings {
  int outputSampleRate = 44100;
  int blockSize = 1024;
  int outputBitDepth = 24;
  std::string outputPath;
  std::string outputFormat = "auto";
  bool writePerExportReportJson = true;
  std::string exportSpeedMode = "final";
  std::string gpuExecutionProvider = "auto";
  int lossyBitrateKbps = 320;
  int lossyQuality = 7;
  bool mp3UseVbr = false;
  int mp3VbrQuality = 4;
  int processingThreads = 0;
  int renderParallelism = 0;
  bool preferHardwareAcceleration = true;
  std::string metadataPolicy = "copy_all";
  std::map<std::string, std::string> metadataTemplate;
  std::string rendererName = "PhaseLimiter";
  bool rendererChainEnabled = false;
  std::string rendererChainMode = "logical_all";
  std::vector<std::string> rendererChain;
  std::string externalRendererPath;
  int externalRendererTimeoutMs = 300000;
};

} // namespace automix::domain
