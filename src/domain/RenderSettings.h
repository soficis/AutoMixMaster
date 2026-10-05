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
  bool referenceMasteringEnabled = false;
  // Experimental ITO-Master route. Not sufficient on its own: ItoMasterStrategy
  // additionally requires the experimental toggle, CC BY-NC consent and a complete
  // pack. Never default this on.
  bool itoMasteringEnabled = false;
  // Tensor (BS-RoFormer) stem separation on single-mix import. Not sufficient on
  // its own: StemSeparator additionally requires an active separation pack that
  // carries a tensor_contract and a native ONNX Runtime build. Never default this on.
  bool tensorSeparationEnabled = false;
  std::string metadataPolicy = "copy_all";
  std::map<std::string, std::string> metadataTemplate;
  // PhaseLimiter is opt-in: selecting it (here or as a custom chain stage) is
  // the only way it runs, including in the logical_all chain.
  std::string rendererName = "BuiltIn";
  bool rendererChainEnabled = false;
  std::string rendererChainMode = "logical_all";
  std::vector<std::string> rendererChain;
  std::string externalRendererPath;
  int externalRendererTimeoutMs = 300000;
};

} // namespace automix::domain
