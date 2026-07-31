#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace automix::ai {

struct GpuInfo {
  std::string name;
  size_t vramBytes = 0;
  bool cuda = false;
  bool directml = false;
  bool coreml = false;
  bool ane = false;
};

class GpuCapabilityDetector final {
 public:
  GpuCapabilityDetector() = delete;

  /// Enumerate GPU adapters visible to the host OS.
  /// The returned list is empty when no GPU-supporting runtime (ONNX Runtime,
  /// CUDA driver, DirectML, CoreML) is available or no compatible hardware
  /// was discovered.
  static std::vector<GpuInfo> detectGpus();

  /// True when the CUDA runtime (NVIDIA GPU driver + CUDA libraries) is
  /// available on this machine.
  static bool hasCudaSupport();

  /// True when DirectML is available (Windows 10+ with a WDDM 2.x driver).
  static bool hasDirectMlSupport();

  /// True when CoreML is available (macOS 10.13+).
  static bool hasCoreMlSupport();

  /// True when the Apple Neural Engine is available (Apple Silicon).
  static bool hasAneSupport();

  /// Returns the canonical provider name best suited for this platform,
  /// preferring GPU over CPU.  Delegates to gpu::platformPreferredProvider()
  /// and validates with the runtime checks above.
  static std::string recommendProvider();
};

} // namespace automix::ai
