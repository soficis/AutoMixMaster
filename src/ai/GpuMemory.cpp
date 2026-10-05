#include "ai/GpuMemory.h"

#include <cstddef>

#include <juce_core/juce_core.h>

#include "ai/GpuRuntimePack.h"
#include "ai/OnnxTensorInference.h"

namespace automix::ai {
namespace {

// Headroom on top of a model's need when judging total memory: a Windows
// desktop held 1.6-2.2 GB on the 16 GB card this was measured on. Sized so a
// 12 GB card (reported as just under 12 GiB) still qualifies for a 10 GiB model.
constexpr std::uint64_t kInstallHeadroomBytes = 1536ull * 1024 * 1024;

using CudaMemGetInfo = int (*)(std::size_t* freeBytes, std::size_t* totalBytes);

} // namespace

std::optional<GpuMemoryInfo> queryCudaDeviceMemory() {
  GpuRuntimePack::preload();  // a per-user CUDA runtime, if installed
#if defined(_WIN32)
  const char* candidates[] = {"cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"};
#elif defined(__APPLE__)
  const char* candidates[] = {"libcudart.dylib"};
#else
  const char* candidates[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so"};
#endif
  for (const auto* name : candidates) {
    juce::DynamicLibrary library;
    if (!library.open(name)) {
      continue;
    }
    const auto query = reinterpret_cast<CudaMemGetInfo>(library.getFunction("cudaMemGetInfo"));
    if (query == nullptr) {
      continue;
    }
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (query(&freeBytes, &totalBytes) != 0 || totalBytes == 0) {  // 0 == cudaSuccess
      return std::nullopt;
    }
    return GpuMemoryInfo{freeBytes, totalBytes};
  }
  return std::nullopt;
}

bool gpuFitsModel(const std::optional<GpuMemoryInfo>& memory, const std::uint64_t requiredBytes) {
  return memory.has_value() && memory->totalBytes >= requiredBytes + kInstallHeadroomBytes;
}

bool gpuHasRoomNow(const std::optional<GpuMemoryInfo>& memory, const std::uint64_t requiredBytes) {
  return !memory.has_value() || memory->freeBytes >= requiredBytes;
}

bool coreMlAutoAllowed(const std::uint64_t physicalMemoryBytes, const bool explicitOptIn) {
  return explicitOptIn || physicalMemoryBytes == 0 || physicalMemoryBytes >= kCoreMlAutoMinMemoryBytes;
}

std::string vocalModelCpuWarning(const std::uint64_t physicalMemoryBytes, const bool gpuSessionUsable,
                                 const bool isMac) {
  if (!isMac || gpuSessionUsable || physicalMemoryBytes == 0 || physicalMemoryBytes >= kCoreMlAutoMinMemoryBytes) {
    return {};
  }
  const auto gib = physicalMemoryBytes / (1024ull * 1024 * 1024);
  return "This Mac has " + std::to_string(gib) +
         " GB of memory and no usable GPU acceleration for the vocal model. It will run on the CPU: expect "
         "several minutes per 20-second chunk (hours for a full song), heavy memory swapping, and possible "
         "failure. Install the light vocal model instead (Open-Unmix, 36 MB, about 20x faster than real time on "
         "a CPU) from the Model Hub, or leave Vocal Model off.";
}

std::string vocalModelCpuWarning() {
#if JUCE_MAC
  std::string provider;
  const bool gpuUsable = gpuTensorSessionAvailable(&provider);
  const auto physicalBytes = static_cast<std::uint64_t>(juce::SystemStats::getMemorySizeInMegabytes()) * 1024 * 1024;
  return vocalModelCpuWarning(physicalBytes, gpuUsable, true);
#else
  return {};
#endif
}

bool coreMlAutoAllowed() {
#if JUCE_MAC
  const bool optIn = juce::SystemStats::getEnvironmentVariable("AUTOMIX_ENABLE_COREML", "") == "1";
  const auto physicalBytes = static_cast<std::uint64_t>(juce::SystemStats::getMemorySizeInMegabytes()) * 1024 * 1024;
  return coreMlAutoAllowed(physicalBytes, optIn);
#else
  return true;
#endif
}

} // namespace automix::ai
