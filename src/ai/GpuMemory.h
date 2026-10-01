#pragma once

#include <cstdint>
#include <optional>

namespace automix::ai {

struct GpuMemoryInfo {
  std::uint64_t freeBytes = 0;
  std::uint64_t totalBytes = 0;
};

// Free and total memory of CUDA device 0 (the device the CUDA execution
// provider uses), read through the CUDA runtime loaded at run time so nothing
// links against CUDA. Empty when no CUDA runtime is loadable or the query
// fails, e.g. on a machine without an NVIDIA GPU.
std::optional<GpuMemoryInfo> queryCudaDeviceMemory();

// Install-time choice: is the device big enough for a model that needs
// `requiredBytes` while it runs? Judged on total memory with headroom for what
// the desktop and other applications typically hold, since free memory at
// install time says little about free memory at separation time.
bool gpuFitsModel(const std::optional<GpuMemoryInfo>& memory, std::uint64_t requiredBytes);

// Run-time choice: can a model needing `requiredBytes` run on the GPU right
// now without spilling into shared system memory? Unknown memory is treated as
// yes, because there is nothing better to go on and a failed GPU run is
// retried on CPU anyway.
bool gpuHasRoomNow(const std::optional<GpuMemoryInfo>& memory, std::uint64_t requiredBytes);

} // namespace automix::ai
