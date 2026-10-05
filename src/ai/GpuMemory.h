#pragma once

#include <cstdint>
#include <optional>
#include <string>

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

// Apple's CoreML and Neural Engine providers are opt-in on Macs with little RAM.
// Measured on an 8 GiB MacBook Neo (BS-RoFormer, 1 run, nothing else running):
// both providers were killed by the OS (SIGKILL) during the first inference, and
// the CPU path did not finish a run in 5 minutes. "auto" therefore skips them
// below kCoreMlAutoMinMemoryBytes unless the user opts in with
// AUTOMIX_ENABLE_COREML=1. Naming the provider explicitly always works.
inline constexpr std::uint64_t kCoreMlAutoMinMemoryBytes = 12ull * 1024 * 1024 * 1024;

// Pure policy: may "auto" try CoreML/ANE on a machine with this much RAM?
// Unknown memory (0) is allowed, because nothing better is known.
bool coreMlAutoAllowed(std::uint64_t physicalMemoryBytes, bool explicitOptIn);

// Warning for the Vocal Model toggle: empty unless the model would run on the CPU of a
// low-memory Mac (below kCoreMlAutoMinMemoryBytes, no GPU session usable). The 8 GiB
// MacBook Neo needed over 5 minutes per chunk on the CPU against about 25 s on a desktop.
// Empty too when the light vocal model (Open-Unmix) is already the active separation pack.
std::string vocalModelCpuWarning(std::uint64_t physicalMemoryBytes, bool gpuSessionUsable, bool isMac,
                                 bool lightModelActive);

// Live version for this machine; probes for a usable GPU session.
std::string vocalModelCpuWarning(bool lightModelActive);

// Live policy for this machine. Always true off macOS, where CoreML does not exist.
bool coreMlAutoAllowed();

} // namespace automix::ai
