#include "ai/GpuCapabilityDetector.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "ai/GpuProvider.h"

namespace automix::ai {

namespace {

// ---------------------------------------------------------------------------
// Low-level OS helpers (no external dependencies beyond the platform CRT)
// ---------------------------------------------------------------------------

#if defined(_WIN32)
constexpr const char* kNvidiaSmiQuery =
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul";
#else
constexpr const char* kNvidiaSmiQuery =
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null";
#endif

// Try to run a command and capture its first line of output.
// Returns true on success and populates |out| with the trimmed line.
bool captureCommandOutput(const char* cmd, std::string& out) {
#if defined(_WIN32)
  FILE* pipe = _popen(cmd, "r");
#else
  FILE* pipe = popen(cmd, "r");
#endif
  if (pipe == nullptr) {
    return false;
  }

  std::array<char, 4096> buffer{};
  out.clear();
  if (std::fgets(buffer.data(), static_cast<int>(buffer.size()) - 1, pipe) != nullptr) {
    out = buffer.data();
    // Trim trailing whitespace / newline.
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) {
      out.pop_back();
    }
  }

#if defined(_WIN32)
  const int rc = _pclose(pipe);
#else
  const int rc = pclose(pipe);
#endif
  return rc == 0 && !out.empty();
}

// Returns true when the nvidia-smi binary is available on this system.
bool nvidiaSmiAvailable() {
  std::string dummy;
#if defined(_WIN32)
  return captureCommandOutput("where nvidia-smi.exe 2>nul", dummy);
#else
  return captureCommandOutput("which nvidia-smi 2>/dev/null", dummy);
#endif
}

// Parse a single line of nvidia-smi CSV output into a GpuInfo.
// Expected format: "NVIDIA GeForce RTX 4090, 24564 MiB"
GpuInfo parseNvidiaSmiLine(const std::string& line) {
  GpuInfo info;
  info.cuda = true;

  const auto comma = line.rfind(',');
  if (comma == std::string::npos) {
    info.name = line;
    return info;
  }

  info.name = line.substr(0, comma);
  // Trim trailing spaces from name.
  while (!info.name.empty() && std::isspace(static_cast<unsigned char>(info.name.back()))) {
    info.name.pop_back();
  }

  // Parse memory — format is "<number> MiB"
  const auto memPart = line.substr(comma + 1);
  std::istringstream memStream(memPart);
  double memValue = 0.0;
  std::string unit;
  memStream >> memValue >> unit;

  if (unit.find("KiB") != std::string::npos) {
    info.vramBytes = static_cast<size_t>(memValue * 1024.0);
  } else if (unit.find("MiB") != std::string::npos) {
    info.vramBytes = static_cast<size_t>(memValue * 1024.0 * 1024.0);
  } else if (unit.find("GiB") != std::string::npos) {
    info.vramBytes = static_cast<size_t>(memValue * 1024.0 * 1024.0 * 1024.0);
  } else {
    // Assume bytes.
    info.vramBytes = static_cast<size_t>(memValue);
  }

  return info;
}

// Check for the presence of a DLL in the system directory (Windows only).
// Avoids linking to the Windows SDK — uses plain filesystem access.
#if defined(_WIN32)
bool systemDllExists(const char* dllName) {
  std::error_code ec;
  // Check both System32 and the current directory.
  const auto sysRoot = std::getenv("SystemRoot");
  if (sysRoot != nullptr) {
    const auto path = std::filesystem::path(sysRoot) / "System32" / dllName;
    if (std::filesystem::exists(path, ec) && !ec) {
      return true;
    }
  }
  // Also try SysWOW64 for 32-bit processes on 64-bit Windows.
  if (sysRoot != nullptr) {
    const auto path = std::filesystem::path(sysRoot) / "SysWOW64" / dllName;
    if (std::filesystem::exists(path, ec) && !ec) {
      return true;
    }
  }
  return false;
}
#endif

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<GpuInfo> GpuCapabilityDetector::detectGpus() {
  std::vector<GpuInfo> gpus;

  // 1. NVIDIA GPUs via nvidia-smi.
  if (nvidiaSmiAvailable()) {
    // Run nvidia-smi and parse every line (one per GPU).
    std::string raw;
    if (captureCommandOutput(kNvidiaSmiQuery, raw) && !raw.empty()) {
      gpus.push_back(parseNvidiaSmiLine(raw));

      // Multi-GPU: nvidia-smi outputs one line per GPU.  We only get the
      // first line from captureCommandOutput; for a full enumeration a more
      // sophisticated approach would loop.  For most users a single GPU is
      // the common case.
    } else {
      // nvidia-smi exists but produced no output — add a generic entry.
      GpuInfo fallback;
      fallback.name = "NVIDIA GPU (unknown)";
      fallback.cuda = true;
      gpus.push_back(fallback);
    }
  }

  // 2. Platform-specific GPU entries.
#if defined(__APPLE__)
  {
    GpuInfo appleGpu;
#if defined(__arm64__)
    appleGpu.name = "Apple Silicon (M-series)";
    appleGpu.coreml = true;
    appleGpu.ane = true;
#else
    appleGpu.name = "Apple (Intel)";
    appleGpu.coreml = true;
#endif
    gpus.push_back(appleGpu);
  }
#endif

#if defined(_WIN32)
  {
    // Add a synthetic DirectML-capable entry if no NVIDIA GPU was found.
    const bool hasNvidia = std::any_of(gpus.begin(), gpus.end(),
                                       [](const GpuInfo& g) { return g.cuda; });
    if (!hasNvidia) {
      GpuInfo dmlGpu;
      dmlGpu.name = "Windows DirectML (WDDM)";
      dmlGpu.directml = true;
      gpus.push_back(dmlGpu);
    }
  }
#endif

  // 3. If nothing was detected, add a CPU entry.
  if (gpus.empty()) {
    GpuInfo cpuGpu;
    cpuGpu.name = "CPU (no GPU detected)";
    gpus.push_back(cpuGpu);
  }

  return gpus;
}

bool GpuCapabilityDetector::hasCudaSupport() {
  // CUDA is available when nvidia-smi works or the CUDA driver DLL is
  // present on disk.
  if (nvidiaSmiAvailable()) {
    std::string output;
    if (captureCommandOutput(kNvidiaSmiQuery, output) && !output.empty()) {
      return output.find("NVIDIA") != std::string::npos ||
             output.find("nvidia") != std::string::npos;
    }
  }

#if defined(_WIN32)
  // nvcuda.dll is installed by the NVIDIA driver even without the CUDA
  // Toolkit.  Check via plain filesystem access.
  return systemDllExists("nvcuda.dll");
#else
  // On Linux / macOS, if nvidia-smi is not available assume no CUDA.
  return false;
#endif
}

bool GpuCapabilityDetector::hasDirectMlSupport() {
#if defined(_WIN32)
  // DirectML ships as a system component on Windows 10 1903+ and is also
  // available as a redistributable.  Check for the DLL on disk.
  if (systemDllExists("DirectML.dll")) {
    return true;
  }
  // Fallback: on modern Windows (10+) DirectML is virtually always present.
  // Use std::getenv("windir") as a proxy for "we are on Windows".
  if (std::getenv("windir") != nullptr) {
    return true;
  }
  return false;
#else
  return false;
#endif
}

bool GpuCapabilityDetector::hasCoreMlSupport() {
#if defined(__APPLE__)
  // CoreML is available on macOS 10.13+ (High Sierra) and later.
  // At compile time we assume the deployment target is at least that.
  return true;
#else
  return false;
#endif
}

bool GpuCapabilityDetector::hasAneSupport() {
#if defined(__APPLE__) && defined(__arm64__)
  // Apple Neural Engine is available on M1 and later.
  return true;
#else
  return false;
#endif
}

std::string GpuCapabilityDetector::recommendProvider() {
  // Start with the platform's preferred provider.
  const auto platformPreferred = gpu::platformPreferredProvider();

  // Validate against runtime availability.
  const auto canon = gpu::canonicalProviderName(platformPreferred);

  if (canon == gpu::kProviderCuda && !hasCudaSupport()) {
    return gpu::kProviderCpu;
  }
  if (canon == gpu::kProviderDirectMl && !hasDirectMlSupport()) {
    return gpu::kProviderCpu;
  }
  if (canon == gpu::kProviderCoreMl && !hasCoreMlSupport()) {
    return gpu::kProviderCpu;
  }
  if (canon == gpu::kProviderAne && !hasAneSupport()) {
    // ANE unavailable — fall back to CoreML (which is less specialised but
    // available on all Macs).
    return gpu::kProviderCoreMl;
  }

  return platformPreferred;
}

} // namespace automix::ai
