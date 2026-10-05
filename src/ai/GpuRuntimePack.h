#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <utility>
#include <string>
#include <vector>

namespace automix::ai {

// The NVIDIA CUDA libraries the CUDA execution provider needs (runtime,
// cuBLAS, cuFFT, cuDNN), installed on demand per user instead of shipping
// ~1.3 GB with every copy of the application. The files come from NVIDIA's own
// redistributable wheels on PyPI, pinned by URL and SHA-256; only their DLLs
// are kept. Nothing is installed system-wide and no PATH is changed: the DLLs
// are preloaded by full path before any CUDA session is created.
namespace GpuRuntimePack {

struct Archive {
  std::string name;     // wheel file name
  std::string url;      // files.pythonhosted.org, pinned
  std::string sha256;   // lower-case hex
  std::uint64_t bytes;  // download size
};

// Pack identity, written into the completion marker; bump it whenever the
// archive list changes so an older pack is reinstalled rather than trusted.
std::string version();
const std::vector<Archive>& archives();
std::uint64_t downloadBytes();

// %LOCALAPPDATA%\AutoMixMaster\gpu-runtime\<version> on Windows; the user
// application-data directory elsewhere.
std::filesystem::path defaultRoot();

// The marker matches this version and every library it lists is present.
bool isInstalled(const std::filesystem::path& root);

// Fetches `url` into `destination`. `progress(bytesSoFar)` returns false to
// abort. Returns an empty string on success, else the reason.
using Fetcher = std::function<std::string(const std::string& url,
                                          const std::filesystem::path& destination,
                                          const std::function<bool(std::uint64_t)>& progress)>;
Fetcher httpFetcher();

struct InstallResult {
  bool success = false;
  bool cancelled = false;
  std::string message;
};

// Downloads, verifies and unpacks every archive into `root`. `progress` gets
// (bytes done, bytes total) and returns false to cancel. The completion marker
// is written last, so an interrupted or failed install is never mistaken for a
// working one; archives are deleted once unpacked.
InstallResult install(const std::filesystem::path& root,
                      const std::function<bool(std::uint64_t, std::uint64_t)>& progress,
                      const Fetcher& fetch = httpFetcher());

// Copies the .dll entries of a wheel (a zip) into `binDirectory`, flattening
// their paths; anything else is ignored. An entry whose name would escape the
// archive is an error. Exposed for tests.
std::string extractLibraries(const std::filesystem::path& wheel,
                             const std::filesystem::path& binDirectory,
                             std::vector<std::string>& extracted);

// Libraries in the wheels that ONNX Runtime never loads (an FFTW-compatible
// shim and a CPU-BLAS interposer); skipped when unpacking. Everything else is
// kept: one model loading only cuBLAS and three cuDNN parts does not make the
// rest unused (convolution, FFT and RNN graphs load others).
const std::vector<std::string>& unusedLibraries();

struct RemoveResult {
  bool removedNow = false;  // false: finishes at next start (files were in use)
  std::string message;
};

// Removes an installed pack. Libraries this process has loaded cannot be
// deleted on Windows, so the marker goes first (nothing preloads it again) and
// whatever remains is deleted by completePendingRemoval() at the next start.
// Refuses any directory that does not look like a GPU runtime pack.
RemoveResult uninstall(const std::filesystem::path& root = defaultRoot());

// Call once at startup, before anything can preload the pack.
void completePendingRemoval(const std::filesystem::path& root = defaultRoot());

// Loads the installed libraries by full path so the CUDA execution provider's
// later lookups by name resolve to them. Idempotent; true once loaded. Returns
// false (and changes nothing) when no installed pack is found.
bool preload(const std::filesystem::path& root = defaultRoot());

// Whether to offer the pack: an NVIDIA adapter with at least
// `minimumDedicatedBytes` of dedicated memory and a CUDA 13 capable driver
// (>= kMinimumDriverMajor) is present - both read through DXGI, so no CUDA is
// needed to decide - and the pack is not installed yet.
bool shouldOffer(std::uint64_t minimumDedicatedBytes, const std::filesystem::path& root = defaultRoot());

// CUDA 13 requires an NVIDIA driver of this major version or newer.
inline constexpr int kMinimumDriverMajor = 580;

struct NvidiaAdapter {
  std::uint64_t dedicatedBytes = 0;
  // NVIDIA's own numbering (e.g. 610.74); empty when Windows did not report it.
  std::optional<int> driverMajor;
  std::optional<int> driverMinor;
};

// NVIDIA driver version from the Windows user-mode driver version DXGI
// reports (a.b.c.d). NVIDIA's number is the last five digits of c and d:
// 32.0.16.1074 -> 610.74.
std::pair<int, int> nvidiaDriverVersion(std::uint64_t userModeDriverVersion);

// The NVIDIA adapter with the most dedicated memory, if any (Windows only).
std::optional<NvidiaAdapter> largestNvidiaAdapter();

} // namespace GpuRuntimePack

} // namespace automix::ai
