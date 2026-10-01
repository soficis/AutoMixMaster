#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
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

// Loads the installed libraries by full path so the CUDA execution provider's
// later lookups by name resolve to them. Idempotent; true once loaded. Returns
// false (and changes nothing) when no installed pack is found.
bool preload(const std::filesystem::path& root = defaultRoot());

// Whether to offer the pack: an NVIDIA adapter with at least
// `minimumDedicatedBytes` of dedicated memory is present (read through DXGI,
// so no CUDA is needed to decide), and the pack is not installed yet.
bool shouldOffer(std::uint64_t minimumDedicatedBytes, const std::filesystem::path& root = defaultRoot());

// Largest dedicated memory among NVIDIA adapters, if any (Windows only).
std::optional<std::uint64_t> largestNvidiaAdapterBytes();

} // namespace GpuRuntimePack

} // namespace automix::ai
