#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace automix::renderers {

struct PhaseLimiterBinaryInfo {
  std::filesystem::path executablePath;
  std::filesystem::path installRoot;
};

// The data files phase_limiter reads. Passed by absolute path: their built-in
// defaults ("./mastering_reference.json") resolve against the working
// directory and silently miss the copies under resource/, which made every
// run fail with "auto mastering error: syntax error".
inline std::filesystem::path masteringReferencePath(const PhaseLimiterBinaryInfo& info) {
  return info.installRoot / "resource" / "mastering_reference.json";
}
inline std::filesystem::path soundQualityCachePath(const PhaseLimiterBinaryInfo& info) {
  return info.installRoot / "resource" / "sound_quality2_cache";
}
inline std::filesystem::path licensesDirectoryPath(const PhaseLimiterBinaryInfo& info) {
  return info.installRoot / "licenses";
}

// A binary without its mastering reference cannot master; treat it as absent.
inline bool isCompleteInstall(const PhaseLimiterBinaryInfo& info) {
  std::error_code error;
  return std::filesystem::is_regular_file(masteringReferencePath(info), error) && !error;
}

struct PhaseLimiterDownloadPin {
  std::string platformKey;
  std::string url;
  std::string sha256;
};

std::vector<PhaseLimiterDownloadPin> phaseLimiterDownloadPinTable();
std::optional<PhaseLimiterDownloadPin> defaultPhaseLimiterDownloadPin();
std::string currentPhaseLimiterPlatformKey();
std::string phaseLimiterPlatformKeyForResolution(const std::string& platformKey);

using FfmpegResolver = std::function<std::optional<std::filesystem::path>()>;

std::optional<std::filesystem::path> ffmpegForPhaseLimiter();
void setFfmpegResolverForTesting(FfmpegResolver resolver);
void resetFfmpegResolverForTesting();

class PhaseLimiterDiscovery {
 public:
  using DownloadFetcher = std::function<bool(const std::string& url, const std::filesystem::path& destination)>;

  std::optional<PhaseLimiterBinaryInfo> find() const;
  std::optional<PhaseLimiterBinaryInfo> findInRoots(const std::vector<std::filesystem::path>& roots) const;

  static std::optional<PhaseLimiterBinaryInfo> downloadAndInstall(
      const std::optional<PhaseLimiterDownloadPin>& pin = std::nullopt);

  static void setDownloadFetcherForTesting(DownloadFetcher fetcher);
  static void resetDownloadFetcherForTesting();
  static void resetAttemptedDownloadForTesting();

  static void setFfmpegResolverForTesting(FfmpegResolver resolver) {
    ::automix::renderers::setFfmpegResolverForTesting(std::move(resolver));
  }
  static void resetFfmpegResolverForTesting() {
    ::automix::renderers::resetFfmpegResolverForTesting();
  }
};

} // namespace automix::renderers
