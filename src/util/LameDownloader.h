#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace automix::util {

class LameDownloader {
 public:
  struct DownloadResult {
    bool success = false;
    bool attempted = false;
    std::filesystem::path executablePath;
    std::string detail;
  };

  struct PinnedSource {
    std::string platformKey;
    std::string url;
    std::string sha256;
  };

  static std::filesystem::path cacheBinaryPath();
  /// True when the cached binary exists and is the one a verified install left there.
  static bool cachedBinaryIsVerified();
  /// Every built-in download with the SHA-256 it must match, for all platforms.
  static std::vector<PinnedSource> pinnedSources();
  /// Checks a downloaded file against its pinned SHA-256; a mismatch deletes the file.
  static bool verifyDownload(const std::filesystem::path& file, const std::string& expectedSha256, std::string* detail);
  static bool isSupportedOnCurrentPlatform();
  static DownloadResult ensureAvailable(bool forceDownload = false);
};

} // namespace automix::util
