#pragma once

#include <filesystem>
#include <string>

namespace automix::util {

class LameDownloader {
 public:
  struct DownloadResult {
    bool success = false;
    bool attempted = false;
    std::filesystem::path executablePath;
    std::string detail;
  };

  static std::filesystem::path cacheBinaryPath();
  static bool isSupportedOnCurrentPlatform();
  static DownloadResult ensureAvailable(bool forceDownload = false);
};

} // namespace automix::util
