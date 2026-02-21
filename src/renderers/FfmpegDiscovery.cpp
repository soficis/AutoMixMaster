#include "renderers/FfmpegDiscovery.h"

#include <string>

#include "renderers/BundledToolDiscovery.h"

namespace automix::renderers {

std::optional<FfmpegBinaryInfo> FfmpegDiscovery::find() const {
  const BundledToolDiscoverySpec spec{
      .environmentVariable = "FFMPEG_BIN",
#if defined(_WIN32)
      .executableNames = {"ffmpeg.exe", "ffmpeg"},
#else
      .executableNames = {"ffmpeg", "ffmpeg.exe"},
#endif
      .assetDirectoryNames = {"ffmpeg", "FFmpeg"},
  };

  if (const auto binary = BundledToolDiscovery::find(spec); binary.has_value()) {
    return FfmpegBinaryInfo{
        .executablePath = binary->executablePath,
        .installRoot = binary->installRoot,
    };
  }
  return std::nullopt;
}

} // namespace automix::renderers
