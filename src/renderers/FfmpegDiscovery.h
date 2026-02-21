#pragma once

#include <filesystem>
#include <optional>

namespace automix::renderers {

struct FfmpegBinaryInfo {
  std::filesystem::path executablePath;
  std::filesystem::path installRoot;
};

class FfmpegDiscovery {
 public:
  std::optional<FfmpegBinaryInfo> find() const;
};

} // namespace automix::renderers
