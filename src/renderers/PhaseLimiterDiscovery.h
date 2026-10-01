#pragma once

#include <filesystem>
#include <optional>
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

// A binary without its mastering reference cannot master; treat it as absent.
inline bool isCompleteInstall(const PhaseLimiterBinaryInfo& info) {
  std::error_code error;
  return std::filesystem::is_regular_file(masteringReferencePath(info), error) && !error;
}

class PhaseLimiterDiscovery {
 public:
  std::optional<PhaseLimiterBinaryInfo> find() const;
  std::optional<PhaseLimiterBinaryInfo> findInRoots(const std::vector<std::filesystem::path>& roots) const;
};

} // namespace automix::renderers
