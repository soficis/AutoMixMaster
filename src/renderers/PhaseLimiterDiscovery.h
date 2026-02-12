#pragma once

#include <filesystem>
#include <optional>
#include <vector>

namespace automix::renderers {

struct PhaseLimiterBinaryInfo {
  std::filesystem::path executablePath;
  std::filesystem::path installRoot;
};

class PhaseLimiterDiscovery {
 public:
  std::optional<PhaseLimiterBinaryInfo> find() const;
  std::optional<PhaseLimiterBinaryInfo> findInRoots(const std::vector<std::filesystem::path>& roots) const;
};

} // namespace automix::renderers
