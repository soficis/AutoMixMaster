#pragma once

#include <filesystem>
#include <optional>

namespace automix::renderers {

struct RsgainBinaryInfo {
  std::filesystem::path executablePath;
  std::filesystem::path installRoot;
};

class RsgainDiscovery {
 public:
  std::optional<RsgainBinaryInfo> find() const;
};

} // namespace automix::renderers
