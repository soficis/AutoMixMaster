#pragma once

#include <filesystem>
#include <optional>

namespace automix::renderers {

struct SoxBinaryInfo {
  std::filesystem::path executablePath;
  std::filesystem::path installRoot;
};

class SoxDiscovery {
 public:
  std::optional<SoxBinaryInfo> find() const;
};

} // namespace automix::renderers
