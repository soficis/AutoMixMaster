#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace automix::renderers {

struct BundledToolBinaryInfo {
  std::filesystem::path executablePath;
  std::filesystem::path installRoot;
};

struct BundledToolDiscoverySpec {
  std::string environmentVariable;
  std::vector<std::string> executableNames;
  std::vector<std::string> assetDirectoryNames;
};

class BundledToolDiscovery {
 public:
  static std::optional<BundledToolBinaryInfo> find(const BundledToolDiscoverySpec& spec);
};

} // namespace automix::renderers
