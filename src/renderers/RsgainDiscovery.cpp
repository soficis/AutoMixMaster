#include "renderers/RsgainDiscovery.h"

#include <string>

#include "renderers/BundledToolDiscovery.h"

namespace automix::renderers {

std::optional<RsgainBinaryInfo> RsgainDiscovery::find() const {
  const BundledToolDiscoverySpec spec{
      .environmentVariable = "RSGAIN_BIN",
#if defined(_WIN32)
      .executableNames = {"rsgain.exe", "rsgain"},
#else
      .executableNames = {"rsgain", "rsgain.exe"},
#endif
      .assetDirectoryNames = {"rsgain", "RSGain"},
  };

  if (const auto binary = BundledToolDiscovery::find(spec); binary.has_value()) {
    return RsgainBinaryInfo{
        .executablePath = binary->executablePath,
        .installRoot = binary->installRoot,
    };
  }
  return std::nullopt;
}

} // namespace automix::renderers
