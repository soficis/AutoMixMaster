#include "renderers/SoxDiscovery.h"

#include <string>

#include "renderers/BundledToolDiscovery.h"

namespace automix::renderers {

std::optional<SoxBinaryInfo> SoxDiscovery::find() const {
  const BundledToolDiscoverySpec spec{
      .environmentVariable = "SOX_BIN",
#if defined(_WIN32)
      .executableNames = {"sox.exe", "sox"},
#else
      .executableNames = {"sox", "sox.exe"},
#endif
      .assetDirectoryNames = {"sox", "SoX"},
  };

  if (const auto binary = BundledToolDiscovery::find(spec); binary.has_value()) {
    return SoxBinaryInfo{
        .executablePath = binary->executablePath,
        .installRoot = binary->installRoot,
    };
  }
  return std::nullopt;
}

} // namespace automix::renderers
