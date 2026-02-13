#pragma once

#include <string>
#include <vector>

namespace automix::platform {

enum class ThirdPartyLinkMode {
  InProcess,
  External,
};

struct ThirdPartyComponent {
  std::string name;
  std::string version;
  std::string licenseId;
  ThirdPartyLinkMode linkMode = ThirdPartyLinkMode::InProcess;
  bool shipsByDefault = false;
  std::string notes;
};

class ThirdPartyComponentRegistry {
 public:
  // Keep this list explicit so every integration documents legal/packaging intent.
  static std::vector<ThirdPartyComponent> all();
};

std::string toString(ThirdPartyLinkMode linkMode);

} // namespace automix::platform
