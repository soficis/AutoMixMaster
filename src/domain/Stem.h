#pragma once

#include <optional>
#include <string>

#include "domain/StemOrigin.h"
#include "domain/StemRole.h"

namespace automix::domain {

struct Stem {
  std::string id;
  std::string name;
  std::string filePath;
  StemRole role = StemRole::Unknown;
  StemOrigin origin = StemOrigin::Recorded;
  std::optional<std::string> busId;
  bool enabled = true;
};

} // namespace automix::domain
