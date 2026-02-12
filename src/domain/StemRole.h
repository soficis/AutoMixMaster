#pragma once

#include <string>

namespace automix::domain {

enum class StemRole {
  Unknown,
  Vocals,
  Bass,
  Kick,
  Snare,
  Drums,
  Guitar,
  Keys,
  Fx,
  Music,
};

std::string toString(StemRole role);
StemRole stemRoleFromString(const std::string& value);

} // namespace automix::domain
