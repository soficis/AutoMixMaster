#pragma once

#include <string>

namespace automix::domain {

enum class StemOrigin {
  Recorded,
  Separated,
};

std::string toString(StemOrigin origin);
StemOrigin stemOriginFromString(const std::string& value);

} // namespace automix::domain
