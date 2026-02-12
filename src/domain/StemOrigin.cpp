#include "domain/StemOrigin.h"

#include <algorithm>
#include <cctype>

namespace automix::domain {

std::string toString(const StemOrigin origin) {
  switch (origin) {
    case StemOrigin::Recorded:
      return "recorded";
    case StemOrigin::Separated:
      return "separated";
  }
  return "recorded";
}

StemOrigin stemOriginFromString(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (normalized == "separated") {
    return StemOrigin::Separated;
  }
  return StemOrigin::Recorded;
}

} // namespace automix::domain
