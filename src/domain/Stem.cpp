#include "domain/StemRole.h"

#include <algorithm>
#include <cctype>

namespace automix::domain {

std::string toString(const StemRole role) {
  switch (role) {
    case StemRole::Unknown:
      return "unknown";
    case StemRole::Vocals:
      return "vocals";
    case StemRole::Bass:
      return "bass";
    case StemRole::Kick:
      return "kick";
    case StemRole::Snare:
      return "snare";
    case StemRole::Drums:
      return "drums";
    case StemRole::Guitar:
      return "guitar";
    case StemRole::Keys:
      return "keys";
    case StemRole::Fx:
      return "fx";
    case StemRole::Music:
      return "music";
  }
  return "unknown";
}

StemRole stemRoleFromString(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (normalized == "vocals") {
    return StemRole::Vocals;
  }
  if (normalized == "bass") {
    return StemRole::Bass;
  }
  if (normalized == "kick") {
    return StemRole::Kick;
  }
  if (normalized == "snare") {
    return StemRole::Snare;
  }
  if (normalized == "drums") {
    return StemRole::Drums;
  }
  if (normalized == "guitar") {
    return StemRole::Guitar;
  }
  if (normalized == "keys") {
    return StemRole::Keys;
  }
  if (normalized == "fx") {
    return StemRole::Fx;
  }
  if (normalized == "music") {
    return StemRole::Music;
  }
  return StemRole::Unknown;
}

} // namespace automix::domain
