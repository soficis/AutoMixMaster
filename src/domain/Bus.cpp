#include "domain/Bus.h"

namespace automix::domain {

std::string toString(const BusType type) {
  switch (type) {
    case BusType::StemGroup:
      return "stem_group";
    case BusType::Mix:
      return "mix";
  }
  return "stem_group";
}

BusType busTypeFromString(const std::string& value) {
  if (value == "mix") {
    return BusType::Mix;
  }
  return BusType::StemGroup;
}

} // namespace automix::domain
