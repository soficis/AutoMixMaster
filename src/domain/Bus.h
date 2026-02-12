#pragma once

#include <string>

namespace automix::domain {

enum class BusType { StemGroup, Mix };

struct Bus {
  std::string id;
  std::string name;
  BusType type = BusType::StemGroup;
  double gainDb = 0.0;
};

std::string toString(BusType type);
BusType busTypeFromString(const std::string& value);

} // namespace automix::domain
