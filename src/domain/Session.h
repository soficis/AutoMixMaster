#pragma once

#include <optional>
#include <string>
#include <vector>

#include "domain/Bus.h"
#include "domain/MasterPlan.h"
#include "domain/MixPlan.h"
#include "domain/RenderSettings.h"
#include "domain/Stem.h"

namespace automix::domain {

struct Session {
  int schemaVersion = 2;
  std::string sessionName;
  std::optional<std::string> originalMixPath;
  double residualBlend = 0.0;
  std::vector<Stem> stems;
  std::vector<Bus> buses;
  RenderSettings renderSettings;
  std::optional<MixPlan> mixPlan;
  std::optional<MasterPlan> masterPlan;
};

} // namespace automix::domain
