#pragma once

#include <vector>

#include "analysis/AnalysisResult.h"
#include "domain/MixPlan.h"
#include "domain/Session.h"

namespace automix::automix {

class IAutoMixStrategy {
 public:
  virtual ~IAutoMixStrategy() = default;
  virtual domain::MixPlan buildPlan(const domain::Session& session,
                                    const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
                                    double dryWet) const = 0;
};

} // namespace automix::automix
