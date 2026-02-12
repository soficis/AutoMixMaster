#pragma once

#include <utility>
#include <vector>

#include "ai/IModelInference.h"
#include "analysis/AnalysisResult.h"
#include "domain/MasterPlan.h"
#include "domain/MixPlan.h"

namespace automix::ai {

class ModelStrategy {
 public:
  std::pair<domain::MixPlan, domain::MasterPlan> applyOverrides(
      IModelInference* inference,
      const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
      const domain::MixPlan& baseMixPlan,
      const domain::MasterPlan& baseMasterPlan) const;
};

} // namespace automix::ai
