#pragma once

#include <vector>

#include "ai/IModelInference.h"
#include "analysis/AnalysisResult.h"
#include "domain/MixPlan.h"
#include "domain/Session.h"

namespace automix::ai {

class AutoMixStrategyAI {
 public:
  domain::MixPlan buildPlan(const domain::Session& session,
                            const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
                            const domain::MixPlan& heuristicPlan,
                            IModelInference* inference) const;
};

} // namespace automix::ai
