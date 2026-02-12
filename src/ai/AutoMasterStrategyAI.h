#pragma once

#include "ai/IModelInference.h"
#include "ai/MasteringCompliance.h"
#include "analysis/AnalysisResult.h"
#include "domain/MasterPlan.h"

namespace automix::ai {

class AutoMasterStrategyAI {
 public:
  domain::MasterPlan buildPlan(const analysis::AnalysisResult& mixBusMetrics,
                               const domain::MasterPlan& heuristicPlan,
                               IModelInference* inference) const;

  engine::AudioBuffer applyPlan(const engine::AudioBuffer& mixBuffer,
                                const domain::MasterPlan& plan,
                                const automaster::HeuristicAutoMasterStrategy& strategy,
                                automaster::MasteringReport* reportOut) const;

 private:
  MasteringCompliance compliance_;
};

} // namespace automix::ai
