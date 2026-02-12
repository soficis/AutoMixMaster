#pragma once

#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "domain/MasterPlan.h"
#include "engine/AudioBuffer.h"

namespace automix::automaster {

class OriginalMixReference {
 public:
  domain::MasterPlan applySoftTarget(const domain::MasterPlan& basePlan,
                                     const engine::AudioBuffer& stemMix,
                                     const engine::AudioBuffer& originalMix,
                                     const HeuristicAutoMasterStrategy& strategy,
                                     const analysis::StemAnalyzer& analyzer) const;
};

} // namespace automix::automaster
