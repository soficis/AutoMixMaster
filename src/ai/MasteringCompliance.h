#pragma once

#include "automaster/HeuristicAutoMasterStrategy.h"
#include "domain/MasterPlan.h"
#include "engine/AudioBuffer.h"

namespace automix::ai {

class MasteringCompliance {
 public:
  domain::MasterPlan enforcePlanBounds(const domain::MasterPlan& plan) const;
  engine::AudioBuffer enforceOutput(const engine::AudioBuffer& masteredBuffer,
                                    const domain::MasterPlan& plan,
                                    const automaster::HeuristicAutoMasterStrategy& strategy,
                                    automaster::MasteringReport* reportOut) const;
};

} // namespace automix::ai
