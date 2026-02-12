#pragma once

#include "automaster/IAutoMasterStrategy.h"

namespace automix::automaster {

class HeuristicAutoMasterStrategy final : public IAutoMasterStrategy {
 public:
  domain::MasterPlan buildPlan(domain::MasterPreset preset, const engine::AudioBuffer& mixBuffer) const override;

  engine::AudioBuffer applyPlan(const engine::AudioBuffer& mixBuffer,
                                const domain::MasterPlan& plan,
                                MasteringReport* reportOut) const override;

  double measureIntegratedLufs(const engine::AudioBuffer& buffer) const;
  double estimateTruePeakDbtp(const engine::AudioBuffer& buffer, int oversampleFactor) const;
};

} // namespace automix::automaster
