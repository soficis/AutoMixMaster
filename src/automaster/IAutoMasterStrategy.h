#pragma once

#include "domain/MasterPlan.h"
#include "engine/AudioBuffer.h"

namespace automix::automaster {

struct MasteringReport {
  double integratedLufs = -120.0;
  double truePeakDbtp = -120.0;
};

class IAutoMasterStrategy {
 public:
  virtual ~IAutoMasterStrategy() = default;

  virtual domain::MasterPlan buildPlan(domain::MasterPreset preset, const engine::AudioBuffer& mixBuffer) const = 0;
  virtual engine::AudioBuffer applyPlan(const engine::AudioBuffer& mixBuffer,
                                        const domain::MasterPlan& plan,
                                        MasteringReport* reportOut) const = 0;
};

} // namespace automix::automaster
