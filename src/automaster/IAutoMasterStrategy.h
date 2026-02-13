#pragma once

#include <vector>

#include "domain/MasterPlan.h"
#include "engine/AudioBuffer.h"

namespace automix::automaster {

struct MasteringReport {
  double integratedLufs = -120.0;
  double shortTermLufs = -120.0;
  double loudnessRange = 0.0;
  double samplePeakDbfs = -120.0;
  double truePeakDbtp = -120.0;
  double crestDb = 0.0;
  double monoCorrelation = 1.0;
  std::vector<std::string> activeModules;
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
