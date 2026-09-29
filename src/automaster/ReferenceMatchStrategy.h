#pragma once

#include <string>

#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automaster/IAutoMasterStrategy.h"
#include "engine/AudioBuffer.h"

namespace automix::automaster {

// Loudness/topology statistics measured from a reference mix. Every field is
// derived deterministically, so a matching target needs no learned model.
struct ReferenceProfile {
  double integratedLufs = -120.0;
  double shortTermLufs = -120.0;
  double truePeakDbtp = -1.0;
  double crestDb = 0.0;
  // 400 ms block loudness percentiles, the BS.1770-5 blocking interval.
  double p50BlockLufs = -120.0;
  double p95BlockLufs = -120.0;
};

ReferenceProfile measureReferenceProfile(const engine::AudioBuffer& reference);

// Optional, NON-DEFAULT mastering route that derives the loudness, ceiling and
// glue targets from a reference track instead of the preset alone.
//
// The route is per-session and disabled by default; buildPlan starts from the
// heuristic plan and overrides only the four reference-derived targets.
// applyPlan delegates to HeuristicAutoMasterStrategy unchanged, so the stage
// order of the default chain is untouched by construction.
class ReferenceMatchStrategy final : public IAutoMasterStrategy {
 public:
  explicit ReferenceMatchStrategy(ReferenceProfile profile);

  void setEnabled(bool enabled);
  bool isEnabled() const;

  domain::MasterPlan buildPlan(domain::MasterPreset preset, const engine::AudioBuffer& mixBuffer) const override;
  engine::AudioBuffer applyPlan(const engine::AudioBuffer& mixBuffer,
                                const domain::MasterPlan& plan,
                                MasteringReport* reportOut) const override;

 private:
  HeuristicAutoMasterStrategy heuristic_;
  ReferenceProfile profile_;
  bool enabled_ = false;
};

} // namespace automix::automaster
