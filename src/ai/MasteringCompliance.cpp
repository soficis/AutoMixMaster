#include "ai/MasteringCompliance.h"

#include <algorithm>
#include <cmath>

namespace automix::ai {
namespace {

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

} // namespace

domain::MasterPlan MasteringCompliance::enforcePlanBounds(const domain::MasterPlan& plan) const {
  domain::MasterPlan output = plan;
  output.targetLufs = std::clamp(output.targetLufs, -30.0, -8.0);
  output.truePeakDbtp = std::clamp(output.truePeakDbtp, -3.0, -0.1);
  output.limiterCeilingDb = std::clamp(output.limiterCeilingDb, -3.0, -0.1);
  output.preGainDb = std::clamp(output.preGainDb, -9.0, 9.0);
  output.glueThresholdDb = std::clamp(output.glueThresholdDb, -36.0, -6.0);
  output.glueRatio = std::clamp(output.glueRatio, 1.1, 6.0);
  return output;
}

engine::AudioBuffer MasteringCompliance::enforceOutput(const engine::AudioBuffer& masteredBuffer,
                                                       const domain::MasterPlan& plan,
                                                       const automaster::HeuristicAutoMasterStrategy& strategy,
                                                       automaster::MasteringReport* reportOut) const {
  engine::AudioBuffer corrected = masteredBuffer;

  double lufs = strategy.measureIntegratedLufs(corrected);
  double truePeak = strategy.estimateTruePeakDbtp(corrected, 4);

  if (truePeak > plan.truePeakDbtp) {
    corrected.applyGain(static_cast<float>(dbToLinear(plan.truePeakDbtp - truePeak)));
    truePeak = strategy.estimateTruePeakDbtp(corrected, 4);
  }

  const double loudnessError = plan.targetLufs - lufs;
  if (std::abs(loudnessError) > 0.7) {
    corrected.applyGain(static_cast<float>(dbToLinear(loudnessError)));
    const double adjustedPeak = strategy.estimateTruePeakDbtp(corrected, 4);
    if (adjustedPeak > plan.truePeakDbtp) {
      corrected.applyGain(static_cast<float>(dbToLinear(plan.truePeakDbtp - adjustedPeak)));
    }
  }

  if (reportOut != nullptr) {
    reportOut->integratedLufs = strategy.measureIntegratedLufs(corrected);
    reportOut->truePeakDbtp = strategy.estimateTruePeakDbtp(corrected, 4);
  }

  return corrected;
}

} // namespace automix::ai
