#include "automaster/OriginalMixReference.h"

#include <algorithm>

namespace automix::automaster {

domain::MasterPlan OriginalMixReference::applySoftTarget(const domain::MasterPlan& basePlan,
                                                          const engine::AudioBuffer& stemMix,
                                                          const engine::AudioBuffer& originalMix,
                                                          const HeuristicAutoMasterStrategy& strategy,
                                                          const analysis::StemAnalyzer& analyzer) const {
  domain::MasterPlan plan = basePlan;
  if (stemMix.getNumSamples() == 0 || originalMix.getNumSamples() == 0) {
    plan.decisionLog.push_back("Original mix reference skipped: empty buffers.");
    return plan;
  }
  if (stemMix.getSampleRate() != originalMix.getSampleRate()) {
    plan.decisionLog.push_back("Original mix reference skipped: sample-rate mismatch.");
    return plan;
  }

  const double currentMixLufs = strategy.measureIntegratedLufs(stemMix);
  const double referenceLufs = strategy.measureIntegratedLufs(originalMix);
  const auto currentSpectrum = analyzer.analyzeBuffer(stemMix);
  const auto referenceSpectrum = analyzer.analyzeBuffer(originalMix);

  const double lufsDelta = referenceLufs - currentMixLufs;
  plan.targetLufs = std::clamp(plan.targetLufs * 0.75 + referenceLufs * 0.25, -30.0, -8.0);
  plan.preGainDb = std::clamp(plan.preGainDb + lufsDelta * 0.35, -9.0, 9.0);

  const double referenceCrest = referenceSpectrum.crestDb;
  const double mixCrest = currentSpectrum.crestDb;
  plan.glueRatio = std::clamp(plan.glueRatio - (referenceCrest - mixCrest) * 0.08, 1.1, 6.0);

  const double referenceTilt = referenceSpectrum.highEnergy - referenceSpectrum.lowEnergy;
  const double mixTilt = currentSpectrum.highEnergy - currentSpectrum.lowEnergy;
  if (std::abs(referenceTilt - mixTilt) > 0.06) {
    plan.applyEq = true;
  }

  plan.decisionLog.push_back("Original mix soft target applied.");
  plan.decisionLog.push_back("Reference LUFS=" + std::to_string(referenceLufs) +
                             ", mix LUFS=" + std::to_string(currentMixLufs));
  plan.decisionLog.push_back("Adjusted target LUFS to " + std::to_string(plan.targetLufs));
  return plan;
}

} // namespace automix::automaster
