#include "ai/ModelStrategy.h"

#include <algorithm>

namespace automix::ai {

std::pair<domain::MixPlan, domain::MasterPlan> ModelStrategy::applyOverrides(
    IModelInference* inference,
    const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
    const domain::MixPlan& baseMixPlan,
    const domain::MasterPlan& baseMasterPlan) const {
  domain::MixPlan mixPlan = baseMixPlan;
  domain::MasterPlan masterPlan = baseMasterPlan;

  if (inference == nullptr || !inference->isAvailable()) {
    return {mixPlan, masterPlan};
  }

  std::vector<double> features;
  features.reserve(analysisEntries.size() * 4);
  for (const auto& entry : analysisEntries) {
    features.push_back(entry.metrics.rmsDb);
    features.push_back(entry.metrics.lowEnergy);
    features.push_back(entry.metrics.midEnergy);
    features.push_back(entry.metrics.highEnergy);
  }

  const InferenceRequest request{
      .task = "mix_master_override",
      .features = features,
  };
  const auto inferenceResult = inference->run(request);
  const auto& outputs = inferenceResult.outputs;

  if (outputs.contains("dryWet")) {
    mixPlan.dryWet = std::clamp(outputs.at("dryWet"), 0.0, 1.0);
    mixPlan.decisionLog.push_back("Model override: dryWet=" + std::to_string(mixPlan.dryWet));
  }

  if (outputs.contains("targetLufs")) {
    masterPlan.targetLufs = std::clamp(outputs.at("targetLufs"), -30.0, -6.0);
    masterPlan.decisionLog.push_back("Model override: targetLufs=" + std::to_string(masterPlan.targetLufs));
  }

  if (outputs.contains("preGainDb")) {
    masterPlan.preGainDb = std::clamp(outputs.at("preGainDb"), -9.0, 9.0);
    masterPlan.decisionLog.push_back("Model override: preGainDb=" + std::to_string(masterPlan.preGainDb));
  }

  return {mixPlan, masterPlan};
}

} // namespace automix::ai
