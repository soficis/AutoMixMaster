#include "ai/AutoMasterStrategyAI.h"

#include <algorithm>

namespace automix::ai {
namespace {

double blend(const double baseValue, const double aiValue, const double confidence) {
  return baseValue * (1.0 - confidence) + aiValue * confidence;
}

} // namespace

domain::MasterPlan AutoMasterStrategyAI::buildPlan(const analysis::AnalysisResult& mixBusMetrics,
                                                    const domain::MasterPlan& heuristicPlan,
                                                    IModelInference* inference) const {
  domain::MasterPlan plan = heuristicPlan;
  if (inference == nullptr || !inference->isAvailable()) {
    plan.decisionLog.push_back("AI master strategy skipped: no inference backend.");
    return compliance_.enforcePlanBounds(plan);
  }

  const InferenceRequest request{
      .task = "master_parameters",
      .features = {mixBusMetrics.rmsDb,
                   mixBusMetrics.crestDb,
                   mixBusMetrics.lowEnergy,
                   mixBusMetrics.midEnergy,
                   mixBusMetrics.highEnergy},
  };
  const InferenceResult result = inference->run(request);
  if (!result.usedModel) {
    plan.decisionLog.push_back("AI master strategy fallback: " + result.logMessage);
    return compliance_.enforcePlanBounds(plan);
  }

  const double confidence = std::clamp(result.outputs.contains("confidence") ? result.outputs.at("confidence") : 0.5, 0.0, 1.0);
  if (result.outputs.contains("target_lufs")) {
    plan.targetLufs = blend(plan.targetLufs, result.outputs.at("target_lufs"), confidence);
  }
  if (result.outputs.contains("pre_gain_db")) {
    plan.preGainDb = blend(plan.preGainDb, result.outputs.at("pre_gain_db"), confidence);
  }
  if (result.outputs.contains("limiter_ceiling_db")) {
    const double blendedCeiling = blend(plan.limiterCeilingDb, result.outputs.at("limiter_ceiling_db"), confidence);
    plan.limiterCeilingDb = blendedCeiling;
    plan.truePeakDbtp = blendedCeiling;
  }
  if (result.outputs.contains("glue_ratio")) {
    plan.glueRatio = blend(plan.glueRatio, result.outputs.at("glue_ratio"), confidence);
  }
  if (result.outputs.contains("glue_threshold_db")) {
    plan.glueThresholdDb = blend(plan.glueThresholdDb, result.outputs.at("glue_threshold_db"), confidence);
  }

  plan.decisionLog.push_back("AI master strategy blended decisions with confidence=" + std::to_string(confidence));
  return compliance_.enforcePlanBounds(plan);
}

engine::AudioBuffer AutoMasterStrategyAI::applyPlan(const engine::AudioBuffer& mixBuffer,
                                                    const domain::MasterPlan& plan,
                                                    const automaster::HeuristicAutoMasterStrategy& strategy,
                                                    automaster::MasteringReport* reportOut) const {
  const auto boundedPlan = compliance_.enforcePlanBounds(plan);
  auto tempReport = automaster::MasteringReport{};
  auto mastered = strategy.applyPlan(mixBuffer, boundedPlan, &tempReport);
  return compliance_.enforceOutput(mastered, boundedPlan, strategy, reportOut);
}

} // namespace automix::ai
