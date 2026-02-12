#include "ai/AutoMixStrategyAI.h"

#include <algorithm>

namespace automix::ai {
namespace {

double blend(const double baseValue, const double aiValue, const double confidence) {
  return baseValue * (1.0 - confidence) + aiValue * confidence;
}

double gainClampForOrigin(const domain::StemOrigin origin) {
  return origin == domain::StemOrigin::Separated ? 12.0 : 18.0;
}

} // namespace

domain::MixPlan AutoMixStrategyAI::buildPlan(const domain::Session& session,
                                             const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
                                             const domain::MixPlan& heuristicPlan,
                                             IModelInference* inference) const {
  domain::MixPlan output = heuristicPlan;
  if (inference == nullptr || !inference->isAvailable()) {
    output.decisionLog.push_back("AI mix strategy skipped: no inference backend.");
    return output;
  }

  std::vector<double> features;
  features.reserve(analysisEntries.size() * 5);
  for (const auto& entry : analysisEntries) {
    features.push_back(entry.metrics.rmsDb);
    features.push_back(entry.metrics.lowEnergy);
    features.push_back(entry.metrics.midEnergy);
    features.push_back(entry.metrics.highEnergy);
    features.push_back(entry.metrics.artifactRisk);
  }

  const InferenceRequest request{
      .task = "mix_parameters",
      .features = features,
  };
  const InferenceResult result = inference->run(request);
  if (!result.usedModel) {
    output.decisionLog.push_back("AI mix strategy fallback: " + result.logMessage);
    return output;
  }

  const double confidence = std::clamp(result.outputs.contains("confidence") ? result.outputs.at("confidence") : 0.5, 0.0, 1.0);
  const double globalGainDelta = std::clamp(result.outputs.contains("global_gain_db") ? result.outputs.at("global_gain_db") : 0.0, -6.0, 6.0);
  const double globalPanBias = std::clamp(result.outputs.contains("global_pan_bias") ? result.outputs.at("global_pan_bias") : 0.0, -0.3, 0.3);

  for (size_t i = 0; i < output.stemDecisions.size(); ++i) {
    auto& decision = output.stemDecisions[i];

    double aiGain = decision.gainDb + globalGainDelta;
    const std::string stemGainKey = "stem" + std::to_string(i) + "_gain_db";
    if (result.outputs.contains(stemGainKey)) {
      aiGain = std::clamp(result.outputs.at(stemGainKey), -24.0, 24.0);
    }

    double aiPan = decision.pan + globalPanBias;
    const std::string stemPanKey = "stem" + std::to_string(i) + "_pan";
    if (result.outputs.contains(stemPanKey)) {
      aiPan = std::clamp(result.outputs.at(stemPanKey), -1.0, 1.0);
    }

    domain::StemOrigin origin = domain::StemOrigin::Recorded;
    for (const auto& stem : session.stems) {
      if (stem.id == decision.stemId) {
        origin = stem.origin;
        break;
      }
    }

    const double gainClamp = gainClampForOrigin(origin);
    decision.gainDb = std::clamp(blend(decision.gainDb, aiGain, confidence), -gainClamp, gainClamp);
    decision.pan = std::clamp(blend(decision.pan, aiPan, confidence), -1.0, 1.0);
    decision.highPassHz = std::clamp(decision.highPassHz, 0.0, 240.0);
  }

  output.decisionLog.push_back("AI mix strategy blended decisions with confidence=" + std::to_string(confidence));
  return output;
}

} // namespace automix::ai
