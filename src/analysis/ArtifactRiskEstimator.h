#pragma once

#include "analysis/AnalysisResult.h"
#include "engine/AudioBuffer.h"

namespace automix::analysis {

class ArtifactRiskEstimator {
 public:
  ArtifactProfile profile(const engine::AudioBuffer& buffer, const AnalysisResult& metrics) const;
  double estimate(const engine::AudioBuffer& buffer, const AnalysisResult& metrics) const;
};

} // namespace automix::analysis
