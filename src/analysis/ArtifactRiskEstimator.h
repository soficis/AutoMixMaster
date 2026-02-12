#pragma once

#include "analysis/AnalysisResult.h"
#include "engine/AudioBuffer.h"

namespace automix::analysis {

class ArtifactRiskEstimator {
 public:
  double estimate(const engine::AudioBuffer& buffer, const AnalysisResult& metrics) const;
};

} // namespace automix::analysis
