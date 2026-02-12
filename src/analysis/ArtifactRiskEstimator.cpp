#include "analysis/ArtifactRiskEstimator.h"

#include <algorithm>
#include <cmath>

namespace automix::analysis {

double ArtifactRiskEstimator::estimate(const engine::AudioBuffer& buffer, const AnalysisResult& metrics) const {
  if (buffer.getNumSamples() < 2 || buffer.getNumChannels() == 0) {
    return 0.0;
  }

  double roughness = 0.0;
  double magnitude = 0.0;
  int observations = 0;

  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    float previous = buffer.getSample(ch, 0);
    for (int i = 1; i < buffer.getNumSamples(); ++i) {
      const float current = buffer.getSample(ch, i);
      roughness += std::abs(current - previous);
      magnitude += std::abs(current);
      previous = current;
      ++observations;
    }
  }

  const double averageRoughness = roughness / std::max(1, observations);
  const double averageMagnitude = magnitude / std::max(1, observations);
  const double normalizedRoughness = averageRoughness / std::max(1.0e-6, averageMagnitude);

  double risk = 0.0;
  risk += std::clamp(metrics.highEnergy, 0.0, 1.0) * 0.35;
  risk += std::clamp(normalizedRoughness / 1.5, 0.0, 1.0) * 0.50;
  risk += std::clamp(metrics.stereoWidth, 0.0, 1.0) * 0.10;
  risk += std::clamp(metrics.silenceRatio, 0.0, 1.0) * 0.05;

  return std::clamp(risk, 0.0, 1.0);
}

} // namespace automix::analysis
