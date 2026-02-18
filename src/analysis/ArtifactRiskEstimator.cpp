#include "analysis/ArtifactRiskEstimator.h"

#include <algorithm>
#include <cmath>

namespace automix::analysis {
namespace {

double clamp01(const double value) { return std::clamp(value, 0.0, 1.0); }

}

ArtifactProfile ArtifactRiskEstimator::profile(const engine::AudioBuffer& buffer, const AnalysisResult& metrics) const {
  ArtifactProfile profile;
  if (buffer.getNumSamples() < 2 || buffer.getNumChannels() == 0) {
    return profile;
  }

  double roughness = 0.0;
  double magnitude = 0.0;
  double fluxInstability = 0.0;
  int observations = 0;

  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    float previous = buffer.getSample(ch, 0);
    float previousDelta = 0.0f;
    for (int i = 1; i < buffer.getNumSamples(); ++i) {
      const float current = buffer.getSample(ch, i);
      const float delta = current - previous;
      roughness += std::abs(delta);
      fluxInstability += std::abs(delta - previousDelta);
      magnitude += std::abs(current);
      previous = current;
      previousDelta = delta;
      ++observations;
    }
  }

  const double averageRoughness = roughness / std::max(1, observations);
  const double averageFluxInstability = fluxInstability / std::max(1, observations);
  const double averageMagnitude = magnitude / std::max(1, observations);
  const double normalizedRoughness = averageRoughness / std::max(1.0e-6, averageMagnitude);
  const double normalizedFluxInstability = averageFluxInstability / std::max(1.0e-6, averageMagnitude);
  const double spectralFluxNorm = clamp01(metrics.spectralFlux * 0.25);
  const double spectralFlatness = clamp01(metrics.spectralFlatness);
  const double centroidRisk = clamp01((metrics.spectralCentroidHz - 4500.0) / 4500.0);
  const double phaseInstability = 1.0 - std::abs(std::clamp(metrics.stereoCorrelation, -1.0, 1.0));

  // Weights for the noiseDominance estimate are normalized to sum to 1.0 so that the
  // resulting score remains a convex combination of the contributing metrics.
  const double noiseWeightHighEnergy            = 0.30;
  const double noiseWeightSpectralFlatness      = 0.35;
  const double noiseWeightNormalizedRoughness   = 0.15;
  const double noiseWeightFluxInstability       = 0.10;
  const double noiseWeightSpectralFluxNorm      = 0.10;

  const double noiseDominance =
      clamp01(metrics.highEnergy * noiseWeightHighEnergy +
              spectralFlatness * noiseWeightSpectralFlatness +
              normalizedRoughness * noiseWeightNormalizedRoughness +
              normalizedFluxInstability * noiseWeightFluxInstability +
              spectralFluxNorm * noiseWeightSpectralFluxNorm);
  const double harmonicity = clamp01(metrics.lowEnergy * 0.30 + metrics.midEnergy * 0.30 +
                                     (1.0 - spectralFlatness) * 0.30 - centroidRisk * 0.10);

  profile.phaseInstability = clamp01(phaseInstability);
  profile.noiseDominance = noiseDominance;
  profile.harmonicity = harmonicity;
  profile.swirlRisk = clamp01(metrics.highEnergy * 0.25 + profile.phaseInstability * 0.25 +
                              normalizedFluxInstability * 0.20 + spectralFluxNorm * 0.20 + centroidRisk * 0.10);
  profile.smearRisk = clamp01(normalizedRoughness * 0.35 + normalizedFluxInstability * 0.20 +
                              spectralFlatness * 0.20 + spectralFluxNorm * 0.15 + clamp01(metrics.silenceRatio) * 0.10);
  return profile;
}

double ArtifactRiskEstimator::estimate(const engine::AudioBuffer& buffer, const AnalysisResult& metrics) const {
  const auto p = profile(buffer, metrics);
  const double risk = p.swirlRisk * 0.35 + p.smearRisk * 0.35 + p.phaseInstability * 0.15 + p.noiseDominance * 0.15;
  return clamp01(risk);
}

} // namespace automix::analysis
