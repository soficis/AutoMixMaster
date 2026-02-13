#include "ai/FeatureSchema.h"

#include "analysis/AnalysisResult.h"

namespace automix::ai {

const std::vector<std::string>& FeatureSchemaV1::names() {
  static const std::vector<std::string> kNames = {
      "rms_db",
      "peak_db",
      "crest_db",
      "dc_offset",
      "low_energy_ratio",
      "mid_energy_ratio",
      "high_energy_ratio",
      "sub_energy_ratio",
      "bass_energy_ratio",
      "low_mid_energy_ratio",
      "high_mid_energy_ratio",
      "presence_energy_ratio",
      "air_energy_ratio",
      "spectral_centroid_hz",
      "spectral_spread_hz",
      "spectral_flatness",
      "spectral_flux",
      "silence_ratio",
      "stereo_correlation",
      "stereo_width",
      "channel_balance_db",
      "artifact_risk",
      "artifact_swirl_risk",
      "artifact_smear_risk",
      "artifact_noise_dominance",
      "artifact_harmonicity",
      "artifact_phase_instability",
  };
  return kNames;
}

bool FeatureSchemaV1::isCompatible(const std::string& version) { return version == kVersion; }

size_t FeatureSchemaV1::featureCount() { return names().size(); }

std::vector<double> FeatureSchemaV1::extract(const analysis::AnalysisResult& metrics) {
  return {
      metrics.rmsDb,
      metrics.peakDb,
      metrics.crestDb,
      metrics.dcOffset,
      metrics.lowEnergy,
      metrics.midEnergy,
      metrics.highEnergy,
      metrics.subEnergy,
      metrics.bassEnergy,
      metrics.lowMidEnergy,
      metrics.highMidEnergy,
      metrics.presenceEnergy,
      metrics.airEnergy,
      metrics.spectralCentroidHz,
      metrics.spectralSpreadHz,
      metrics.spectralFlatness,
      metrics.spectralFlux,
      metrics.silenceRatio,
      metrics.stereoCorrelation,
      metrics.stereoWidth,
      metrics.channelBalanceDb,
      metrics.artifactRisk,
      metrics.artifactProfile.swirlRisk,
      metrics.artifactProfile.smearRisk,
      metrics.artifactProfile.noiseDominance,
      metrics.artifactProfile.harmonicity,
      metrics.artifactProfile.phaseInstability,
  };
}

} // namespace automix::ai
