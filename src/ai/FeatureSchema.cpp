#include "ai/FeatureSchema.h"

#include "analysis/AnalysisResult.h"

namespace automix::ai {
namespace {

double vectorValueOrDefault(const std::vector<double>& values, const size_t index) {
  return index < values.size() ? values[index] : 0.0;
}

struct SemanticVersion {
  int major = 0;
  int minor = 0;
  int patch = 0;
  bool valid = false;
};

SemanticVersion parseVersion(const std::string& versionStr) {
  SemanticVersion version;
  
  if (versionStr.empty()) {
    return version;
  }
  
  size_t pos = 0;
  size_t dotPos = versionStr.find('.');
  
  try {
    // Parse major version
    if (dotPos == std::string::npos) {
      version.major = std::stoi(versionStr);
      version.valid = true;
      return version;
    }
    
    version.major = std::stoi(versionStr.substr(pos, dotPos - pos));
    pos = dotPos + 1;
    
    // Parse minor version
    dotPos = versionStr.find('.', pos);
    if (dotPos == std::string::npos) {
      version.minor = std::stoi(versionStr.substr(pos));
      version.valid = true;
      return version;
    }
    
    version.minor = std::stoi(versionStr.substr(pos, dotPos - pos));
    pos = dotPos + 1;
    
    // Parse patch version
    version.patch = std::stoi(versionStr.substr(pos));
    version.valid = true;
  } catch (...) {
    version.valid = false;
  }
  
  return version;
}

} // namespace

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
      "crest_factor",
      "onset_strength",
      "mfcc_0",
      "mfcc_1",
      "mfcc_2",
      "mfcc_3",
      "mfcc_4",
      "mfcc_5",
      "mfcc_6",
      "mfcc_7",
      "mfcc_8",
      "mfcc_9",
      "mfcc_10",
      "mfcc_11",
      "mfcc_12",
      "cqt_0",
      "cqt_1",
      "cqt_2",
      "cqt_3",
      "cqt_4",
      "cqt_5",
      "cqt_6",
      "cqt_7",
      "cqt_8",
      "cqt_9",
      "cqt_10",
      "cqt_11",
      "cqt_12",
      "cqt_13",
      "cqt_14",
      "cqt_15",
      "cqt_16",
      "cqt_17",
      "cqt_18",
      "cqt_19",
      "cqt_20",
      "cqt_21",
      "cqt_22",
      "cqt_23",
  };
  return kNames;
}

bool FeatureSchemaV1::isCompatible(const std::string& version) {
  const auto current = parseVersion(kVersion);
  const auto provided = parseVersion(version);
  
  // Both versions must be valid
  if (!current.valid || !provided.valid) {
    return false;
  }
  
  // Major version must match exactly (breaking changes)
  if (provided.major != current.major) {
    return false;
  }
  
  // Minor version must be >= current (backward compatible)
  if (provided.minor < current.minor) {
    return false;
  }
  
  // If minor version is greater, patch doesn't matter (newer compatible version)
  if (provided.minor > current.minor) {
    return true;
  }
  
  // Minor versions match, so patch must be >= current
  return provided.patch >= current.patch;
}

size_t FeatureSchemaV1::featureCount() { return names().size(); }

std::vector<double> FeatureSchemaV1::extract(const analysis::AnalysisResult& metrics) {
  std::vector<double> values = {
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
      metrics.crestFactor,
      metrics.onsetStrength,
  };

  for (size_t i = 0; i < 13; ++i) {
    values.push_back(vectorValueOrDefault(metrics.mfccCoefficients, i));
  }
  for (size_t i = 0; i < 24; ++i) {
    values.push_back(vectorValueOrDefault(metrics.constantQBins, i));
  }

  return values;
}

} // namespace automix::ai
