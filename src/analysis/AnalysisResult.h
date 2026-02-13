#pragma once

#include <string>

#include "analysis/ArtifactProfile.h"

namespace automix::analysis {

struct AnalysisResult {
  double peakDb = -120.0;
  double rmsDb = -120.0;
  double crestDb = 0.0;
  double dcOffset = 0.0;
  double lowEnergy = 0.0;
  double midEnergy = 0.0;
  double highEnergy = 0.0;
  double subEnergy = 0.0;
  double bassEnergy = 0.0;
  double lowMidEnergy = 0.0;
  double highMidEnergy = 0.0;
  double presenceEnergy = 0.0;
  double airEnergy = 0.0;
  double spectralCentroidHz = 0.0;
  double spectralSpreadHz = 0.0;
  double spectralFlatness = 0.0;
  double spectralFlux = 0.0;
  double silenceRatio = 0.0;
  double stereoCorrelation = 1.0;
  double stereoWidth = 0.0;
  double channelBalanceDb = 0.0;
  double artifactRisk = 0.0;
  ArtifactProfile artifactProfile;
};

struct StemAnalysisEntry {
  std::string stemId;
  std::string stemName;
  AnalysisResult metrics;
};

} // namespace automix::analysis
