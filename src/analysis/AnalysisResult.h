#pragma once

#include <string>

namespace automix::analysis {

struct AnalysisResult {
  double peakDb = -120.0;
  double rmsDb = -120.0;
  double crestDb = 0.0;
  double lowEnergy = 0.0;
  double midEnergy = 0.0;
  double highEnergy = 0.0;
  double silenceRatio = 0.0;
  double stereoCorrelation = 1.0;
  double stereoWidth = 0.0;
  double artifactRisk = 0.0;
};

struct StemAnalysisEntry {
  std::string stemId;
  std::string stemName;
  AnalysisResult metrics;
};

} // namespace automix::analysis
