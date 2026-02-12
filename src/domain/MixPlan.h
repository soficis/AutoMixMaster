#pragma once

#include <optional>
#include <string>
#include <vector>

namespace automix::domain {

struct StemMixDecision {
  std::string stemId;
  double gainDb = 0.0;
  double pan = 0.0;
  double highPassHz = 0.0;
  double mudCutDb = 0.0;
  bool enableCompressor = false;
  double compressorThresholdDb = -16.0;
  double compressorRatio = 3.0;
  double compressorReleaseMs = 80.0;
  bool enableExpander = false;
  double expanderThresholdDb = -45.0;
  double expanderRatio = 1.3;
};

struct MixPlan {
  double dryWet = 1.0;
  double mixBusHeadroomDb = 6.0;
  std::vector<StemMixDecision> stemDecisions;
  std::vector<std::string> decisionLog;
};

} // namespace automix::domain
