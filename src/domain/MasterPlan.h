#pragma once

#include <string>
#include <vector>

namespace automix::domain {

enum class MasterPreset { DefaultStreaming, Broadcast, Custom };

struct MasterPlan {
  MasterPreset preset = MasterPreset::DefaultStreaming;
  double targetLufs = -14.0;
  double truePeakDbtp = -1.0;
  double preGainDb = 0.0;
  bool applyEq = true;
  double glueThresholdDb = -18.0;
  double glueRatio = 2.0;
  double limiterCeilingDb = -1.0;
  int ditherBitDepth = 24;
  std::vector<std::string> decisionLog;
};

std::string toString(MasterPreset preset);
MasterPreset masterPresetFromString(const std::string& value);

} // namespace automix::domain
