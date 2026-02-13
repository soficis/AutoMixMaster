#pragma once

#include <string>
#include <vector>

namespace automix::domain {

enum class MasterPreset { DefaultStreaming, Broadcast, UdioOptimized, Custom };

struct MasterPlan {
  MasterPreset preset = MasterPreset::DefaultStreaming;
  std::string presetName = "DefaultStreaming";
  double targetLufs = -14.0;
  double truePeakDbtp = -1.0;
  double preGainDb = 0.0;
  bool applyEq = true;
  double glueThresholdDb = -18.0;
  double glueRatio = 2.0;
  double limiterCeilingDb = -1.0;
  bool limiterTruePeakEnabled = true;
  double limiterLookaheadMs = 7.0;
  double limiterAttackMs = 1.0;
  double limiterReleaseMs = 80.0;
  int ditherBitDepth = 24;
  bool enableDeEsser = false;
  double deEsserStrength = 0.35;
  bool enableDeHarshEq = false;
  double deHarshStrength = 0.30;
  bool enableLowMono = false;
  double lowMonoHz = 120.0;
  double stereoWidth = 1.0;
  bool enableSoftClipper = false;
  double softClipDrive = 1.15;
  std::vector<std::string> decisionLog;
};

std::string toString(MasterPreset preset);
MasterPreset masterPresetFromString(const std::string& value);

} // namespace automix::domain
