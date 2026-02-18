#pragma once

#include <string>
#include <vector>

namespace automix::domain {

enum class MasterPreset {
  DefaultStreaming,
  Broadcast,
  UdioOptimized,
  Spotify,
  AppleMusic,
  YouTube,
  AmazonMusic,
  Tidal,
  BroadcastEbuR128,
  Custom,
};

struct MultibandBandSettings {
  bool enabled = true;
  double thresholdDb = -18.0;
  double ratio = 2.0;
  double makeupGainDb = 0.0;
  double width = 1.0;
};

struct MultibandSettings {
  std::vector<double> crossoverHz = {120.0, 500.0, 2000.0, 8000.0};
  std::vector<MultibandBandSettings> bands = {
      MultibandBandSettings{},
      MultibandBandSettings{},
      MultibandBandSettings{},
      MultibandBandSettings{},
      MultibandBandSettings{},
  };
  bool linearPhase = false;
};

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
  bool enableMultibandCompressor = false;
  MultibandSettings multibandSettings;
  std::vector<std::string> decisionLog;
};

std::string toString(MasterPreset preset);
MasterPreset masterPresetFromString(const std::string& value);

} // namespace automix::domain
