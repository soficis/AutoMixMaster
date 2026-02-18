#pragma once

#include <cstddef>
#include <vector>

#include "dsp/TruePeakDetector.h"
#include "engine/AudioBuffer.h"

namespace automix::dsp {

struct LookaheadLimiterSettings {
  double ceilingDb = -1.0;
  double lookaheadMs = 7.0;
  double attackMs = 1.0;
  double releaseMs = 80.0;
  bool truePeakEnabled = true;
  int truePeakOversampleFactor = 4;
  bool softClipEnabled = false;
  double softClipDrive = 1.0;
};

class LookaheadLimiter {
 public:
  void prepare(double sampleRate, int channels, const LookaheadLimiterSettings& settings);
  void setSettings(const LookaheadLimiterSettings& settings);
  void reset();

  int latencySamples() const;
  void process(engine::AudioBuffer& buffer);

 private:
  float softClipSample(float input) const;

  double sampleRate_ = 44100.0;
  int channels_ = 0;
  int lookaheadSamples_ = 0;
  LookaheadLimiterSettings settings_;

  std::vector<std::vector<float>> delayLines_;
  std::vector<size_t> delayWriteIndex_;
  std::vector<float> detectorLine_;
  size_t detectorWriteIndex_ = 0;
  float smoothedGain_ = 1.0f;

  TruePeakDetector truePeakDetector_;
};

} // namespace automix::dsp
