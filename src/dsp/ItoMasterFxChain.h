#pragma once

#include <array>
#include <vector>

#include "engine/AudioBuffer.h"

namespace automix::dsp {

// ─────────────────────────────────────────────────────────────────────────────
// ITO-Master white-box mastering FX chain.
//
// Native, deterministic DSP counterpart to the ITO-Master ONNX mastering route
// (src/ai/ItoMasterAdapter.* maps the 46 normalized ITO-Master parameters onto
// these settings). Every stage is exposed as a standalone, testable static
// entry point so unit tests can drive a single stage in isolation.
//
// Stage order matches the ITO-Master fx_order contract:
//   eq (18 params) → distortion (2) → multiband_comp (21) → gain (1)
//   → imager (1) → limiter (3)   [46 total]
//
// The EQ + crossover filtering are implemented with JUCE DSP biquads
// (juce_dsp is linked via automix_core); the dynamics stages use deterministic
// envelope followers so per-band behaviour is exact and unit-testable.
// ─────────────────────────────────────────────────────────────────────────────

struct ItoEqBandSettings {
  double gainDb = 0.0;
  double freqHz = 1000.0;
  double qFactor = 0.7071;
};

struct ItoEqSettings {
  ItoEqBandSettings lowShelf;
  std::array<ItoEqBandSettings, 4> bands{};
  ItoEqBandSettings highShelf;
  bool enabled = true;
};

struct ItoDistortionSettings {
  double driveDb = 0.0;         // 10^(drive/20) pre-gain before tanh
  double parallelWeight = 0.35; // dry*(1-w) + sat*w
  bool enabled = true;
};

struct ItoCompressorBandSettings {
  double compThresholdDb = -20.0;
  double compRatio = 2.0;
  double expThresholdDb = -60.0;
  double expRatio = 0.5;
  double attackMs = 20.0;
  double releaseMs = 80.0;
};

struct ItoMultibandCompSettings {
  double lowCrossoverHz = 120.0;  // Linkwitz-Riley crossover, low/mid
  double highCrossoverHz = 6000.0; // Linkwitz-Riley crossover, mid/high
  double parallelWeight = 0.35;   // dry*(1-w) + wet*w
  std::array<ItoCompressorBandSettings, 3> bands{}; // low, mid, high
  bool enabled = true;
};

struct ItoGainSettings {
  double gainDb = 0.0;
  bool enabled = true;
};

struct ItoImagerSettings {
  double width = 1.0; // M/S width; 1.0 = unchanged
  bool enabled = true;
};

struct ItoLimiterSettings {
  double thresholdDb = -1.0; // acts as the output ceiling
  double attackMs = 10.0;
  double releaseMs = 80.0;
  double lookaheadMs = 5.0;
  bool enabled = true;
};

struct ItoMasterChainSettings {
  ItoEqSettings eq;
  ItoDistortionSettings distortion;
  ItoMultibandCompSettings multiband;
  ItoGainSettings gain;
  ItoImagerSettings imager;
  ItoLimiterSettings limiter;
};

class ItoMasterFxChain {
 public:
  // ── Standalone stages (testable in isolation) ──────────────────────────────

  // 6-section biquad EQ: low shelf → band0..3 (peaking) → high shelf.
  static void processEq(engine::AudioBuffer& buffer, const ItoEqSettings& settings);

  // Tanh saturation with parallel dry/wet blend.
  static void processDistortion(engine::AudioBuffer& buffer, const ItoDistortionSettings& settings);

  // 3-band split via Linkwitz-Riley (LR4) crossovers, per-band compressor +
  // expander, parallel dry/wet blend.
  static void processMultiband(engine::AudioBuffer& buffer,
                               double sampleRate,
                               const ItoMultibandCompSettings& settings);

  static void processGain(engine::AudioBuffer& buffer, const ItoGainSettings& settings);

  // Mid/side stereo width.
  static void processImager(engine::AudioBuffer& buffer, const ItoImagerSettings& settings);

  // Lookahead ceiling limiter (stereo-linked envelope, delay-line lookahead).
  static void processLimiter(engine::AudioBuffer& buffer,
                             double sampleRate,
                             const ItoLimiterSettings& settings);

  // ── Whole-chain entry point ────────────────────────────────────────────────

  void prepare(double sampleRate, int numChannels);
  void reset();
  void setSettings(const ItoMasterChainSettings& settings);
  void process(engine::AudioBuffer& buffer);

 private:
  double sampleRate_ = 44100.0;
  int numChannels_ = 2;
  ItoMasterChainSettings settings_;
};

} // namespace automix::dsp
