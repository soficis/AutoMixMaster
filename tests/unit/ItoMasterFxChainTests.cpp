#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dsp/ItoMasterFxChain.h"
#include "engine/AudioBuffer.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

automix::engine::AudioBuffer makeStereo(const int samples, const double sampleRate, const double freq,
                               const double amplitude) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const double t = static_cast<double>(i) / sampleRate;
    const float value = static_cast<float>(amplitude * std::sin(2.0 * kPi * freq * t));
    buffer.setSample(0, i, value);
    buffer.setSample(1, i, value);
  }
  return buffer;
}

double rmsDb(const automix::engine::AudioBuffer& buffer, const int startSample) {
  double sumSquares = 0.0;
  int count = 0;
  for (int i = startSample; i < buffer.getNumSamples(); ++i) {
    const double value = buffer.getSample(0, i);
    sumSquares += value * value;
    ++count;
  }
  if (count == 0) {
    return -120.0;
  }
  return 10.0 * std::log10(sumSquares / static_cast<double>(count) + 1.0e-12);
}

float peakAbs(const automix::engine::AudioBuffer& buffer) {
  float peak = 0.0f;
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      peak = std::max(peak, std::abs(buffer.getSample(ch, i)));
    }
  }
  return peak;
}

} // namespace

TEST_CASE("ItoMaster EQ biquad gain matches the setting at the band centre", "[dsp][ito]") {
  constexpr double sampleRate = 44100.0;
  constexpr int samples = 44100;
  auto input = makeStereo(samples, sampleRate, 1000.0, 0.25);
  const double inputDb = rmsDb(input, samples / 2);

  automix::dsp::ItoEqSettings eq;
  eq.bands[1].gainDb = 6.0;
  eq.bands[1].freqHz = 1000.0;
  eq.bands[1].qFactor = 1.0;

  automix::dsp::ItoMasterFxChain::processEq(input, eq);

  const double outputDb = rmsDb(input, samples / 2);
  const double measuredGainDb = outputDb - inputDb;
  REQUIRE(measuredGainDb == Catch::Approx(6.0).margin(0.75));
}

TEST_CASE("ItoMaster multiband compressor applies gain reduction under overdrive", "[dsp][ito]") {
  constexpr double sampleRate = 44100.0;
  constexpr int samples = 44100;
  auto input = makeStereo(samples, sampleRate, 50.0, 0.5); // fully inside the low band
  const double inputDb = rmsDb(input, samples / 4);

  automix::dsp::ItoMultibandCompSettings comp;
  comp.lowCrossoverHz = 120.0;
  comp.highCrossoverHz = 6000.0;
  comp.parallelWeight = 0.7;
  comp.bands[0].compThresholdDb = -20.0;
  comp.bands[0].compRatio = 4.0;
  comp.bands[0].attackMs = 5.0;
  comp.bands[0].releaseMs = 50.0;

  automix::dsp::ItoMasterFxChain::processMultiband(input, sampleRate, comp);

  const double outputDb = rmsDb(input, samples / 4);
  REQUIRE(outputDb < inputDb - 3.0);  // clear gain reduction
  REQUIRE(outputDb > inputDb - 12.0); // not over-the-top
}

TEST_CASE("ItoMaster limiter clamps output to the ceiling", "[dsp][ito]") {
  constexpr double sampleRate = 44100.0;
  constexpr int samples = 44100;
  auto input = makeStereo(samples, sampleRate, 1000.0, 1.0); // 0 dBFS
  const double ceilingDb = -3.0;
  const float ceiling = static_cast<float>(std::pow(10.0, ceilingDb / 20.0));

  automix::dsp::ItoLimiterSettings limiter;
  limiter.thresholdDb = ceilingDb;
  limiter.attackMs = 5.0;
  limiter.releaseMs = 80.0;
  limiter.lookaheadMs = 5.0;

  automix::dsp::ItoMasterFxChain::processLimiter(input, sampleRate, limiter);

  const float peak = peakAbs(input);
  REQUIRE(peak <= ceiling + 1.0e-3f);
  REQUIRE(peak > ceiling * 0.9f); // the limiter actually engaged
}

TEST_CASE("ItoMaster Linkwitz-Riley crossover bands sum to unity magnitude", "[dsp][ito]") {
  constexpr double sampleRate = 44100.0;
  constexpr int samples = 44100;

  automix::dsp::ItoMultibandCompSettings comp;
  comp.lowCrossoverHz = 120.0;
  comp.highCrossoverHz = 6000.0;
  comp.parallelWeight = 1.0; // wet-only so the band sum is measured directly
  for (auto& band : comp.bands) {
    band.compThresholdDb = -30.0;
    band.compRatio = 1.000001; // ~0 dB compression
    band.expThresholdDb = -80.0;
    band.expRatio = 0.999999; // ~0 dB expansion
    band.attackMs = 5.0;
    band.releaseMs = 50.0;
  }

  for (const double freq : {60.0, 200.0, 1000.0, 3000.0, 9000.0, 15000.0}) {
    auto input = makeStereo(samples, sampleRate, freq, 0.25);
    const double inputDb = rmsDb(input, samples / 2);
    automix::dsp::ItoMasterFxChain::processMultiband(input, sampleRate, comp);
    const double outputDb = rmsDb(input, samples / 2);
    REQUIRE(outputDb == Catch::Approx(inputDb).margin(0.75));
  }
}

TEST_CASE("ItoMaster imager collapses to mono at width zero and is transparent at width one", "[dsp][ito]") {
  constexpr int samples = 8192;
  constexpr double sampleRate = 44100.0;
  auto input = makeStereo(samples, sampleRate, 440.0, 0.3);
  input.setSample(0, 0, 0.3f);
  input.setSample(1, 0, -0.3f);

  automix::dsp::ItoImagerSettings collapsed;
  collapsed.width = 0.0;
  auto mono = input;
  automix::dsp::ItoMasterFxChain::processImager(mono, collapsed);
  for (int i = 0; i < samples; ++i) {
    REQUIRE(mono.getSample(0, i) == Catch::Approx(mono.getSample(1, i)).margin(1.0e-5f));
  }

  automix::dsp::ItoImagerSettings transparent;
  transparent.width = 1.0;
  auto passThrough = input;
  automix::dsp::ItoMasterFxChain::processImager(passThrough, transparent);
  for (int i = 0; i < samples; ++i) {
    REQUIRE(passThrough.getSample(0, i) == Catch::Approx(input.getSample(0, i)).margin(1.0e-5f));
    REQUIRE(passThrough.getSample(1, i) == Catch::Approx(input.getSample(1, i)).margin(1.0e-5f));
  }
}

TEST_CASE("ItoMaster chain processes a full pass with neutral settings", "[dsp][ito]") {
  constexpr double sampleRate = 44100.0;
  constexpr int samples = 44100;
  auto input = makeStereo(samples, sampleRate, 440.0, 0.2);

  automix::dsp::ItoMasterChainSettings settings;
  // Neutral: EQ shelves/bands at 0 dB, no drive, near-unity ratios, 0 dB gain,
  // full-width imager, and a ceiling far above the signal level.
  settings.multiband.bands[0].compRatio = 1.000001;
  settings.multiband.bands[0].expRatio = 0.999999;
  settings.multiband.bands[1].compRatio = 1.000001;
  settings.multiband.bands[1].expRatio = 0.999999;
  settings.multiband.bands[2].compRatio = 1.000001;
  settings.multiband.bands[2].expRatio = 0.999999;
  settings.limiter.thresholdDb = -0.1;
  settings.limiter.lookaheadMs = 5.0;

  const double inputDb = rmsDb(input, samples / 4);
  automix::dsp::ItoMasterFxChain chain;
  chain.prepare(sampleRate, 2);
  chain.setSettings(settings);
  chain.process(input);
  const double outputDb = rmsDb(input, samples / 4);

  REQUIRE(outputDb == Catch::Approx(inputDb).margin(1.5));
  REQUIRE(peakAbs(input) > 0.0f);
}
