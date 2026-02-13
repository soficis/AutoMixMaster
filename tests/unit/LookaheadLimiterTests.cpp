#include <algorithm>
#include <cmath>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dsp/LookaheadLimiter.h"
#include "dsp/TruePeakDetector.h"
#include "engine/AudioBuffer.h"

namespace {

automix::engine::AudioBuffer makeHotSignal() {
  constexpr double sampleRate = 48000.0;
  constexpr int samples = 48000 * 2;
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const double t = static_cast<double>(i) / sampleRate;
    const float sample = static_cast<float>(1.35 * std::sin(2.0 * 3.14159265358979323846 * 220.0 * t) +
                                            0.35 * std::sin(2.0 * 3.14159265358979323846 * 5100.0 * t));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

double peakLinear(const automix::engine::AudioBuffer& buffer) {
  double peak = 0.0;
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      peak = std::max(peak, static_cast<double>(std::abs(buffer.getSample(ch, i))));
    }
  }
  return peak;
}

bool hasFiniteValues(const automix::engine::AudioBuffer& buffer) {
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const float sample = buffer.getSample(ch, i);
      if (!std::isfinite(sample)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

TEST_CASE("Lookahead limiter enforces ceiling and true-peak bounds", "[limiter]") {
  auto signal = makeHotSignal();

  automix::dsp::LookaheadLimiterSettings settings;
  settings.ceilingDb = -1.0;
  settings.lookaheadMs = 8.0;
  settings.attackMs = 1.0;
  settings.releaseMs = 80.0;
  settings.truePeakEnabled = true;
  settings.truePeakOversampleFactor = 4;
  settings.softClipEnabled = true;
  settings.softClipDrive = 1.2;

  automix::dsp::LookaheadLimiter limiter;
  limiter.prepare(signal.getSampleRate(), signal.getNumChannels(), settings);
  limiter.process(signal);

  REQUIRE(hasFiniteValues(signal));
  REQUIRE(peakLinear(signal) <= Catch::Approx(std::pow(10.0, settings.ceilingDb / 20.0)).epsilon(0.08));

  automix::dsp::TruePeakDetector truePeak(4);
  REQUIRE(truePeak.computeTruePeakDbtp(signal) <= -0.8);
}

TEST_CASE("Lookahead limiter is deterministic for identical input", "[limiter]") {
  auto signalA = makeHotSignal();
  auto signalB = signalA;

  automix::dsp::LookaheadLimiterSettings settings;
  settings.ceilingDb = -1.0;
  settings.truePeakEnabled = true;
  settings.truePeakOversampleFactor = 4;

  automix::dsp::LookaheadLimiter limiterA;
  limiterA.prepare(signalA.getSampleRate(), signalA.getNumChannels(), settings);
  limiterA.process(signalA);

  automix::dsp::LookaheadLimiter limiterB;
  limiterB.prepare(signalB.getSampleRate(), signalB.getNumChannels(), settings);
  limiterB.process(signalB);

  for (int ch = 0; ch < signalA.getNumChannels(); ++ch) {
    for (int i = 0; i < signalA.getNumSamples(); ++i) {
      REQUIRE(signalA.getSample(ch, i) == Catch::Approx(signalB.getSample(ch, i)).margin(1.0e-6));
    }
  }
}

TEST_CASE("Lookahead limiter keeps silence stable", "[limiter]") {
  automix::engine::AudioBuffer silence(2, 4096, 48000.0);

  automix::dsp::LookaheadLimiterSettings settings;
  settings.ceilingDb = -1.0;
  settings.truePeakEnabled = true;

  automix::dsp::LookaheadLimiter limiter;
  limiter.prepare(silence.getSampleRate(), silence.getNumChannels(), settings);
  limiter.process(silence);

  REQUIRE(hasFiniteValues(silence));
  REQUIRE(peakLinear(silence) == Catch::Approx(0.0).margin(1.0e-8));
}
