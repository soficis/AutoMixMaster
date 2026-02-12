#include <algorithm>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dsp/SignalMath.h"
#include "engine/ResidualBlendProcessor.h"

namespace {

automix::engine::AudioBuffer makeStereoSignal(const int samples, const double sampleRate) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  uint32_t state = 0xABCD1234u;
  for (int i = 0; i < samples; ++i) {
    state = state * 1664525u + 1013904223u;
    const float noise = (static_cast<float>((state >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 0.05f;
    const float toneA = static_cast<float>(0.35 * std::sin(2.0 * 3.14159265358979323846 * 220.0 * i / sampleRate));
    const float toneB = static_cast<float>(0.22 * std::sin(2.0 * 3.14159265358979323846 * 880.0 * i / sampleRate));
    buffer.setSample(0, i, toneA + toneB + noise);
    buffer.setSample(1, i, toneA - toneB + noise * 0.8f);
  }
  return buffer;
}

automix::engine::AudioBuffer delaySignal(const automix::engine::AudioBuffer& input, const int delaySamples) {
  automix::engine::AudioBuffer delayed(input.getNumChannels(), input.getNumSamples(), input.getSampleRate());
  for (int ch = 0; ch < input.getNumChannels(); ++ch) {
    for (int i = 0; i < input.getNumSamples(); ++i) {
      const int source = i - delaySamples;
      const float sample = (source >= 0 && source < input.getNumSamples()) ? input.getSample(ch, source) : 0.0f;
      delayed.setSample(ch, i, sample);
    }
  }
  return delayed;
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

} // namespace

TEST_CASE("Alignment detects known offsets on synthetic signals", "[a6][alignment]") {
  constexpr int kSamples = 48000;
  constexpr double kSampleRate = 48000.0;
  constexpr int kDelay = 137;

  const auto stemSum = makeStereoSignal(kSamples, kSampleRate);
  const auto delayedOriginal = delaySignal(stemSum, kDelay);

  automix::engine::ResidualBlendProcessor processor;
  const auto alignment = processor.estimateAlignment(stemSum, delayedOriginal, 512);

  REQUIRE(alignment.sampleOffset == kDelay);
  REQUIRE(alignment.normalizedCorrelation > 0.98);
}

TEST_CASE("Residual computation is stable and blend stays below ceiling", "[a6][residual]") {
  constexpr int kSamples = 44100;
  constexpr double kSampleRate = 44100.0;

  auto stemSum = makeStereoSignal(kSamples, kSampleRate);
  auto original = stemSum;

  for (int i = 0; i < original.getNumSamples(); ++i) {
    const float residual = static_cast<float>(0.08 * std::sin(2.0 * 3.14159265358979323846 * 6200.0 * i / kSampleRate));
    original.setSample(0, i, original.getSample(0, i) + residual);
    original.setSample(1, i, original.getSample(1, i) - residual);
  }

  automix::engine::ResidualBlendProcessor processor;
  const auto residualComputation = processor.computeResidual(stemSum, original, 128);
  const auto blended = processor.applyResidualBlend(stemSum, residualComputation.residual, 10.0, -1.0);

  REQUIRE(residualComputation.alignment.normalizedCorrelation > 0.95);

  bool hasFiniteOnly = true;
  for (int ch = 0; ch < blended.getNumChannels(); ++ch) {
    for (int i = 0; i < blended.getNumSamples(); ++i) {
      if (!std::isfinite(static_cast<double>(blended.getSample(ch, i)))) {
        hasFiniteOnly = false;
        break;
      }
    }
  }

  REQUIRE(hasFiniteOnly);
  REQUIRE(peakLinear(blended) <= Catch::Approx(automix::dsp::dbToLinear(-1.0)).margin(1.0e-4));
}
