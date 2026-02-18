#include <cmath>
#include <random>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "engine/AudioBuffer.h"
#include "engine/LoudnessMeter.h"

namespace {

automix::engine::AudioBuffer makeSine(const double sampleRate, const int samples, const double frequency, const double amplitude) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample =
        static_cast<float>(amplitude * std::sin(2.0 * 3.14159265358979323846 * frequency * static_cast<double>(i) / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

automix::engine::AudioBuffer makeNoise(const double sampleRate, const int samples, const double amplitude) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  std::mt19937 rng(12345u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < samples; ++i) {
    const float sample = dist(rng) * static_cast<float>(amplitude);
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

} // namespace

TEST_CASE("Loudness meter sanity for silence tone and noise", "[loudness]") {
  automix::engine::LoudnessMeter meter;
  automix::engine::AudioBuffer silence(2, 48000, 48000.0);
  const auto tone = makeSine(48000.0, 48000 * 2, 1000.0, 0.2);
  const auto noise = makeNoise(48000.0, 48000 * 2, 0.2);

  const auto silenceMetrics = meter.analyze(silence);
  const auto toneMetrics = meter.analyze(tone);
  const auto noiseMetrics = meter.analyze(noise);

  REQUIRE(silenceMetrics.integratedLufs <= -70.0);
  REQUIRE(toneMetrics.integratedLufs > silenceMetrics.integratedLufs);
  REQUIRE(noiseMetrics.integratedLufs > silenceMetrics.integratedLufs);
}

TEST_CASE("Loudness meter integrated LUFS remains stable across chunk sizes", "[loudness]") {
  automix::engine::LoudnessMeter meter;
  const auto tone = makeSine(48000.0, 48000 * 3, 330.0, 0.25);

  const double lufs256 = meter.computeIntegratedLufs(tone, 256);
  const double lufs1024 = meter.computeIntegratedLufs(tone, 1024);
  const double lufs4096 = meter.computeIntegratedLufs(tone, 4096);

  REQUIRE(lufs256 == Catch::Approx(lufs1024).margin(0.25));
  REQUIRE(lufs1024 == Catch::Approx(lufs4096).margin(0.25));
}
