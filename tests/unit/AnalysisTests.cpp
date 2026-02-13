#include <cmath>
#include <cstdint>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "analysis/StemAnalyzer.h"

namespace {

automix::engine::AudioBuffer makeSine(const double freq, const double amplitude, const double sampleRate, const int samples) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample = static_cast<float>(amplitude * std::sin(2.0 * 3.14159265358979323846 * freq * i / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

automix::engine::AudioBuffer makeNoise(const double sampleRate, const int samples, const float amplitude) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  uint32_t state = 0xA5B3571Du;
  for (int i = 0; i < samples; ++i) {
    state = state * 1664525u + 1013904223u;
    const float normalized = static_cast<float>((state >> 9) & 0x7FFFFF) / static_cast<float>(0x7FFFFF);
    const float value = (normalized * 2.0f - 1.0f) * amplitude;
    buffer.setSample(0, i, value);
    buffer.setSample(1, i, value);
  }
  return buffer;
}

} // namespace

TEST_CASE("Analysis computes expected peak and RMS for sine", "[analysis]") {
  const auto buffer = makeSine(440.0, 0.5, 44100.0, 44100);

  automix::analysis::StemAnalyzer analyzer;
  const auto metrics = analyzer.analyzeBuffer(buffer);

  REQUIRE(metrics.peakDb == Catch::Approx(-6.02).margin(0.5));
  REQUIRE(metrics.rmsDb == Catch::Approx(-9.03).margin(0.7));
  REQUIRE(metrics.crestDb == Catch::Approx(3.0).margin(0.7));
  REQUIRE(metrics.spectralCentroidHz == Catch::Approx(440.0).margin(180.0));
  REQUIRE(metrics.spectralFlatness < 0.4);
}

TEST_CASE("Analysis silence detection catches mostly silent buffers", "[analysis]") {
  automix::engine::AudioBuffer buffer(2, 2000, 44100.0);
  for (int i = 1000; i < 1020; ++i) {
    buffer.setSample(0, i, 0.2f);
    buffer.setSample(1, i, 0.2f);
  }

  automix::analysis::StemAnalyzer analyzer;
  const auto metrics = analyzer.analyzeBuffer(buffer);

  REQUIRE(metrics.silenceRatio > 0.95);
  REQUIRE(std::isfinite(metrics.spectralFlux));
  REQUIRE(metrics.spectralFlux >= 0.0);
}

TEST_CASE("Analysis stereo width distinguishes decorrelated channels", "[analysis]") {
  automix::engine::AudioBuffer buffer(2, 4096, 44100.0);
  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    const float left = (i % 2 == 0) ? 0.4f : -0.4f;
    const float right = -left;
    buffer.setSample(0, i, left);
    buffer.setSample(1, i, right);
  }

  automix::analysis::StemAnalyzer analyzer;
  const auto metrics = analyzer.analyzeBuffer(buffer);

  REQUIRE(metrics.stereoCorrelation < -0.8);
  REQUIRE(metrics.stereoWidth > 0.9);
  REQUIRE(metrics.artifactProfile.phaseInstability >= 0.0);
  REQUIRE(metrics.artifactProfile.phaseInstability <= 1.0);
}

TEST_CASE("Analysis spectral flatness is higher for noise than tone", "[analysis]") {
  automix::analysis::StemAnalyzer analyzer;

  const auto tone = makeSine(700.0, 0.35, 44100.0, 44100);
  const auto noise = makeNoise(44100.0, 44100, 0.35f);

  const auto toneMetrics = analyzer.analyzeBuffer(tone);
  const auto noiseMetrics = analyzer.analyzeBuffer(noise);

  REQUIRE(noiseMetrics.spectralFlatness > toneMetrics.spectralFlatness);
  REQUIRE(noiseMetrics.artifactRisk >= toneMetrics.artifactRisk);
}

TEST_CASE("Analysis handles short and full-scale buffers", "[analysis][edge]") {
  automix::engine::AudioBuffer shortBuffer(1, 32, 48000.0);
  for (int i = 0; i < shortBuffer.getNumSamples(); ++i) {
    const float value = (i % 2 == 0) ? 1.0f : -1.0f;
    shortBuffer.setSample(0, i, value);
  }

  automix::analysis::StemAnalyzer analyzer;
  const auto metrics = analyzer.analyzeBuffer(shortBuffer);

  REQUIRE(std::isfinite(metrics.rmsDb));
  REQUIRE(std::isfinite(metrics.peakDb));
  REQUIRE(metrics.peakDb <= 0.1);
  REQUIRE(metrics.peakDb >= -0.2);
  REQUIRE(metrics.silenceRatio < 0.2);
}
