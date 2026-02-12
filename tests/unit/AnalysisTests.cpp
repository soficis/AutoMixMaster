#include <cmath>

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

} // namespace

TEST_CASE("Analysis computes expected peak and RMS for sine", "[analysis]") {
  const auto buffer = makeSine(440.0, 0.5, 44100.0, 44100);

  automix::analysis::StemAnalyzer analyzer;
  const auto metrics = analyzer.analyzeBuffer(buffer);

  REQUIRE(metrics.peakDb == Catch::Approx(-6.02).margin(0.5));
  REQUIRE(metrics.rmsDb == Catch::Approx(-9.03).margin(0.7));
  REQUIRE(metrics.crestDb == Catch::Approx(3.0).margin(0.7));
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
}
