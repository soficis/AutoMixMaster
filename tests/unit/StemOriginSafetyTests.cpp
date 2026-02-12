#include <cmath>

#include <catch2/catch_test_macros.hpp>

#include "analysis/StemAnalyzer.h"
#include "automix/HeuristicAutoMixStrategy.h"

namespace {

automix::engine::AudioBuffer makeTone(const double frequency, const double amplitude) {
  const double sampleRate = 44100.0;
  const int samples = 44100;
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample = static_cast<float>(amplitude * std::sin(2.0 * 3.14159265358979323846 * frequency * i / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

automix::engine::AudioBuffer makeNoisyHighFrequencySignal() {
  const double sampleRate = 44100.0;
  const int samples = 44100;
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  uint32_t state = 0x12345678u;

  for (int i = 0; i < samples; ++i) {
    state = state * 1664525u + 1013904223u;
    const float noise = (static_cast<float>((state >> 9) & 0x7FFFFF) / static_cast<float>(0x7FFFFF) - 0.5f) * 0.6f;
    const float hfTone = static_cast<float>(0.25 * std::sin(2.0 * 3.14159265358979323846 * 10000.0 * i / sampleRate));
    const float sample = noise + hfTone;
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }

  return buffer;
}

} // namespace

TEST_CASE("Artifact risk is higher for noisy high-frequency content", "[analysis][a1]") {
  automix::analysis::StemAnalyzer analyzer;

  const auto cleanTone = makeTone(1000.0, 0.4);
  const auto noisySignal = makeNoisyHighFrequencySignal();

  const auto cleanMetrics = analyzer.analyzeBuffer(cleanTone);
  const auto noisyMetrics = analyzer.analyzeBuffer(noisySignal);

  REQUIRE(noisyMetrics.artifactRisk > cleanMetrics.artifactRisk);
}

TEST_CASE("AutoMix applies safer caps for separated stems", "[automix][a1]") {
  automix::domain::Session session;

  automix::domain::Stem separatedStem;
  separatedStem.id = "s_sep";
  separatedStem.name = "Vocal Separated";
  separatedStem.origin = automix::domain::StemOrigin::Separated;

  automix::domain::Stem recordedStem;
  recordedStem.id = "s_rec";
  recordedStem.name = "Vocal Recorded";
  recordedStem.origin = automix::domain::StemOrigin::Recorded;

  session.stems.push_back(separatedStem);
  session.stems.push_back(recordedStem);

  std::vector<automix::analysis::StemAnalysisEntry> analysisEntries;
  analysisEntries.push_back({.stemId = "s_sep", .stemName = "Vocal Separated", .metrics = {.rmsDb = -40.0, .highEnergy = 0.8, .artifactRisk = 0.9}});
  analysisEntries.push_back({.stemId = "s_rec", .stemName = "Vocal Recorded", .metrics = {.rmsDb = -40.0, .highEnergy = 0.3, .artifactRisk = 0.1}});

  automix::automix::HeuristicAutoMixStrategy strategy;
  const auto plan = strategy.buildPlan(session, analysisEntries, 1.0);

  REQUIRE(plan.stemDecisions.size() == 2);

  const auto& separatedDecision = plan.stemDecisions[0];
  const auto& recordedDecision = plan.stemDecisions[1];

  REQUIRE(std::abs(separatedDecision.gainDb) <= 5.0);
  REQUIRE(std::abs(recordedDecision.gainDb) >= std::abs(separatedDecision.gainDb));
  REQUIRE(separatedDecision.enableCompressor == false);
  REQUIRE(separatedDecision.enableExpander == true);
}
