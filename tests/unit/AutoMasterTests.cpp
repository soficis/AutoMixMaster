#include <cmath>
#include <algorithm>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automaster/ReferenceMatchStrategy.h"

namespace {

automix::engine::AudioBuffer makeBusySignal() {
  const double sampleRate = 44100.0;
  const int samples = 44100 * 2;
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const double t = static_cast<double>(i) / sampleRate;
    const float sample = static_cast<float>(0.85 * std::sin(2.0 * 3.14159265358979323846 * 220.0 * t) +
                                            0.35 * std::sin(2.0 * 3.14159265358979323846 * 1100.0 * t));
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

automix::engine::AudioBuffer makeAlternatingLoudnessSignal() {
  const double sampleRate = 44100.0;
  const int blockSamples = static_cast<int>(sampleRate * 0.4);
  const int blocks = 20;
  automix::engine::AudioBuffer buffer(2, blockSamples * blocks, sampleRate);
  for (int block = 0; block < blocks; ++block) {
    const double amplitude = (block % 2 == 0) ? 0.5 : 0.02;
    for (int i = 0; i < blockSamples; ++i) {
      const int index = block * blockSamples + i;
      const double t = static_cast<double>(index) / sampleRate;
      const float sample =
          static_cast<float>(amplitude * std::sin(2.0 * 3.14159265358979323846 * 220.0 * t));
      buffer.setSample(0, index, sample);
      buffer.setSample(1, index, sample);
    }
  }
  return buffer;
}

} // namespace

TEST_CASE("Mastering enforces true peak ceiling", "[master]") {
  automix::automaster::HeuristicAutoMasterStrategy strategy;
  const auto input = makeBusySignal();
  auto plan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);
  plan.truePeakDbtp = -1.0;
  plan.limiterCeilingDb = -1.0;

  automix::automaster::MasteringReport report;
  const auto output = strategy.applyPlan(input, plan, &report);

  REQUIRE(output.getNumSamples() == input.getNumSamples());
  REQUIRE(report.truePeakDbtp <= -0.8);
  REQUIRE(report.monoCorrelation <= 1.0);
  REQUIRE(report.monoCorrelation >= -1.0);
}

TEST_CASE("Mastering lands near target loudness", "[master]") {
  automix::automaster::HeuristicAutoMasterStrategy strategy;
  const auto input = makeBusySignal();

  auto plan = strategy.buildPlan(automix::domain::MasterPreset::Broadcast, input);
  automix::automaster::MasteringReport report;
  strategy.applyPlan(input, plan, &report);

  REQUIRE(report.integratedLufs == Catch::Approx(plan.targetLufs).margin(1.2));
}

TEST_CASE("Mastering dither stage remains peak safe", "[master]") {
  automix::automaster::HeuristicAutoMasterStrategy strategy;
  const auto input = makeBusySignal();

  auto plan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);
  plan.ditherBitDepth = 16;
  plan.truePeakDbtp = -1.0;
  plan.limiterCeilingDb = -1.0;

  const auto output = strategy.applyPlan(input, plan, nullptr);
  REQUIRE(peakLinear(output) <= Catch::Approx(std::pow(10.0, -1.0 / 20.0)).epsilon(0.1));
}

TEST_CASE("Reference match is disabled by default and preserves the heuristic plan", "[master]") {
  const auto reference = makeAlternatingLoudnessSignal();
  const auto input = makeBusySignal();
  const auto profile = automix::automaster::measureReferenceProfile(reference);

  automix::automaster::HeuristicAutoMasterStrategy heuristic;
  automix::automaster::ReferenceMatchStrategy strategy(profile);
  REQUIRE_FALSE(strategy.isEnabled());

  const auto base =
      heuristic.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);
  const auto plan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);

  REQUIRE(plan.targetLufs == Catch::Approx(base.targetLufs));
  REQUIRE(plan.preGainDb == Catch::Approx(base.preGainDb));
  REQUIRE(plan.glueRatio == Catch::Approx(base.glueRatio));
  REQUIRE(plan.limiterCeilingDb == Catch::Approx(base.limiterCeilingDb));
}

TEST_CASE("Reference match derives loudness, ceiling and glue from the reference", "[master]") {
  const auto reference = makeAlternatingLoudnessSignal();
  const auto input = makeBusySignal();
  const auto profile = automix::automaster::measureReferenceProfile(reference);

  automix::automaster::ReferenceMatchStrategy strategy(profile);
  strategy.setEnabled(true);
  const auto plan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);

  REQUIRE(plan.targetLufs >= -14.0);
  REQUIRE(plan.targetLufs <= -7.0);
  REQUIRE(plan.limiterCeilingDb <= -1.0);
  REQUIRE(plan.glueRatio >= 1.0);
  REQUIRE(plan.glueRatio <= 4.0);

  const std::string log = [&plan] {
    std::string joined;
    for (const auto& entry : plan.decisionLog) {
      joined += entry;
    }
    return joined;
  }();
  REQUIRE(log.find("Reference match active") != std::string::npos);
}

TEST_CASE("Reference match keeps the heuristic mastering stage order", "[master]") {
  const auto reference = makeAlternatingLoudnessSignal();
  const auto input = makeBusySignal();

  automix::automaster::HeuristicAutoMasterStrategy heuristic;
  automix::automaster::ReferenceMatchStrategy strategy(automix::automaster::measureReferenceProfile(reference));
  strategy.setEnabled(true);
  const auto plan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);

  automix::automaster::MasteringReport heuristicReport;
  const auto heuristicOutput = heuristic.applyPlan(input, plan, &heuristicReport);
  automix::automaster::MasteringReport strategyReport;
  const auto strategyOutput = strategy.applyPlan(input, plan, &strategyReport);

  REQUIRE(strategyReport.activeModules == heuristicReport.activeModules);
  REQUIRE(strategyOutput.getNumSamples() == heuristicOutput.getNumSamples());
}

TEST_CASE("Reference match glue ratio follows reference dynamics direction", "[master]") {
  const auto input = makeBusySignal();

  automix::automaster::ReferenceProfile dynamicReference;
  dynamicReference.integratedLufs = -10.0;
  dynamicReference.truePeakDbtp = -1.0;
  dynamicReference.p50BlockLufs = -20.0;
  dynamicReference.p95BlockLufs = -5.0;

  automix::automaster::ReferenceProfile compressedReference = dynamicReference;
  compressedReference.p50BlockLufs = -11.0;
  compressedReference.p95BlockLufs = -10.0;

  automix::automaster::ReferenceMatchStrategy againstDynamicReference(dynamicReference);
  againstDynamicReference.setEnabled(true);
  automix::automaster::ReferenceMatchStrategy againstCompressedReference(compressedReference);
  againstCompressedReference.setEnabled(true);

  const auto dynamicPlan =
      againstDynamicReference.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);
  const auto compressedPlan =
      againstCompressedReference.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);

  REQUIRE(compressedPlan.glueRatio >= dynamicPlan.glueRatio);
}

TEST_CASE("Reference profile measurement reports usable block dynamics", "[master]") {
  const auto profile = automix::automaster::measureReferenceProfile(makeAlternatingLoudnessSignal());

  REQUIRE(profile.integratedLufs > -120.0);
  REQUIRE(profile.truePeakDbtp < 0.0);
  REQUIRE(profile.p95BlockLufs > profile.p50BlockLufs);
}
