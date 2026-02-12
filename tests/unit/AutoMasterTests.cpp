#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "automaster/HeuristicAutoMasterStrategy.h"

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
}

TEST_CASE("Mastering lands near target loudness", "[master]") {
  automix::automaster::HeuristicAutoMasterStrategy strategy;
  const auto input = makeBusySignal();

  auto plan = strategy.buildPlan(automix::domain::MasterPreset::Broadcast, input);
  automix::automaster::MasteringReport report;
  strategy.applyPlan(input, plan, &report);

  REQUIRE(report.integratedLufs == Catch::Approx(plan.targetLufs).margin(1.2));
}
