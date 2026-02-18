#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "automaster/HeuristicAutoMasterStrategy.h"
#include "dsp/DeEsser.h"
#include "dsp/DynamicDeHarshEq.h"
#include "dsp/MidSideProcessor.h"
#include "dsp/SoftClipper.h"
#include "domain/MasterPlan.h"
#include "engine/AudioBuffer.h"

namespace {

automix::engine::AudioBuffer makeSibilantLikeSignal() {
  constexpr double sampleRate = 48000.0;
  constexpr int samples = 48000;
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const double t = static_cast<double>(i) / sampleRate;
    const float sample = static_cast<float>(0.25 * std::sin(2.0 * 3.14159265358979323846 * 200.0 * t) +
                                            0.12 * std::sin(2.0 * 3.14159265358979323846 * 7000.0 * t));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

bool allFinite(const automix::engine::AudioBuffer& buffer) {
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      if (!std::isfinite(buffer.getSample(ch, i))) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

TEST_CASE("Deterministic mastering modules keep buffers finite", "[master][modules]") {
  auto signal = makeSibilantLikeSignal();

  automix::dsp::DeEsser deEsser;
  automix::dsp::DynamicDeHarshEq deHarsh;
  automix::dsp::MidSideProcessor midSide;
  automix::dsp::SoftClipper softClipper;

  deEsser.process(signal, 0.6);
  deHarsh.process(signal, 0.5);
  midSide.process(signal, 120.0, 0.9);
  softClipper.process(signal, 1.2);

  REQUIRE(allFinite(signal));
}

TEST_CASE("Udio optimized preset enables deterministic safety modules", "[master][modules]") {
  automix::automaster::HeuristicAutoMasterStrategy strategy;
  const auto signal = makeSibilantLikeSignal();

  const auto plan = strategy.buildPlan(automix::domain::MasterPreset::UdioOptimized, signal);
  REQUIRE(plan.enableDeEsser);
  REQUIRE(plan.enableDeHarshEq);
  REQUIRE(plan.enableLowMono);
  REQUIRE(plan.enableSoftClipper);
  REQUIRE(plan.targetLufs == Catch::Approx(-14.0));
  REQUIRE(plan.truePeakDbtp == Catch::Approx(-1.0));
}
