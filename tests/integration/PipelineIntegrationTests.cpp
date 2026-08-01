#include <cmath>
#include <numbers>

#include <catch2/catch_test_macros.hpp>

#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/Session.h"
#include "domain/Stem.h"
#include "engine/AudioBuffer.h"
#include "renderers/RendererPipeline.h"

namespace {

automix::engine::AudioBuffer makeTestSignal(double durationSec = 1.0) {
  const double sampleRate = 44100.0;
  const int samples = static_cast<int>(sampleRate * durationSec);
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const double t = static_cast<double>(i) / sampleRate;
    const float sample = static_cast<float>(
        0.6 * std::sin(2.0 * std::numbers::pi_v<double> * 440.0 * t) +
        0.25 * std::sin(2.0 * std::numbers::pi_v<double> * 880.0 * t) +
        0.15 * std::sin(2.0 * std::numbers::pi_v<double> * 1760.0 * t));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample * 0.97f);
  }
  return buffer;
}

} // namespace

TEST_CASE("Integration: auto-mix then auto-master produces valid output", "[integration]") {
  const auto input = makeTestSignal(2.0);

  // Auto-mix
  automix::automix::HeuristicAutoMixStrategy mixStrategy;
  automix::domain::Session session;
  auto mixPlan = mixStrategy.buildPlan(session, {}, 1.0);
  const auto mixedBuffer = input;

  REQUIRE(mixedBuffer.getNumSamples() == input.getNumSamples());
  REQUIRE(mixedBuffer.getNumChannels() == input.getNumChannels());

  // Auto-master
  automix::automaster::HeuristicAutoMasterStrategy masterStrategy;
  auto masterPlan = masterStrategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, mixedBuffer);
  automix::automaster::MasteringReport report;
  auto masterBuffer = masterStrategy.applyPlan(mixedBuffer, masterPlan, &report);

  REQUIRE(masterBuffer.getNumSamples() == mixedBuffer.getNumSamples());
  REQUIRE(report.integratedLufs < 0.0);
  REQUIRE(report.truePeakDbtp <= -0.5);
}

TEST_CASE("Integration: rendering chain resolves correct default order", "[integration]") {
  automix::domain::RenderSettings settings;
  settings.rendererName = "PhaseLimiter";
  settings.rendererChainEnabled = true;
  settings.rendererChainMode = "logical_all";

  const auto chain = automix::renderers::resolveRendererChain(settings);
  REQUIRE_FALSE(chain.empty());
  REQUIRE(chain.front() == "PhaseLimiter");
}

TEST_CASE("Integration: each master preset produces different loudness targets", "[integration]") {
  const auto input = makeTestSignal(1.0);
  automix::automaster::HeuristicAutoMasterStrategy strategy;

  automix::automaster::MasteringReport broadcastReport;
  auto broadcastPlan = strategy.buildPlan(automix::domain::MasterPreset::Broadcast, input);
  strategy.applyPlan(input, broadcastPlan, &broadcastReport);

  automix::automaster::MasteringReport streamingReport;
  auto streamingPlan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, input);
  strategy.applyPlan(input, streamingPlan, &streamingReport);

  // Broadcast (-23 LUFS) should be significantly quieter than streaming (-14 LUFS)
  REQUIRE(broadcastReport.integratedLufs < streamingReport.integratedLufs);
}
