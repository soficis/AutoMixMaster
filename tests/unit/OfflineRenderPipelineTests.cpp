#include <algorithm>
#include <cmath>
#include <filesystem>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/Bus.h"
#include "domain/MixPlan.h"
#include "domain/Session.h"
#include "engine/OfflineRenderPipeline.h"
#include "util/WavWriter.h"

namespace {

automix::engine::AudioBuffer makeTone(const double sampleRate, const int samples, const double freq, const float amplitude) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample =
        static_cast<float>(amplitude * std::sin(2.0 * 3.14159265358979323846 * freq * static_cast<double>(i) / sampleRate));
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

double rmsLinear(const automix::engine::AudioBuffer& buffer) {
  double sum = 0.0;
  int count = 0;
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const double sample = buffer.getSample(ch, i);
      sum += sample * sample;
      ++count;
    }
  }
  return std::sqrt(sum / std::max(1, count));
}

automix::domain::Session makeSessionForStem(const std::filesystem::path& stemPath) {
  automix::domain::Session session;
  session.sessionName = "pipeline_test";

  automix::domain::Stem stem;
  stem.id = "s1";
  stem.name = "Stem";
  stem.filePath = stemPath.string();
  session.stems.push_back(stem);
  return session;
}

} // namespace

TEST_CASE("Offline render pipeline applies role bus gain routing", "[render][pipeline]") {
  const std::filesystem::path stemPath = std::filesystem::temp_directory_path() / "automix_pipeline_bus_stem.wav";
  automix::util::WavWriter writer;
  writer.write(stemPath, makeTone(44100.0, 44100, 440.0, 0.45f), 24);

  automix::domain::RenderSettings settings;
  settings.outputSampleRate = 44100;
  settings.blockSize = 512;

  automix::engine::OfflineRenderPipeline pipeline;
  auto baselineSession = makeSessionForStem(stemPath);
  const auto baseline = pipeline.renderRawMix(baselineSession, settings, {}, nullptr);

  auto routedSession = makeSessionForStem(stemPath);
  routedSession.stems.front().busId = "bus_music";
  routedSession.buses.push_back(automix::domain::Bus{
      .id = "bus_music",
      .name = "Music",
      .type = automix::domain::BusType::StemGroup,
      .gainDb = -12.0,
  });
  const auto routed = pipeline.renderRawMix(routedSession, settings, {}, nullptr);

  REQUIRE(routed.cancelled == false);
  REQUIRE(peakLinear(routed.mixBuffer) < peakLinear(baseline.mixBuffer) * 0.4);

  std::filesystem::remove(stemPath);
}

TEST_CASE("Offline render pipeline applies mud cut from mix plan", "[render][pipeline]") {
  const std::filesystem::path stemPath = std::filesystem::temp_directory_path() / "automix_pipeline_mudcut_stem.wav";
  automix::util::WavWriter writer;
  writer.write(stemPath, makeTone(44100.0, 44100, 320.0, 0.50f), 24);

  automix::domain::RenderSettings settings;
  settings.outputSampleRate = 44100;
  settings.blockSize = 512;

  automix::engine::OfflineRenderPipeline pipeline;
  auto baselineSession = makeSessionForStem(stemPath);
  const auto baseline = pipeline.renderRawMix(baselineSession, settings, {}, nullptr);

  auto processedSession = makeSessionForStem(stemPath);
  automix::domain::MixPlan plan;
  automix::domain::StemMixDecision decision;
  decision.stemId = "s1";
  decision.mudCutDb = -6.0;
  decision.highPassHz = 0.0;
  plan.stemDecisions.push_back(decision);
  processedSession.mixPlan = plan;

  const auto processed = pipeline.renderRawMix(processedSession, settings, {}, nullptr);
  REQUIRE(processed.cancelled == false);
  REQUIRE(rmsLinear(processed.mixBuffer) < rmsLinear(baseline.mixBuffer) * 0.9);

  std::filesystem::remove(stemPath);
}

TEST_CASE("Offline render pipeline is deterministic for identical input", "[render][pipeline]") {
  const std::filesystem::path stemPath = std::filesystem::temp_directory_path() / "automix_pipeline_determinism_stem.wav";
  automix::util::WavWriter writer;
  writer.write(stemPath, makeTone(44100.0, 44100, 220.0, 0.40f), 24);

  automix::domain::RenderSettings settings;
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;

  auto session = makeSessionForStem(stemPath);
  session.stems.front().busId = "bus_music";
  session.buses.push_back(automix::domain::Bus{
      .id = "bus_music",
      .name = "Music",
      .type = automix::domain::BusType::StemGroup,
      .gainDb = -3.0,
  });

  automix::engine::OfflineRenderPipeline pipeline;
  const auto runA = pipeline.renderRawMix(session, settings, {}, nullptr);
  const auto runB = pipeline.renderRawMix(session, settings, {}, nullptr);

  REQUIRE(runA.cancelled == false);
  REQUIRE(runB.cancelled == false);
  REQUIRE(runA.mixBuffer.getNumSamples() == runB.mixBuffer.getNumSamples());
  REQUIRE(runA.mixBuffer.getNumChannels() == runB.mixBuffer.getNumChannels());
  for (int ch = 0; ch < runA.mixBuffer.getNumChannels(); ++ch) {
    for (int i = 0; i < runA.mixBuffer.getNumSamples(); ++i) {
      REQUIRE(runA.mixBuffer.getSample(ch, i) == Catch::Approx(runB.mixBuffer.getSample(ch, i)).margin(1.0e-7));
    }
  }

  std::filesystem::remove(stemPath);
}
