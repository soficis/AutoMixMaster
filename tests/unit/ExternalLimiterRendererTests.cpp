#include <cmath>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "automaster/OriginalMixReference.h"
#include "domain/Session.h"
#include "renderers/ExternalLimiterRenderer.h"
#include "util/WavWriter.h"

namespace {

automix::engine::AudioBuffer makeTone(const double sampleRate, const int samples, const double frequency) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample =
        static_cast<float>(0.25 * std::sin(2.0 * 3.14159265358979323846 * frequency * static_cast<double>(i) / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

automix::engine::AudioBuffer makeMonoTone(const double sampleRate, const int samples, const double frequency, const float amplitude) {
  automix::engine::AudioBuffer buffer(1, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample =
        static_cast<float>(amplitude * std::sin(2.0 * 3.14159265358979323846 * frequency * static_cast<double>(i) / sampleRate));
    buffer.setSample(0, i, sample);
  }
  return buffer;
}

automix::domain::MasterPlan applySoftTarget(const automix::engine::AudioBuffer& stemMix,
                                            const automix::engine::AudioBuffer& originalMix) {
  automix::automaster::OriginalMixReference reference;
  automix::automaster::HeuristicAutoMasterStrategy strategy;
  automix::analysis::StemAnalyzer analyzer;
  automix::domain::MasterPlan basePlan;
  return reference.applySoftTarget(basePlan, stemMix, originalMix, strategy, analyzer);
}

} // namespace

TEST_CASE("External limiter renderer falls back to BuiltIn when binary is missing", "[renderer][external]") {
  const std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "automix_external_limiter_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  automix::util::WavWriter writer;
  const auto stemPath = tempDir / "stem.wav";
  writer.write(stemPath, makeTone(44100.0, 22050, 330.0), 24);

  automix::domain::Session session;
  automix::domain::Stem stem;
  stem.id = "s1";
  stem.name = "Stem";
  stem.filePath = stemPath.string();
  session.stems.push_back(stem);

  automix::domain::RenderSettings settings;
  settings.outputPath = (tempDir / "out.wav").string();
  settings.externalRendererPath = (tempDir / "missing_external_tool.exe").string();

  automix::renderers::ExternalLimiterRenderer renderer;
  const auto result = renderer.render(session, settings, {}, nullptr);

  REQUIRE(result.success);
  REQUIRE(result.rendererName.find("fallback") != std::string::npos);
  REQUIRE(std::filesystem::exists(settings.outputPath));

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("Original mix soft target keeps plan in bounds and applies blend", "[automaster][reference]") {
  const double sampleRate = 44100.0;
  const int samples = 44100;  // 1 second
  const automix::engine::AudioBuffer stemMix = makeTone(sampleRate, samples, 330.0);
  const automix::engine::AudioBuffer originalMix = makeTone(sampleRate, samples, 110.0);
  const automix::domain::MasterPlan basePlan;

  const auto plan = applySoftTarget(stemMix, originalMix);

  REQUIRE(plan.targetLufs >= -30.0);
  REQUIRE(plan.targetLufs <= -8.0);
  REQUIRE(plan.preGainDb >= -9.0);
  REQUIRE(plan.preGainDb <= 9.0);
  REQUIRE(plan.glueRatio >= 1.1);
  REQUIRE(plan.glueRatio <= 6.0);
  REQUIRE(plan.targetLufs != basePlan.targetLufs);
  // Applied path pushes exactly 3 decision-log entries; skip paths push 1.
  REQUIRE(plan.decisionLog.size() == basePlan.decisionLog.size() + 3);
}

TEST_CASE("Original mix soft target survives channel-count mismatch", "[automaster][reference]") {
  const double sampleRate = 44100.0;
  const int samples = 44100;  // 1 second
  const automix::engine::AudioBuffer stemMix = makeTone(sampleRate, samples, 330.0);
  const automix::engine::AudioBuffer originalMix = makeMonoTone(sampleRate, samples, 110.0, 0.5f);
  const automix::domain::MasterPlan basePlan;

  const auto plan = applySoftTarget(stemMix, originalMix);

  REQUIRE(plan.targetLufs >= -30.0);
  REQUIRE(plan.targetLufs <= -8.0);
  REQUIRE(plan.preGainDb >= -9.0);
  REQUIRE(plan.preGainDb <= 9.0);
  REQUIRE(plan.glueRatio >= 1.1);
  REQUIRE(plan.glueRatio <= 6.0);
  // A mono reference must not trip a skip guard: the applied path still runs.
  REQUIRE(plan.decisionLog.size() == basePlan.decisionLog.size() + 3);
}

TEST_CASE("Original mix soft target is deterministic across repeated calls", "[automaster][reference]") {
  const double sampleRate = 44100.0;
  const int samples = 44100;  // 1 second
  const automix::engine::AudioBuffer stemMix = makeTone(sampleRate, samples, 330.0);
  const automix::engine::AudioBuffer originalMix = makeTone(sampleRate, samples, 110.0);

  const auto first = applySoftTarget(stemMix, originalMix);
  const auto second = applySoftTarget(stemMix, originalMix);

  REQUIRE(second.targetLufs == first.targetLufs);
  REQUIRE(second.preGainDb == first.preGainDb);
  REQUIRE(second.glueRatio == first.glueRatio);
  REQUIRE(second.applyEq == first.applyEq);
  REQUIRE(second.decisionLog == first.decisionLog);
}
