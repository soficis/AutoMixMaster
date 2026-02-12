#include <cmath>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "domain/Session.h"
#include "renderers/PhaseLimiterRenderer.h"
#include "util/WavWriter.h"

namespace {

automix::engine::AudioBuffer makeTone(const double sampleRate,
                                      const int samples,
                                      const double frequency,
                                      const double amplitude) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample = static_cast<float>(amplitude * std::sin(2.0 * 3.14159265358979323846 * frequency * i / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

} // namespace

TEST_CASE("PhaseLimiter renderer never crashes and always returns a valid render result", "[phaselimiter][renderer]") {
  const std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "automix_phaselimiter_renderer_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  automix::util::WavWriter writer;
  const auto stemA = makeTone(44100.0, 22050, 220.0, 0.40);
  const auto stemB = makeTone(44100.0, 22050, 880.0, 0.20);

  const auto stemPathA = tempDir / "bass.wav";
  const auto stemPathB = tempDir / "lead.wav";
  writer.write(stemPathA, stemA, 24);
  writer.write(stemPathB, stemB, 24);

  automix::domain::Session session;
  automix::domain::Stem s1;
  s1.id = "s1";
  s1.name = "Bass";
  s1.filePath = stemPathA.string();
  automix::domain::Stem s2;
  s2.id = "s2";
  s2.name = "Lead";
  s2.filePath = stemPathB.string();
  session.stems.push_back(s1);
  session.stems.push_back(s2);

  automix::domain::RenderSettings settings;
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;
  settings.outputBitDepth = 24;
  settings.rendererName = "PhaseLimiter";
  settings.outputPath = (tempDir / "phaselimiter_render.wav").string();

  automix::renderers::PhaseLimiterRenderer renderer;
  const auto result = renderer.render(session, settings, {}, nullptr);

  REQUIRE(result.cancelled == false);
  REQUIRE(result.success == true);
  REQUIRE(std::filesystem::exists(settings.outputPath));
  REQUIRE(std::filesystem::exists(result.reportPath));
  REQUIRE(result.logs.empty() == false);

  std::filesystem::remove_all(tempDir);
}
