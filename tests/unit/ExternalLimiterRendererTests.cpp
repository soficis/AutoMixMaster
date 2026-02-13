#include <cmath>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

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
