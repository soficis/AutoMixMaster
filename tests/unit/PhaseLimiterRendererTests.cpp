#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "domain/Session.h"
#include "renderers/PhaseLimiterDiscovery.h"
#include "renderers/PhaseLimiterRenderer.h"
#include "util/WavWriter.h"

namespace {

void setEnvValue(const char* key, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(key, value.c_str());
#else
  if (value.empty()) {
    unsetenv(key);
  } else {
    setenv(key, value.c_str(), 1);
  }
#endif
}

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

TEST_CASE("A selected PhaseLimiter really renders and leaves no scratch behind", "[phaselimiter][renderer]") {
  automix::renderers::PhaseLimiterRenderer renderer;
  if (!renderer.isAvailable()) {
    SKIP("No complete PhaseLimiter install found");
  }
  const auto installRoot = automix::renderers::PhaseLimiterDiscovery{}.find()->installRoot;
  const auto countEntries = [](const std::filesystem::path& dir) {
    std::error_code error;
    std::size_t count = 0;
    if (std::filesystem::is_directory(dir, error)) {
      for ([[maybe_unused]] const auto& entry : std::filesystem::directory_iterator(dir, error)) {
        ++count;
      }
    }
    return count;
  };

  const std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "automix_phaselimiter_real_render";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);
  automix::util::WavWriter writer;
  writer.write(tempDir / "bass.wav", makeTone(44100.0, 44100, 110.0, 0.40), 24);
  writer.write(tempDir / "lead.wav", makeTone(44100.0, 44100, 660.0, 0.20), 24);
  automix::domain::Session session;
  automix::domain::Stem bass;
  bass.id = "bass";
  bass.name = "Bass";
  bass.filePath = (tempDir / "bass.wav").string();
  automix::domain::Stem lead;
  lead.id = "lead";
  lead.name = "Lead";
  lead.filePath = (tempDir / "lead.wav").string();
  session.stems = {bass, lead};

  automix::domain::RenderSettings settings;
  settings.rendererName = "PhaseLimiter";
  settings.outputPath = (tempDir / "out.wav").string();

  const auto scratchBase = std::filesystem::temp_directory_path() / "automix_phaselimiter";
  const auto workingDirectory = std::filesystem::current_path();
  const auto scratchBefore = countEntries(scratchBase);
  const auto legacyBefore = countEntries(installRoot / "tmp");

  const auto result = renderer.render(session, settings, {}, nullptr);
  std::string logs;
  for (const auto& line : result.logs) {
    logs += line + "\n";
  }
  INFO(logs);
  REQUIRE(result.success);
  REQUIRE(result.rendererName == "PhaseLimiter");  // not "PhaseLimiter (fallback BuiltIn)"
  REQUIRE(logs.find("fallback") == std::string::npos);
  REQUIRE(std::filesystem::exists(settings.outputPath));

  REQUIRE(countEntries(scratchBase) <= scratchBefore);            // its own run is gone
  REQUIRE(countEntries(installRoot / "tmp") <= legacyBefore);     // nothing new in the install
  REQUIRE(std::filesystem::current_path() == workingDirectory);   // no process-wide cwd change
  std::filesystem::remove_all(tempDir);
}

TEST_CASE("PhaseLimiter render reports missing ffmpeg instead of falling back", "[phaselimiter][renderer]") {
  const std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "automix_pl_test_missing_ffmpeg";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

  const std::filesystem::path fakeInstall = tempDir / "fake_phaselimiter";
  const std::filesystem::path fakeBinDir = fakeInstall / "bin";
  const std::filesystem::path fakeResourceDir = fakeInstall / "resource";
  std::filesystem::create_directories(fakeBinDir);
  std::filesystem::create_directories(fakeResourceDir);

#if defined(_WIN32)
  const std::filesystem::path fakeExe = fakeBinDir / "phase_limiter.exe";
#else
  const std::filesystem::path fakeExe = fakeBinDir / "phase_limiter";
#endif
  {
    std::ofstream out(fakeExe);
    out << "binary stub\n";
  }
  {
    std::ofstream out(fakeResourceDir / "mastering_reference.json");
    out << "{}\n";
  }
#if !defined(_WIN32)
  std::filesystem::permissions(fakeExe, std::filesystem::perms::owner_exec | std::filesystem::perms::owner_read, std::filesystem::perm_options::add);
#endif

  setEnvValue("PHASELIMITER_BIN", fakeExe.string());
  automix::renderers::setFfmpegResolverForTesting([]() -> std::optional<std::filesystem::path> {
    return std::nullopt;
  });

  struct CleanupGuard {
    std::filesystem::path dir;
    ~CleanupGuard() {
      automix::renderers::resetFfmpegResolverForTesting();
      setEnvValue("PHASELIMITER_BIN", "");
      std::filesystem::remove_all(dir);
    }
  } cleanup{tempDir};

  automix::util::WavWriter writer;
  const auto stemA = makeTone(44100.0, 22050, 220.0, 0.40);
  const auto stemPathA = tempDir / "tone.wav";
  writer.write(stemPathA, stemA, 24);

  automix::domain::Session session;
  automix::domain::Stem s1;
  s1.id = "s1";
  s1.name = "Tone";
  s1.filePath = stemPathA.string();
  session.stems.push_back(s1);

  automix::domain::RenderSettings settings;
  settings.outputSampleRate = 44100;
  settings.blockSize = 1024;
  settings.outputBitDepth = 24;
  settings.rendererName = "PhaseLimiter";
  settings.outputPath = (tempDir / "out.wav").string();

  automix::renderers::PhaseLimiterRenderer renderer;
  const auto result = renderer.render(session, settings, {}, nullptr);

  REQUIRE(result.success == false);
  REQUIRE(result.rendererName == "PhaseLimiter");
  const std::string expectedMessage = "PhaseLimiter needs ffmpeg, which was not found. Install ffmpeg or set FFMPEG_BIN.";
  REQUIRE_FALSE(result.logs.empty());
  REQUIRE(result.logs.back() == expectedMessage);
}

TEST_CASE("ffmpegForPhaseLimiter honours FFMPEG_BIN first", "[phaselimiter][discovery]") {
  const std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "automix_test_ffmpeg_bin";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);

#if defined(_WIN32)
  const auto fakeFfmpeg = tempDir / "fake_ffmpeg.exe";
#else
  const auto fakeFfmpeg = tempDir / "fake_ffmpeg";
#endif
  {
    std::ofstream out(fakeFfmpeg);
    out << "stub\n";
  }
#if !defined(_WIN32)
  std::filesystem::permissions(fakeFfmpeg, std::filesystem::perms::owner_exec | std::filesystem::perms::owner_read, std::filesystem::perm_options::add);
#endif

  automix::renderers::resetFfmpegResolverForTesting();
  setEnvValue("FFMPEG_BIN", fakeFfmpeg.string());

  struct CleanupGuard {
    std::filesystem::path dir;
    ~CleanupGuard() {
      setEnvValue("FFMPEG_BIN", "");
      std::filesystem::remove_all(dir);
    }
  } cleanup{tempDir};

  const auto found = automix::renderers::ffmpegForPhaseLimiter();
  REQUIRE(found.has_value());
  REQUIRE(std::filesystem::equivalent(*found, fakeFfmpeg));
}