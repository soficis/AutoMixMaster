#include <cmath>
#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "ai/StemSeparator.h"
#include "engine/AudioBuffer.h"
#include "util/WavWriter.h"

namespace {

automix::engine::AudioBuffer makeTone(const double sampleRate, const int samples, const double frequency) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample = static_cast<float>(0.35 * std::sin(2.0 * 3.14159265358979323846 * frequency * i / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

} // namespace

TEST_CASE("Stem separator uses model-backed overlap-add when model metadata is present", "[ai][separator]") {
  const std::filesystem::path tempRoot = std::filesystem::temp_directory_path() / "automix_separator_model_test";
  const auto modelDir = tempRoot / "model";
  const auto outputDir = tempRoot / "output";
  const auto mixPath = tempRoot / "mix.wav";
  std::filesystem::remove_all(tempRoot);
  std::filesystem::create_directories(modelDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(mixPath, makeTone(44100.0, 44100, 330.0), 24, "wav");

  {
    std::ofstream model(modelDir / "separator.onnx", std::ios::binary);
    model << "dummy_separator_model";
  }

  {
    nlohmann::json meta = {
        {"input_feature_count", 27},
        {"allowed_tasks", {"stem_separation"}},
        {"execution_providers", {"cpu"}},
    };
    std::ofstream metaOut((modelDir / "separator.onnx.meta.json").string());
    metaOut << meta.dump(2);
  }

  automix::ai::StemSeparator separator(modelDir);
  REQUIRE(separator.isModelAvailable());

  const auto result = separator.separate(mixPath, outputDir);
  REQUIRE(result.success);
  REQUIRE(result.usedModel);
  REQUIRE(result.stems.size() == 4);
  REQUIRE(result.generatedFiles.size() == 4);

  for (const auto& stem : result.stems) {
    REQUIRE(std::filesystem::exists(stem.filePath));
    REQUIRE(stem.separationConfidence.has_value());
    REQUIRE(stem.separationArtifactRisk.has_value());
    REQUIRE(stem.separationConfidence.value() >= 0.0);
    REQUIRE(stem.separationConfidence.value() <= 1.0);
    REQUIRE(stem.separationArtifactRisk.value() >= 0.0);
    REQUIRE(stem.separationArtifactRisk.value() <= 1.0);
  }

  std::filesystem::remove_all(tempRoot);
}
