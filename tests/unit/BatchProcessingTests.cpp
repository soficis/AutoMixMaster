#include <cmath>
#include <filesystem>
#include <regex>

#include <catch2/catch_test_macros.hpp>

#include "domain/BatchTypes.h"
#include "engine/BatchQueueRunner.h"
#include "util/WavWriter.h"

namespace {

automix::engine::AudioBuffer makeTone(const double sampleRate, const int samples, const double frequency) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample =
        static_cast<float>(0.2 * std::sin(2.0 * 3.14159265358979323846 * frequency * static_cast<double>(i) / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

} // namespace

TEST_CASE("Batch grouping detects stem sets by filename suffix", "[batch]") {
  const std::filesystem::path inputDir = std::filesystem::temp_directory_path() / "automix_batch_grouping_input";
  const std::filesystem::path outputDir = std::filesystem::temp_directory_path() / "automix_batch_grouping_output";
  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
  std::filesystem::create_directories(inputDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(inputDir / "songa_vocals.wav", makeTone(44100.0, 4096, 220.0), 24);
  writer.write(inputDir / "songa_bass.wav", makeTone(44100.0, 4096, 110.0), 24);
  writer.write(inputDir / "songb_vocals.wav", makeTone(44100.0, 4096, 330.0), 24);
  writer.write(inputDir / "songb_drums.wav", makeTone(44100.0, 4096, 440.0), 24);

  automix::engine::BatchQueueRunner runner;
  const auto items = runner.buildItemsFromFolder(inputDir, outputDir);
  REQUIRE(items.size() == 2);

  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
}

TEST_CASE("Batch runner processes synthetic batch job with BuiltIn renderer", "[batch]") {
  const std::filesystem::path inputDir = std::filesystem::temp_directory_path() / "automix_batch_process_input";
  const std::filesystem::path outputDir = std::filesystem::temp_directory_path() / "automix_batch_process_output";
  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
  std::filesystem::create_directories(inputDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(inputDir / "songc_vocals.wav", makeTone(44100.0, 8192, 260.0), 24);
  writer.write(inputDir / "songc_bass.wav", makeTone(44100.0, 8192, 90.0), 24);

  automix::engine::BatchQueueRunner runner;
  auto items = runner.buildItemsFromFolder(inputDir, outputDir);
  REQUIRE(items.empty() == false);

  automix::domain::BatchJob job;
  job.items = std::move(items);
  job.settings.outputFolder = outputDir;
  job.settings.analysisThreads = 1;
  job.settings.parallelAnalysis = false;
  job.settings.renderSettings.outputSampleRate = 44100;
  job.settings.renderSettings.blockSize = 512;
  job.settings.renderSettings.outputBitDepth = 24;
  job.settings.renderSettings.rendererName = "BuiltIn";

  const auto result = runner.process(job, {}, nullptr);
  REQUIRE(result.completed >= 1);

  bool foundOutput = false;
  for (const auto& item : job.items) {
    if (std::filesystem::exists(item.outputPath)) {
      foundOutput = true;
      break;
    }
  }
  REQUIRE(foundOutput);

  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
}

TEST_CASE("Batch runner supports renderer chain settings", "[batch]") {
  const std::filesystem::path inputDir = std::filesystem::temp_directory_path() / "automix_batch_chain_input";
  const std::filesystem::path outputDir = std::filesystem::temp_directory_path() / "automix_batch_chain_output";
  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
  std::filesystem::create_directories(inputDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(inputDir / "songchain_vocals.wav", makeTone(44100.0, 8192, 260.0), 24);
  writer.write(inputDir / "songchain_bass.wav", makeTone(44100.0, 8192, 90.0), 24);

  automix::engine::BatchQueueRunner runner;
  auto items = runner.buildItemsFromFolder(inputDir, outputDir);
  REQUIRE_FALSE(items.empty());

  automix::domain::BatchJob job;
  job.items = std::move(items);
  job.settings.outputFolder = outputDir;
  job.settings.analysisThreads = 1;
  job.settings.parallelAnalysis = false;
  job.settings.renderSettings.outputSampleRate = 44100;
  job.settings.renderSettings.blockSize = 512;
  job.settings.renderSettings.outputBitDepth = 24;
  job.settings.renderSettings.rendererName = "BuiltIn";
  job.settings.renderSettings.rendererChainEnabled = true;
  job.settings.renderSettings.rendererChainMode = "master_then_rsgain";

  const auto result = runner.process(job, {}, nullptr);
  REQUIRE(result.completed >= 1);
  REQUIRE(std::filesystem::exists(job.items.front().outputPath));

  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
}

TEST_CASE("Batch runner applies requested lossy output extension", "[batch]") {
  const std::filesystem::path inputDir = std::filesystem::temp_directory_path() / "automix_batch_lossy_input";
  const std::filesystem::path outputDir = std::filesystem::temp_directory_path() / "automix_batch_lossy_output";
  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
  std::filesystem::create_directories(inputDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(inputDir / "songd_vocals.wav", makeTone(44100.0, 4096, 260.0), 24);
  writer.write(inputDir / "songd_bass.wav", makeTone(44100.0, 4096, 90.0), 24);

  automix::engine::BatchQueueRunner runner;
  auto items = runner.buildItemsFromFolder(inputDir, outputDir);
  REQUIRE(items.empty() == false);

  automix::domain::BatchJob job;
  job.items = std::move(items);
  job.settings.outputFolder = outputDir;
  job.settings.analysisThreads = 1;
  job.settings.parallelAnalysis = false;
  job.settings.renderSettings.outputSampleRate = 44100;
  job.settings.renderSettings.blockSize = 512;
  job.settings.renderSettings.outputBitDepth = 24;
  job.settings.renderSettings.rendererName = "BuiltIn";
  job.settings.renderSettings.outputFormat = "mp3";
  job.settings.renderSettings.lossyBitrateKbps = 192;
  job.settings.renderSettings.lossyQuality = 7;

  const auto result = runner.process(job, {}, nullptr);
  REQUIRE(result.failed + result.completed + result.cancelled == static_cast<int>(job.items.size()));
  REQUIRE(job.items.front().outputPath.extension().string() == ".mp3");
  REQUIRE(std::regex_match(job.items.front().outputPath.filename().string(),
                           std::regex(R"(songd_AutoMixMaster_[0-9]{8}_[0-9]{2}\.mp3)")));

  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
}

TEST_CASE("Batch output filename uses song title prefix with date stamp only", "[batch]") {
  const std::filesystem::path inputDir = std::filesystem::temp_directory_path() / "automix_batch_naming_input";
  const std::filesystem::path outputDir = std::filesystem::temp_directory_path() / "automix_batch_naming_output";
  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
  std::filesystem::create_directories(inputDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(inputDir / "songtitle_vocals.wav", makeTone(44100.0, 4096, 220.0), 24);
  writer.write(inputDir / "songtitle_bass.wav", makeTone(44100.0, 4096, 110.0), 24);

  automix::engine::BatchQueueRunner runner;
  const auto items = runner.buildItemsFromFolder(inputDir, outputDir);
  REQUIRE(items.size() == 1);
  REQUIRE(std::regex_match(items.front().outputPath.filename().string(),
                           std::regex(R"(songtitle_AutoMixMaster_[0-9]{8}_[0-9]{2}\.wav)")));

  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
}

TEST_CASE("Batch naming increments sequence when target filename already exists", "[batch]") {
  const std::filesystem::path inputDir = std::filesystem::temp_directory_path() / "automix_batch_sequence_input";
  const std::filesystem::path outputDir = std::filesystem::temp_directory_path() / "automix_batch_sequence_output";
  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
  std::filesystem::create_directories(inputDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(inputDir / "songseq_vocals.wav", makeTone(44100.0, 4096, 220.0), 24);
  writer.write(inputDir / "songseq_bass.wav", makeTone(44100.0, 4096, 110.0), 24);

  automix::engine::BatchQueueRunner runner;
  auto firstPass = runner.buildItemsFromFolder(inputDir, outputDir);
  REQUIRE(firstPass.size() == 1);
  REQUIRE(std::regex_match(firstPass.front().outputPath.filename().string(),
                           std::regex(R"(songseq_AutoMixMaster_[0-9]{8}_01\.wav)")));

  writer.write(firstPass.front().outputPath, makeTone(44100.0, 4096, 330.0), 24);

  auto secondPass = runner.buildItemsFromFolder(inputDir, outputDir);
  REQUIRE(secondPass.size() == 1);
  REQUIRE(std::regex_match(secondPass.front().outputPath.filename().string(),
                           std::regex(R"(songseq_AutoMixMaster_[0-9]{8}_02\.wav)")));

  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
}

TEST_CASE("Batch grouping supports recursive folder scan when enabled", "[batch]") {
  const std::filesystem::path inputDir = std::filesystem::temp_directory_path() / "automix_batch_recursive_input";
  const std::filesystem::path nestedDir = inputDir / "nested" / "album_a";
  const std::filesystem::path outputDir = std::filesystem::temp_directory_path() / "automix_batch_recursive_output";
  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
  std::filesystem::create_directories(nestedDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(nestedDir / "songx_vocals.wav", makeTone(44100.0, 4096, 300.0), 24);
  writer.write(nestedDir / "songx_bass.wav", makeTone(44100.0, 4096, 100.0), 24);

  automix::engine::BatchQueueRunner runner;
  const auto nonRecursiveItems = runner.buildItemsFromFolder(inputDir, outputDir, false);
  REQUIRE(nonRecursiveItems.empty());

  const auto recursiveItems = runner.buildItemsFromFolder(inputDir, outputDir, true);
  REQUIRE(recursiveItems.size() == 1);

  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
}

TEST_CASE("Batch grouping detects parenthesized role suffixes", "[batch]") {
  const std::filesystem::path inputDir = std::filesystem::temp_directory_path() / "automix_batch_parenthesized_input";
  const std::filesystem::path outputDir = std::filesystem::temp_directory_path() / "automix_batch_parenthesized_output";
  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
  std::filesystem::create_directories(inputDir);
  std::filesystem::create_directories(outputDir);

  automix::util::WavWriter writer;
  writer.write(inputDir / "adrift_(bass).wav", makeTone(44100.0, 4096, 120.0), 24);
  writer.write(inputDir / "adrift_(drums).wav", makeTone(44100.0, 4096, 240.0), 24);
  writer.write(inputDir / "adrift_(vocals).wav", makeTone(44100.0, 4096, 360.0), 24);

  automix::engine::BatchQueueRunner runner;
  const auto items = runner.buildItemsFromFolder(inputDir, outputDir);
  REQUIRE(items.size() == 1);
  REQUIRE(items.front().session.stems.size() == 3);

  std::filesystem::remove_all(inputDir);
  std::filesystem::remove_all(outputDir);
}
