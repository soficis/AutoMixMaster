#include <atomic>
#include <cmath>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "engine/AudioBuffer.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/TransportController.h"
#include "util/WavWriter.h"

namespace {

automix::engine::AudioBuffer makeTone(const double sampleRate, const int samples, const double freq) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const float sample = static_cast<float>(0.4 * std::sin(2.0 * 3.14159265358979323846 * freq * i / sampleRate));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

int countZeroCrossings(const automix::engine::AudioBuffer& buffer) {
  int crossings = 0;
  float previous = buffer.getSample(0, 0);
  for (int i = 1; i < buffer.getNumSamples(); ++i) {
    const float current = buffer.getSample(0, i);
    if ((previous <= 0.0f && current > 0.0f) || (previous >= 0.0f && current < 0.0f)) {
      ++crossings;
    }
    previous = current;
  }
  return crossings;
}

} // namespace

TEST_CASE("Audio IO reads identical sample count and close values", "[audioio]") {
  const auto input = makeTone(44100.0, 44100, 440.0);
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "automix_io_identity.wav";

  automix::util::WavWriter writer;
  writer.write(path, input, 24);

  automix::engine::AudioFileIO reader;
  const auto output = reader.readAudioFile(path);

  REQUIRE(output.getNumChannels() == 2);
  REQUIRE(output.getNumSamples() == input.getNumSamples());

  const float sampleDiff = std::abs(output.getSample(0, 1000) - input.getSample(0, 1000));
  REQUIRE(sampleDiff < 0.002f);

  std::filesystem::remove(path);
}

TEST_CASE("Resampling preserves tone behavior approximately", "[audioio]") {
  const auto input = makeTone(44100.0, 44100, 1000.0);

  automix::engine::AudioResampler resampler;
  const auto resampled = resampler.resampleLinear(input, 48000.0);

  REQUIRE(resampled.getNumSamples() > input.getNumSamples());

  const int inCrossings = countZeroCrossings(input);
  const int outCrossings = countZeroCrossings(resampled);
  const double ratio = static_cast<double>(outCrossings) / static_cast<double>(inCrossings);

  REQUIRE(ratio > 0.95);
  REQUIRE(ratio < 1.05);
}

TEST_CASE("Audio writer format resolution supports lossy formats", "[audioio]") {
  REQUIRE(automix::util::WavWriter::resolveFormat("mix.wav", "mp3") == "mp3");
  REQUIRE(automix::util::WavWriter::resolveFormat("mix.ogg", "auto") == "ogg");
  REQUIRE(automix::util::WavWriter::isLossyFormat("mp3"));
  REQUIRE(automix::util::WavWriter::isLossyFormat("ogg"));
  REQUIRE_FALSE(automix::util::WavWriter::isLossyFormat("wav"));
}

TEST_CASE("Preview bridge transport atomics round-trip without torn reads", "[audioio][previewbridge]") {
  constexpr int kIterations = 10000;
  automix::engine::TransportController transport;
  transport.setTimeline(48000, 48000.0);
  transport.play();

  std::atomic<bool> start{false};
  std::atomic<bool> done{false};
  std::vector<int64_t> observedPositions;
  std::atomic<bool> readerOk{true};

  std::thread writer([&] {
    while (!start.load())
      ;
    for (int i = 0; i <= kIterations; ++i) {
      transport.setPositionRealtime(i);
      if (i == kIterations / 2)
        transport.pause();
      if (i == kIterations / 2 + 1)
        transport.play();
    }
    done.store(true);
  });

  std::thread reader([&] {
    while (!start.load())
      ;
    while (!done.load() || observedPositions.size() < 1000) {
      const int64_t position = transport.positionSamples();
      observedPositions.push_back(position);
      if (position < 0 || position > kIterations)
        readerOk.store(false);
    }
  });

  start.store(true);
  writer.join();
  reader.join();

  // Writer's last store wins; both threads have joined, so this is final.
  REQUIRE(transport.positionSamples() == kIterations);
  REQUIRE(transport.isPlaying());

  // No torn reads: every observed position was a value the writer actually
  // published, and (single writer, x86/x64) the sequence never goes backwards.
  REQUIRE(readerOk.load());
  REQUIRE_FALSE(observedPositions.empty());
  for (size_t i = 1; i < observedPositions.size(); ++i)
    REQUIRE(observedPositions[i] >= observedPositions[i - 1]);
}

TEST_CASE("Preview bridge stopFromAudioThread publishes stopped state", "[audioio][previewbridge]") {
  automix::engine::TransportController transport;
  transport.setTimeline(48000, 48000.0);
  transport.play();
  REQUIRE(transport.isPlaying());

  transport.setPositionRealtime(12345);
  REQUIRE(transport.positionSamples() == 12345);

  transport.stopFromAudioThread();
  REQUIRE_FALSE(transport.isPlaying());
  REQUIRE(transport.positionSamples() == 0);
}

TEST_CASE("Preview bridge meter targets round-trip without torn reads", "[audioio][previewbridge]") {
  constexpr int kIterations = 5000;
  std::atomic<float> leftLevel{-60.0f};
  std::atomic<float> leftPeak{-60.0f};
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};
  std::vector<float> seenLevels;
  std::vector<float> seenPeaks;
  std::atomic<bool> readerOk{true};

  std::thread writer([&] {
    while (!start.load())
      ;
    for (int i = 0; i < kIterations; ++i) {
      const float value = static_cast<float>(i) / static_cast<float>(kIterations);
      leftLevel.store(value, std::memory_order_relaxed);
      leftPeak.store(2.0f * value, std::memory_order_relaxed);
    }
    done.store(true);
  });

  std::thread reader([&] {
    while (!start.load())
      ;
    while (!done.load() || seenLevels.size() < 500) {
      const float level = leftLevel.load(std::memory_order_relaxed);
      const float peak = leftPeak.load(std::memory_order_relaxed);
      seenLevels.push_back(level);
      seenPeaks.push_back(peak);
      if (level < 0.0f || level > 1.0f || peak < 0.0f || peak > 2.0f)
        readerOk.store(false);
    }
  });

  start.store(true);
  writer.join();
  reader.join();

  REQUIRE(readerOk.load());
  REQUIRE_FALSE(seenLevels.empty());
  REQUIRE(leftLevel.load() ==
          static_cast<float>(kIterations - 1) / static_cast<float>(kIterations));
  REQUIRE(leftPeak.load() ==
          2.0f * static_cast<float>(kIterations - 1) / static_cast<float>(kIterations));
}

TEST_CASE("Preview bridge buffer swap never tears and keeps last writer", "[audioio][previewbridge]") {
  constexpr int kIterations = 2000;
  std::atomic<std::shared_ptr<const automix::engine::AudioBuffer>> buffer{nullptr};
  std::atomic<bool> done{false};
  std::atomic<bool> readerOk{true};

  std::thread writer([&] {
    for (int i = 0; i < kIterations; ++i) {
      auto next =
          std::make_shared<automix::engine::AudioBuffer>(2, 256 + (i % 3), 44100.0);
      next->setSample(0, 0, static_cast<float>(i));
      buffer.store(std::move(next), std::memory_order_release);
    }
    done.store(true);
  });

  std::thread reader([&] {
    while (!done.load()) {
      const auto current = buffer.load(std::memory_order_acquire);
      if (current == nullptr)
        continue;
      // While holding the shared_ptr the published buffer must be fully valid
      // even if the writer swaps in a new one mid-read.
      const int numSamples = current->getNumSamples();
      const float firstSample = current->getSample(0, 0);
      if (numSamples < 256 || numSamples > 258)
        readerOk.store(false);
      if (firstSample < 0.0f || firstSample >= static_cast<float>(kIterations))
        readerOk.store(false);
    }
  });

  writer.join();
  reader.join();

  REQUIRE(readerOk.load());
  const auto finalBuffer = buffer.load();
  REQUIRE(finalBuffer != nullptr);
  REQUIRE(finalBuffer->getNumSamples() == 256 + ((kIterations - 1) % 3));
  const float lastSample = finalBuffer->getSample(0, 0);
  REQUIRE(std::abs(lastSample - static_cast<float>(kIterations - 1)) < 0.001f);
}
