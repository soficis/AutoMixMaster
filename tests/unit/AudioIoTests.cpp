#include <cmath>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
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
