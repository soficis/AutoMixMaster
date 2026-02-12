#include "engine/AudioFileIO.h"

#include <algorithm>
#include <stdexcept>

#include <juce_audio_formats/juce_audio_formats.h>

namespace automix::engine {

AudioBuffer AudioFileIO::readAudioFile(const std::filesystem::path& filePath) const {
  juce::AudioFormatManager manager;
  manager.registerBasicFormats();

  const juce::File juceFile(filePath.string());
  std::unique_ptr<juce::AudioFormatReader> reader(manager.createReaderFor(juceFile));
  if (reader == nullptr) {
    throw std::runtime_error("Failed to open audio file: " + filePath.string());
  }

  const int channels = static_cast<int>(reader->numChannels);
  const int samples = static_cast<int>(reader->lengthInSamples);
  AudioBuffer output(channels, samples, reader->sampleRate);

  juce::AudioBuffer<float> juceBuffer(channels, samples);
  const bool success = reader->read(&juceBuffer, 0, samples, 0, true, true);
  if (!success) {
    throw std::runtime_error("Failed to read audio data: " + filePath.string());
  }

  for (int channel = 0; channel < channels; ++channel) {
    const float* src = juceBuffer.getReadPointer(channel);
    float* dst = output.getWritePointer(channel);
    std::copy(src, src + samples, dst);
  }

  return output;
}

} // namespace automix::engine
