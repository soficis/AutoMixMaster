#include "engine/AudioFileIO.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>

#include <juce_audio_formats/juce_audio_formats.h>

namespace automix::engine {

namespace {

std::unique_ptr<juce::AudioFormatReader> createReader(const std::filesystem::path& filePath,
                                                       juce::AudioFormatManager* managerOut) {
  managerOut->registerBasicFormats();
  const juce::File juceFile(filePath.string());
  return std::unique_ptr<juce::AudioFormatReader>(managerOut->createReaderFor(juceFile));
}

} // namespace

AudioBuffer AudioFileIO::readAudioFile(const std::filesystem::path& filePath) const {
  juce::AudioFormatManager manager;
  std::unique_ptr<juce::AudioFormatReader> reader = createReader(filePath, &manager);
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

std::map<std::string, std::string> AudioFileIO::readMetadata(const std::filesystem::path& filePath) const {
  juce::AudioFormatManager manager;
  std::unique_ptr<juce::AudioFormatReader> reader = createReader(filePath, &manager);
  if (reader == nullptr) {
    throw std::runtime_error("Failed to open audio file metadata: " + filePath.string());
  }

  std::map<std::string, std::string> metadata;
  const auto keys = reader->metadataValues.getAllKeys();
  for (const auto& key : keys) {
    const auto value = reader->metadataValues.getValue(key, {});
    if (value.isNotEmpty()) {
      metadata[key.toStdString()] = value.toStdString();
    }
  }
  return metadata;
}

} // namespace automix::engine
