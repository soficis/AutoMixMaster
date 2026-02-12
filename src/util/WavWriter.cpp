#include "util/WavWriter.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include <juce_audio_formats/juce_audio_formats.h>

namespace automix::util {

void WavWriter::write(const std::filesystem::path& path, const engine::AudioBuffer& buffer, const int bitDepth) const {
  juce::File outputFile(path.string());
  outputFile.deleteFile();

  std::unique_ptr<juce::FileOutputStream> stream(outputFile.createOutputStream());
  if (stream == nullptr || !stream->openedOk()) {
    throw std::runtime_error("Failed to open output WAV file: " + path.string());
  }

  juce::WavAudioFormat format;
  std::unique_ptr<juce::AudioFormatWriter> writer(
      format.createWriterFor(stream.get(),
                             buffer.getSampleRate(),
                             static_cast<unsigned int>(buffer.getNumChannels()),
                             bitDepth,
                             {},
                             0));
  if (writer == nullptr) {
    throw std::runtime_error("Failed to create WAV writer for: " + path.string());
  }
  stream.release();

  juce::AudioBuffer<float> juceBuffer(buffer.getNumChannels(), buffer.getNumSamples());
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    float* dst = juceBuffer.getWritePointer(ch);
    const float* src = buffer.getReadPointer(ch);
    std::copy(src, src + buffer.getNumSamples(), dst);
  }

  writer->writeFromAudioSampleBuffer(juceBuffer, 0, juceBuffer.getNumSamples());
}

} // namespace automix::util
