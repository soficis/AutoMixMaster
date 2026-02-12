#include "engine/AudioBuffer.h"

#include <algorithm>

namespace automix::engine {

AudioBuffer::AudioBuffer(const int channels, const int samples, const double sampleRate)
    : sampleRate_(sampleRate), channels_(static_cast<size_t>(channels), std::vector<float>(static_cast<size_t>(samples), 0.0f)) {}

int AudioBuffer::getNumChannels() const { return static_cast<int>(channels_.size()); }

int AudioBuffer::getNumSamples() const {
  if (channels_.empty()) {
    return 0;
  }
  return static_cast<int>(channels_.front().size());
}

double AudioBuffer::getSampleRate() const { return sampleRate_; }

float AudioBuffer::getSample(const int channel, const int sampleIndex) const {
  return channels_.at(static_cast<size_t>(channel)).at(static_cast<size_t>(sampleIndex));
}

void AudioBuffer::setSample(const int channel, const int sampleIndex, const float value) {
  channels_.at(static_cast<size_t>(channel)).at(static_cast<size_t>(sampleIndex)) = value;
}

float* AudioBuffer::getWritePointer(const int channel) {
  return channels_.at(static_cast<size_t>(channel)).data();
}

const float* AudioBuffer::getReadPointer(const int channel) const {
  return channels_.at(static_cast<size_t>(channel)).data();
}

void AudioBuffer::clear() {
  for (auto& channel : channels_) {
    std::fill(channel.begin(), channel.end(), 0.0f);
  }
}

void AudioBuffer::applyGain(const float linearGain) {
  for (auto& channel : channels_) {
    for (float& sample : channel) {
      sample *= linearGain;
    }
  }
}

} // namespace automix::engine
