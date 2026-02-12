#pragma once

#include <cstddef>
#include <vector>

namespace automix::engine {

class AudioBuffer {
 public:
  AudioBuffer() = default;
  AudioBuffer(int channels, int samples, double sampleRate);

  int getNumChannels() const;
  int getNumSamples() const;
  double getSampleRate() const;

  float getSample(int channel, int sampleIndex) const;
  void setSample(int channel, int sampleIndex, float value);
  float* getWritePointer(int channel);
  const float* getReadPointer(int channel) const;

  void clear();
  void applyGain(float linearGain);

 private:
  double sampleRate_ = 44100.0;
  std::vector<std::vector<float>> channels_;
};

} // namespace automix::engine
