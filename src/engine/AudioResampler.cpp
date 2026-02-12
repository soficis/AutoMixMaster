#include "engine/AudioResampler.h"

#include <algorithm>
#include <cmath>

namespace automix::engine {

AudioBuffer AudioResampler::resampleLinear(const AudioBuffer& input, const double targetSampleRate) const {
  const double sourceSampleRate = input.getSampleRate();
  if (std::abs(sourceSampleRate - targetSampleRate) < 1e-6) {
    return input;
  }

  const int inputSamples = input.getNumSamples();
  const int outputSamples = static_cast<int>(std::round(static_cast<double>(inputSamples) * targetSampleRate / sourceSampleRate));

  AudioBuffer output(input.getNumChannels(), std::max(1, outputSamples), targetSampleRate);

  for (int channel = 0; channel < input.getNumChannels(); ++channel) {
    for (int i = 0; i < output.getNumSamples(); ++i) {
      const double position = static_cast<double>(i) * sourceSampleRate / targetSampleRate;
      const int indexA = static_cast<int>(std::floor(position));
      const int indexB = std::min(indexA + 1, inputSamples - 1);
      const float sampleA = input.getSample(channel, std::clamp(indexA, 0, inputSamples - 1));
      const float sampleB = input.getSample(channel, indexB);
      const float alpha = static_cast<float>(position - std::floor(position));
      const float value = sampleA + (sampleB - sampleA) * alpha;
      output.setSample(channel, i, value);
    }
  }

  return output;
}

} // namespace automix::engine
