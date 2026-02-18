#include "dsp/MidSideProcessor.h"

#include <algorithm>
#include <cmath>

namespace automix::dsp {

void MidSideProcessor::process(engine::AudioBuffer& buffer, const double monoBelowHz, const double width) const {
  if (buffer.getNumChannels() < 2 || buffer.getNumSamples() == 0) {
    return;
  }

  const double cutoff = std::clamp(monoBelowHz, 20.0, 400.0);
  const float widthGain = static_cast<float>(std::clamp(width, 0.0, 1.8));

  const double x = std::exp(-2.0 * 3.14159265358979323846 * cutoff / buffer.getSampleRate());
  const float lpCoeff = static_cast<float>(1.0 - x);

  float lowSide = 0.0f;
  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    const float left = buffer.getSample(0, i);
    const float right = buffer.getSample(1, i);

    const float mid = 0.5f * (left + right);
    const float side = 0.5f * (left - right);

    lowSide += lpCoeff * (side - lowSide);
    const float highSide = side - lowSide;
    const float processedSide = highSide * widthGain;

    buffer.setSample(0, i, mid + processedSide);
    buffer.setSample(1, i, mid - processedSide);
  }
}

} // namespace automix::dsp
