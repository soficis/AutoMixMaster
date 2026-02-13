#include "dsp/SoftClipper.h"

#include <algorithm>
#include <cmath>

namespace automix::dsp {

void SoftClipper::process(engine::AudioBuffer& buffer, const double drive) const {
  const double clippedDrive = std::clamp(drive, 0.1, 8.0);
  const double normalizer = std::tanh(clippedDrive);
  if (std::abs(normalizer) < 1.0e-9) {
    return;
  }

  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const double shaped = std::tanh(static_cast<double>(buffer.getSample(ch, i)) * clippedDrive) / normalizer;
      buffer.setSample(ch, i, static_cast<float>(shaped));
    }
  }
}

} // namespace automix::dsp
