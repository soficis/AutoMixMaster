#include "dsp/DynamicDeHarshEq.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace automix::dsp {
namespace {

struct OnePole {
  float a = 0.0f;
  float z = 0.0f;
  float process(const float input) {
    z += a * (input - z);
    return z;
  }
};

OnePole makeLowPass(const double sampleRate, const double cutoffHz) {
  const double x = std::exp(-2.0 * 3.14159265358979323846 * cutoffHz / sampleRate);
  OnePole filter;
  filter.a = static_cast<float>(1.0 - x);
  return filter;
}

} // namespace

void DynamicDeHarshEq::process(engine::AudioBuffer& buffer, const double strength) const {
  if (buffer.getNumChannels() == 0 || buffer.getNumSamples() == 0) {
    return;
  }

  const float amount = static_cast<float>(std::clamp(strength, 0.0, 1.0));
  if (amount <= 0.0f) {
    return;
  }

  std::array<OnePole, 2> lp2k;
  std::array<OnePole, 2> lp6k;
  std::array<float, 2> harshEnv = {0.0f, 0.0f};

  for (int ch = 0; ch < std::min(2, buffer.getNumChannels()); ++ch) {
    lp2k[static_cast<size_t>(ch)] = makeLowPass(buffer.getSampleRate(), 2000.0);
    lp6k[static_cast<size_t>(ch)] = makeLowPass(buffer.getSampleRate(), 6000.0);
  }

  constexpr float attack = 0.08f;
  constexpr float release = 0.01f;
  constexpr float threshold = 0.08f;

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      auto& low2 = lp2k[static_cast<size_t>(std::min(ch, 1))];
      auto& low6 = lp6k[static_cast<size_t>(std::min(ch, 1))];
      auto& env = harshEnv[static_cast<size_t>(std::min(ch, 1))];

      const float x = buffer.getSample(ch, i);
      const float b2 = low2.process(x);
      const float b6 = low6.process(x);
      const float harshBand = b6 - b2;
      const float harshAbs = std::abs(harshBand);

      const float coeff = harshAbs > env ? attack : release;
      env += coeff * (harshAbs - env);
      const float over = std::max(0.0f, env - threshold);
      const float gain = std::clamp(1.0f - over * amount * 4.0f, 0.25f, 1.0f);

      const float out = b2 + harshBand * gain + (x - b6);
      buffer.setSample(ch, i, out);
    }
  }
}

} // namespace automix::dsp
