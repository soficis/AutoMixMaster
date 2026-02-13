#include "dsp/DeEsser.h"

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

void DeEsser::process(engine::AudioBuffer& buffer, const double strength) const {
  if (buffer.getNumChannels() == 0 || buffer.getNumSamples() == 0) {
    return;
  }

  const float amount = static_cast<float>(std::clamp(strength, 0.0, 1.0));
  if (amount <= 0.0f) {
    return;
  }

  std::array<OnePole, 2> lp4k;
  std::array<OnePole, 2> lp10k;
  std::array<float, 2> fullEnv = {0.0f, 0.0f};
  std::array<float, 2> sibEnv = {0.0f, 0.0f};

  for (int ch = 0; ch < std::min(2, buffer.getNumChannels()); ++ch) {
    lp4k[static_cast<size_t>(ch)] = makeLowPass(buffer.getSampleRate(), 4000.0);
    lp10k[static_cast<size_t>(ch)] = makeLowPass(buffer.getSampleRate(), 10000.0);
  }

  constexpr float envelopeAttack = 0.1f;
  constexpr float envelopeRelease = 0.01f;
  constexpr float ratioThreshold = 0.28f;

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      auto& l4 = lp4k[static_cast<size_t>(std::min(ch, 1))];
      auto& l10 = lp10k[static_cast<size_t>(std::min(ch, 1))];
      auto& envFull = fullEnv[static_cast<size_t>(std::min(ch, 1))];
      auto& envSib = sibEnv[static_cast<size_t>(std::min(ch, 1))];

      const float x = buffer.getSample(ch, i);
      const float low4 = l4.process(x);
      const float low10 = l10.process(x);
      const float sibBand = low10 - low4;

      const float absFull = std::abs(x);
      const float absSib = std::abs(sibBand);

      const float fullCoeff = absFull > envFull ? envelopeAttack : envelopeRelease;
      const float sibCoeff = absSib > envSib ? envelopeAttack : envelopeRelease;
      envFull += fullCoeff * (absFull - envFull);
      envSib += sibCoeff * (absSib - envSib);

      const float ratio = envSib / std::max(envFull, 1.0e-6f);
      const float over = std::max(0.0f, ratio - ratioThreshold);
      const float gain = std::clamp(1.0f - over * amount * 2.5f, 0.35f, 1.0f);

      const float out = low4 + sibBand * gain + (x - low10);
      buffer.setSample(ch, i, out);
    }
  }
}

} // namespace automix::dsp
