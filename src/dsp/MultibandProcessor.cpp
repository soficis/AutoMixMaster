#include "dsp/MultibandProcessor.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace automix::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

double lowPassAlpha(const double sampleRate, const double cutoffHz) {
  const double clampedCutoff = std::clamp(cutoffHz, 20.0, sampleRate * 0.45);
  return 1.0 - std::exp(-2.0 * kPi * clampedCutoff / sampleRate);
}

void applyStereoWidth(engine::AudioBuffer& bandBuffer, const double width) {
  if (bandBuffer.getNumChannels() < 2 || std::abs(width - 1.0) < 1.0e-6) {
    return;
  }

  const float widthGain = static_cast<float>(std::clamp(width, 0.0, 2.0));
  for (int i = 0; i < bandBuffer.getNumSamples(); ++i) {
    const float l = bandBuffer.getSample(0, i);
    const float r = bandBuffer.getSample(1, i);
    const float mid = 0.5f * (l + r);
    const float side = 0.5f * (l - r) * widthGain;
    bandBuffer.setSample(0, i, mid + side);
    bandBuffer.setSample(1, i, mid - side);
  }
}

void applyBandCompressor(engine::AudioBuffer& bandBuffer, const domain::MultibandBandSettings& settings) {
  if (!settings.enabled || bandBuffer.getNumSamples() == 0 || bandBuffer.getNumChannels() == 0) {
    return;
  }

  const float threshold = static_cast<float>(dbToLinear(std::clamp(settings.thresholdDb, -60.0, 0.0)));
  const float ratio = static_cast<float>(std::clamp(settings.ratio, 1.0, 20.0));
  const float attackCoeff = static_cast<float>(std::exp(-1.0 / std::max(1.0, bandBuffer.getSampleRate() * 0.008)));
  const float releaseCoeff = static_cast<float>(std::exp(-1.0 / std::max(1.0, bandBuffer.getSampleRate() * 0.120)));
  const float makeup = static_cast<float>(dbToLinear(std::clamp(settings.makeupGainDb, -18.0, 18.0)));

  float envelope = 0.0f;
  for (int i = 0; i < bandBuffer.getNumSamples(); ++i) {
    float detector = 0.0f;
    for (int ch = 0; ch < bandBuffer.getNumChannels(); ++ch) {
      detector = std::max(detector, std::abs(bandBuffer.getSample(ch, i)));
    }

    if (detector > envelope) {
      envelope = detector + attackCoeff * (envelope - detector);
    } else {
      envelope = detector + releaseCoeff * (envelope - detector);
    }

    float gain = 1.0f;
    if (envelope > threshold && threshold > 0.0f) {
      const float over = envelope / threshold;
      const float compressed = std::pow(over, 1.0f / ratio);
      gain = 1.0f / std::max(compressed, 1.0e-6f);
    }

    const float finalGain = gain * makeup;
    for (int ch = 0; ch < bandBuffer.getNumChannels(); ++ch) {
      bandBuffer.setSample(ch, i, bandBuffer.getSample(ch, i) * finalGain);
    }
  }
}

} // namespace

void MultibandProcessor::process(engine::AudioBuffer& buffer, const domain::MultibandSettings& settings) const {
  if (buffer.getNumChannels() <= 0 || buffer.getNumSamples() <= 0) {
    return;
  }

  const int crossoverCount = static_cast<int>(settings.crossoverHz.size());
  if (crossoverCount <= 0) {
    return;
  }

  const int bandCount = crossoverCount + 1;
  std::vector<engine::AudioBuffer> bands;
  bands.reserve(static_cast<size_t>(bandCount));
  for (int i = 0; i < bandCount; ++i) {
    bands.emplace_back(buffer.getNumChannels(), buffer.getNumSamples(), buffer.getSampleRate());
  }

  std::vector<double> sortedCrossovers = settings.crossoverHz;
  std::sort(sortedCrossovers.begin(), sortedCrossovers.end());
  sortedCrossovers.erase(std::unique(sortedCrossovers.begin(), sortedCrossovers.end()), sortedCrossovers.end());

  const int effectiveCrossovers = static_cast<int>(sortedCrossovers.size());
  if (effectiveCrossovers <= 0) {
    return;
  }

  std::vector<double> alphaByBand(static_cast<size_t>(effectiveCrossovers), 0.0);
  for (int i = 0; i < effectiveCrossovers; ++i) {
    alphaByBand[static_cast<size_t>(i)] = lowPassAlpha(buffer.getSampleRate(), sortedCrossovers[static_cast<size_t>(i)]);
  }

  std::vector<std::vector<double>> lpState(static_cast<size_t>(effectiveCrossovers),
                                           std::vector<double>(static_cast<size_t>(buffer.getNumChannels()), 0.0));

  for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      double remainder = buffer.getSample(ch, sample);
      for (int crossover = 0; crossover < effectiveCrossovers; ++crossover) {
        auto& state = lpState[static_cast<size_t>(crossover)][static_cast<size_t>(ch)];
        state += alphaByBand[static_cast<size_t>(crossover)] * (remainder - state);
        const float lowBand = static_cast<float>(state);
        bands[static_cast<size_t>(crossover)].setSample(ch, sample, lowBand);
        remainder -= static_cast<double>(lowBand);
      }
      bands[static_cast<size_t>(effectiveCrossovers)].setSample(ch, sample, static_cast<float>(remainder));
    }
  }

  for (int i = 0; i < static_cast<int>(bands.size()); ++i) {
    const domain::MultibandBandSettings bandSettings =
        i < static_cast<int>(settings.bands.size()) ? settings.bands[static_cast<size_t>(i)] : domain::MultibandBandSettings{};
    applyBandCompressor(bands[static_cast<size_t>(i)], bandSettings);
    applyStereoWidth(bands[static_cast<size_t>(i)], bandSettings.width);
  }

  buffer.clear();
  for (const auto& band : bands) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        buffer.setSample(ch, i, buffer.getSample(ch, i) + band.getSample(ch, i));
      }
    }
  }
}

} // namespace automix::dsp
