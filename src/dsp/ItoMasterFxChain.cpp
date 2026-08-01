#include "dsp/ItoMasterFxChain.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <juce_dsp/juce_dsp.h>

namespace automix::dsp {
namespace {

using Filter = juce::dsp::IIR::Filter<float>;
using Coeffs = juce::dsp::IIR::Coefficients<float>;

constexpr float kTinyLevel = 1.0e-9f;

double envelopeCoefficient(const double sampleRate, const double timeMs) {
  const double seconds = std::max(0.1, timeMs) / 1000.0;
  return std::exp(-1.0 / std::max(1.0, sampleRate * seconds));
}

void processBandDynamics(double sampleRate,
                         const ItoCompressorBandSettings& settings,
                         std::vector<std::vector<float>>& band) {
  const int channels = static_cast<int>(band.size());
  if (channels == 0) {
    return;
  }
  const int samples = static_cast<int>(band[0].size());
  if (samples == 0) {
    return;
  }

  const double attackCoeff = envelopeCoefficient(sampleRate, settings.attackMs);
  const double releaseCoeff = envelopeCoefficient(sampleRate, settings.releaseMs);
  const double compRatio = std::max(1.000001, settings.compRatio);
  const double expRatio = std::clamp(settings.expRatio, 1.0e-6, 0.999999);

  double gain = 1.0;
  for (int i = 0; i < samples; ++i) {
    float peak = 0.0f;
    for (int ch = 0; ch < channels; ++ch) {
      peak = std::max(peak, std::abs(band[static_cast<size_t>(ch)][static_cast<size_t>(i)]));
    }

    const double levelDb = 20.0 * std::log10(std::max(peak, kTinyLevel));
    double reductionDb = 0.0;
    if (levelDb > settings.compThresholdDb) {
      reductionDb += (levelDb - settings.compThresholdDb) * (1.0 - 1.0 / compRatio);
    }
    if (levelDb < settings.expThresholdDb) {
      reductionDb += (settings.expThresholdDb - levelDb) * (1.0 - expRatio);
    }

    const double targetGain = std::pow(10.0, -reductionDb / 20.0);
    if (targetGain < gain) {
      gain = targetGain + attackCoeff * (gain - targetGain);
    } else {
      gain = targetGain + releaseCoeff * (gain - targetGain);
    }

    for (int ch = 0; ch < channels; ++ch) {
      band[static_cast<size_t>(ch)][static_cast<size_t>(i)] *= static_cast<float>(gain);
    }
  }
}

} // namespace

void ItoMasterFxChain::processEq(engine::AudioBuffer& buffer, const ItoEqSettings& settings) {
  if (!settings.enabled) {
    return;
  }
  const int channels = buffer.getNumChannels();
  const int samples = buffer.getNumSamples();
  const double sampleRate = buffer.getSampleRate();
  if (channels == 0 || samples == 0 || sampleRate <= 0.0) {
    return;
  }

  // Six sections in the ITO-Master eq order: low shelf, band0..band3 (peaking),
  // high shelf. Each channel gets its own IIR::Filter instance (constructed with
  // the coefficients so the filter order is derived correctly).
  enum class SectionKind { LowShelf, Peak, HighShelf };
  struct SectionSpec {
    SectionKind kind;
    ItoEqBandSettings band;
  };
  const SectionSpec sections[6] = {
      {SectionKind::LowShelf, settings.lowShelf},
      {SectionKind::Peak, settings.bands[0]},
      {SectionKind::Peak, settings.bands[1]},
      {SectionKind::Peak, settings.bands[2]},
      {SectionKind::Peak, settings.bands[3]},
      {SectionKind::HighShelf, settings.highShelf},
  };

  const auto makeCoefficients = [&](const SectionSpec& spec) -> Coeffs::Ptr {
    // The JUCE factory gain argument is a LINEAR factor, not decibels; a raw
    // dB value of 0.0 would be clamped to the -100 dB floor and silence the
    // section, so convert first.
    const float gainLinear = static_cast<float>(std::pow(10.0, spec.band.gainDb / 20.0));
    const float freq = static_cast<float>(spec.band.freqHz);
    const float qFactor = static_cast<float>(spec.band.qFactor);
    switch (spec.kind) {
      case SectionKind::LowShelf:
        return Coeffs::makeLowShelf(sampleRate, freq, qFactor, gainLinear);
      case SectionKind::Peak:
        return Coeffs::makePeakFilter(sampleRate, freq, qFactor, gainLinear);
      case SectionKind::HighShelf:
        return Coeffs::makeHighShelf(sampleRate, freq, qFactor, gainLinear);
    }
    return Coeffs::makePeakFilter(sampleRate, 1000.0f, 1.0f, 1.0f);
  };

  juce::dsp::ProcessSpec monoSpec;
  monoSpec.sampleRate = sampleRate;
  monoSpec.maximumBlockSize = 512;
  monoSpec.numChannels = 1;

  std::vector<std::vector<Filter>> banks(6);
  for (int section = 0; section < 6; ++section) {
    const auto coefficients = makeCoefficients(sections[section]);
    auto& bank = banks[static_cast<size_t>(section)];
    bank.reserve(static_cast<size_t>(channels));
    for (int ch = 0; ch < channels; ++ch) {
      bank.emplace_back(coefficients);
      bank.back().prepare(monoSpec);
    }
  }

  for (int section = 0; section < 6; ++section) {
    auto& bank = banks[static_cast<size_t>(section)];
    for (int i = 0; i < samples; ++i) {
      for (int ch = 0; ch < channels; ++ch) {
        buffer.setSample(ch, i, bank[static_cast<size_t>(ch)].processSample(buffer.getSample(ch, i)));
      }
    }
  }
}

void ItoMasterFxChain::processDistortion(engine::AudioBuffer& buffer, const ItoDistortionSettings& settings) {
  if (!settings.enabled) {
    return;
  }
  const int channels = buffer.getNumChannels();
  const int samples = buffer.getNumSamples();
  if (channels == 0 || samples == 0) {
    return;
  }

  const double drive = std::pow(10.0, std::clamp(settings.driveDb, 0.0, 24.0) / 20.0);
  const float dryWeight = static_cast<float>(std::clamp(1.0 - settings.parallelWeight, 0.0, 1.0));
  const float wetWeight = static_cast<float>(std::clamp(settings.parallelWeight, 0.0, 1.0));

  for (int i = 0; i < samples; ++i) {
    for (int ch = 0; ch < channels; ++ch) {
      const float input = buffer.getSample(ch, i);
      const float saturated = static_cast<float>(std::tanh(drive * static_cast<double>(input)));
      buffer.setSample(ch, i, dryWeight * input + wetWeight * saturated);
    }
  }
}

void ItoMasterFxChain::processMultiband(engine::AudioBuffer& buffer,
                                        const double sampleRate,
                                        const ItoMultibandCompSettings& settings) {
  if (!settings.enabled) {
    return;
  }
  const int channels = buffer.getNumChannels();
  const int samples = buffer.getNumSamples();
  if (channels == 0 || samples == 0 || sampleRate <= 0.0) {
    return;
  }

  juce::dsp::ProcessSpec spec;
  spec.sampleRate = sampleRate;
  spec.maximumBlockSize = static_cast<juce::uint32>(std::max(1, samples));
  spec.numChannels = static_cast<juce::uint32>(channels);

  // LR4 crossovers (4th order, -24 dB/octave). For a Linkwitz-Riley pair the
  // low/high magnitudes sum to unity at every frequency, so the three bands
  // reconstruct the input (magnitude-wise) exactly.
  juce::dsp::LinkwitzRileyFilter<float> lowCrossover;
  juce::dsp::LinkwitzRileyFilter<float> highCrossover;
  lowCrossover.setType(juce::dsp::LinkwitzRileyFilter<float>::Type::lowpass);
  lowCrossover.setCutoffFrequency(static_cast<float>(settings.lowCrossoverHz));
  highCrossover.setType(juce::dsp::LinkwitzRileyFilter<float>::Type::lowpass);
  highCrossover.setCutoffFrequency(static_cast<float>(settings.highCrossoverHz));
  lowCrossover.prepare(spec);
  lowCrossover.reset();
  highCrossover.prepare(spec);
  highCrossover.reset();

  std::vector<std::vector<float>> low(static_cast<size_t>(channels));
  std::vector<std::vector<float>> mid(static_cast<size_t>(channels));
  std::vector<std::vector<float>> high(static_cast<size_t>(channels));
  for (int ch = 0; ch < channels; ++ch) {
    low[static_cast<size_t>(ch)].resize(static_cast<size_t>(samples), 0.0f);
    mid[static_cast<size_t>(ch)].resize(static_cast<size_t>(samples), 0.0f);
    high[static_cast<size_t>(ch)].resize(static_cast<size_t>(samples), 0.0f);
  }

  // low  = LP(f1)(x)
  // mid  = LP(f2)(HP(f1)(x))
  // high = HP(f2)(HP(f1)(x))
  for (int i = 0; i < samples; ++i) {
    for (int ch = 0; ch < channels; ++ch) {
      const float input = buffer.getSample(ch, i);
      float lowOut = 0.0f;
      float midPre = 0.0f;
      lowCrossover.processSample(ch, input, lowOut, midPre);
      float midOut = 0.0f;
      float highOut = 0.0f;
      highCrossover.processSample(ch, midPre, midOut, highOut);
      low[static_cast<size_t>(ch)][static_cast<size_t>(i)] = lowOut;
      mid[static_cast<size_t>(ch)][static_cast<size_t>(i)] = midOut;
      high[static_cast<size_t>(ch)][static_cast<size_t>(i)] = highOut;
    }
  }

  processBandDynamics(sampleRate, settings.bands[0], low);
  processBandDynamics(sampleRate, settings.bands[1], mid);
  processBandDynamics(sampleRate, settings.bands[2], high);

  const float dryWeight = static_cast<float>(std::clamp(1.0 - settings.parallelWeight, 0.0, 1.0));
  const float wetWeight = static_cast<float>(std::clamp(settings.parallelWeight, 0.0, 1.0));

  for (int i = 0; i < samples; ++i) {
    for (int ch = 0; ch < channels; ++ch) {
      const size_t channel = static_cast<size_t>(ch);
      const size_t sample = static_cast<size_t>(i);
      const float dry = buffer.getSample(ch, i);
      const float wet = low[channel][sample] + mid[channel][sample] + high[channel][sample];
      buffer.setSample(ch, i, dryWeight * dry + wetWeight * wet);
    }
  }
}

void ItoMasterFxChain::processGain(engine::AudioBuffer& buffer, const ItoGainSettings& settings) {
  if (!settings.enabled) {
    return;
  }
  buffer.applyGain(static_cast<float>(std::pow(10.0, settings.gainDb / 20.0)));
}

void ItoMasterFxChain::processImager(engine::AudioBuffer& buffer, const ItoImagerSettings& settings) {
  if (!settings.enabled) {
    return;
  }
  const int channels = buffer.getNumChannels();
  const int samples = buffer.getNumSamples();
  if (channels < 2 || samples == 0) {
    return;
  }

  const float width = static_cast<float>(std::clamp(settings.width, 0.0, 2.0));
  for (int i = 0; i < samples; ++i) {
    const float left = buffer.getSample(0, i);
    const float right = buffer.getSample(1, i);
    const float mid = 0.5f * (left + right);
    const float side = 0.5f * (left - right);
    buffer.setSample(0, i, mid + width * side);
    buffer.setSample(1, i, mid - width * side);
  }
}

void ItoMasterFxChain::processLimiter(engine::AudioBuffer& buffer,
                                      const double sampleRate,
                                      const ItoLimiterSettings& settings) {
  if (!settings.enabled) {
    return;
  }
  const int channels = buffer.getNumChannels();
  const int samples = buffer.getNumSamples();
  if (channels == 0 || samples == 0 || sampleRate <= 0.0) {
    return;
  }

  const double ceiling = std::pow(10.0, std::clamp(settings.thresholdDb, -60.0, -1.0e-6) / 20.0);
  const int lookahead =
      std::max(1, static_cast<int>(std::lround(std::max(0.0, settings.lookaheadMs) * sampleRate / 1000.0)));
  const double attackCoeff = envelopeCoefficient(sampleRate, settings.attackMs);
  const double releaseCoeff = envelopeCoefficient(sampleRate, settings.releaseMs);
  const float ceilingFloat = static_cast<float>(ceiling);

  // Per-channel lookahead delay lines (ring buffers).
  std::vector<std::vector<float>> delayLines(static_cast<size_t>(channels),
                                             std::vector<float>(static_cast<size_t>(lookahead), 0.0f));
  std::vector<int> writeIndex(static_cast<size_t>(channels), 0);

  double envelope = 0.0;
  for (int i = 0; i < samples; ++i) {
    float peak = 0.0f;
    for (int ch = 0; ch < channels; ++ch) {
      peak = std::max(peak, std::abs(buffer.getSample(ch, i)));
    }

    if (peak > envelope) {
      envelope = peak + attackCoeff * (envelope - peak);
    } else {
      envelope = peak + releaseCoeff * (envelope - peak);
    }

    const float gain = envelope > ceiling ? static_cast<float>(ceiling / envelope) : 1.0f;

    for (int ch = 0; ch < channels; ++ch) {
      auto& line = delayLines[static_cast<size_t>(ch)];
      auto& index = writeIndex[static_cast<size_t>(ch)];
      const float input = buffer.getSample(ch, i);
      line[static_cast<size_t>(index)] = input;
      index = (index + 1) % lookahead;
      const float delayed = line[static_cast<size_t>(index)];
      const float limited = std::clamp(delayed * gain, -ceilingFloat, ceilingFloat);
      buffer.setSample(ch, i, limited);
    }
  }
}

void ItoMasterFxChain::prepare(const double sampleRate, const int numChannels) {
  sampleRate_ = sampleRate > 0.0 ? sampleRate : 44100.0;
  numChannels_ = std::max(1, numChannels);
  settings_ = ItoMasterChainSettings{};
}

void ItoMasterFxChain::reset() {}

void ItoMasterFxChain::setSettings(const ItoMasterChainSettings& settings) { settings_ = settings; }

void ItoMasterFxChain::process(engine::AudioBuffer& buffer) {
  processEq(buffer, settings_.eq);
  processDistortion(buffer, settings_.distortion);
  processMultiband(buffer, sampleRate_, settings_.multiband);
  processGain(buffer, settings_.gain);
  processImager(buffer, settings_.imager);
  processLimiter(buffer, sampleRate_, settings_.limiter);
}

} // namespace automix::dsp
