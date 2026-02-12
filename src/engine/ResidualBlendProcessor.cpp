#include "engine/ResidualBlendProcessor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dsp/SignalMath.h"

namespace automix::engine {
namespace {

constexpr int kDefaultChannels = 2;

bool isFiniteSample(const float value) {
  return std::isfinite(static_cast<double>(value));
}

AudioBuffer toStereoBuffer(const AudioBuffer& input) {
  const int samples = input.getNumSamples();
  AudioBuffer stereo(kDefaultChannels, samples, input.getSampleRate());

  for (int i = 0; i < samples; ++i) {
    const float left = input.getNumChannels() > 0 ? input.getSample(0, i) : 0.0f;
    const float right = input.getNumChannels() > 1 ? input.getSample(1, i) : left;
    stereo.setSample(0, i, left);
    stereo.setSample(1, i, right);
  }

  return stereo;
}

std::vector<double> toMono(const AudioBuffer& input) {
  std::vector<double> mono(static_cast<size_t>(input.getNumSamples()), 0.0);
  const int channels = std::max(1, input.getNumChannels());
  for (int i = 0; i < input.getNumSamples(); ++i) {
    double sum = 0.0;
    for (int ch = 0; ch < channels; ++ch) {
      sum += input.getSample(ch, i);
    }
    mono[static_cast<size_t>(i)] = sum / static_cast<double>(channels);
  }
  return mono;
}

double energyForWindow(const std::vector<double>& signal, const int start, const int length) {
  double energy = 0.0;
  for (int i = 0; i < length; ++i) {
    const double sample = signal[static_cast<size_t>(start + i)];
    energy += sample * sample;
  }
  return energy;
}

int overlapLengthForLag(const int leftSamples, const int rightSamples, const int lag) {
  const int leftStart = std::max(0, -lag);
  const int rightStart = std::max(0, lag);
  const int leftAvailable = leftSamples - leftStart;
  const int rightAvailable = rightSamples - rightStart;
  return std::max(0, std::min(leftAvailable, rightAvailable));
}

AudioBuffer alignTargetToReference(const AudioBuffer& reference,
                                   const AudioBuffer& target,
                                   const int lag) {
  const int samples = reference.getNumSamples();
  AudioBuffer aligned(target.getNumChannels(), samples, reference.getSampleRate());

  for (int ch = 0; ch < aligned.getNumChannels(); ++ch) {
    for (int i = 0; i < samples; ++i) {
      const int sourceIndex = i + lag;
      float sample = 0.0f;
      if (sourceIndex >= 0 && sourceIndex < target.getNumSamples()) {
        sample = target.getSample(ch, sourceIndex);
      }
      aligned.setSample(ch, i, sample);
    }
  }

  return aligned;
}

void sanitizeBuffer(AudioBuffer& buffer) {
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const float sample = buffer.getSample(ch, i);
      if (!isFiniteSample(sample)) {
        buffer.setSample(ch, i, 0.0f);
      }
    }
  }
}

double peakLinear(const AudioBuffer& buffer) {
  double peak = 0.0;
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      peak = std::max(peak, static_cast<double>(std::abs(buffer.getSample(ch, i))));
    }
  }
  return peak;
}

} // namespace

AlignmentResult ResidualBlendProcessor::estimateAlignment(const AudioBuffer& stemSum,
                                                          const AudioBuffer& originalMix,
                                                          const int maxOffsetSamples) const {
  if (stemSum.getNumSamples() == 0 || originalMix.getNumSamples() == 0) {
    return {};
  }
  if (stemSum.getSampleRate() != originalMix.getSampleRate()) {
    throw std::invalid_argument("Alignment requires matching sample rates.");
  }

  const auto referenceMono = toMono(stemSum);
  const auto targetMono = toMono(originalMix);
  const int maxLag = std::max(0, maxOffsetSamples);
  const int minOverlap = std::max(512, std::min(static_cast<int>(referenceMono.size()),
                                                static_cast<int>(targetMono.size())) / 4);

  double bestCorrelation = -1.0;
  int bestLag = 0;

  for (int lag = -maxLag; lag <= maxLag; ++lag) {
    const int overlap = overlapLengthForLag(static_cast<int>(referenceMono.size()),
                                            static_cast<int>(targetMono.size()),
                                            lag);
    if (overlap < minOverlap) {
      continue;
    }

    const int refStart = std::max(0, -lag);
    const int targetStart = std::max(0, lag);
    const double referenceEnergy = energyForWindow(referenceMono, refStart, overlap);
    const double targetEnergy = energyForWindow(targetMono, targetStart, overlap);
    if (referenceEnergy <= 1.0e-12 || targetEnergy <= 1.0e-12) {
      continue;
    }

    double dot = 0.0;
    for (int i = 0; i < overlap; ++i) {
      dot += referenceMono[static_cast<size_t>(refStart + i)] *
             targetMono[static_cast<size_t>(targetStart + i)];
    }

    const double correlation = dot / std::sqrt(referenceEnergy * targetEnergy);
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestLag = lag;
    }
  }

  if (bestCorrelation < 0.0) {
    return {};
  }
  return AlignmentResult{.sampleOffset = bestLag, .normalizedCorrelation = bestCorrelation};
}

ResidualComputation ResidualBlendProcessor::computeResidual(const AudioBuffer& stemSum,
                                                            const AudioBuffer& originalMix,
                                                            const int maxOffsetSamples) const {
  if (stemSum.getNumSamples() == 0) {
    throw std::invalid_argument("Stem sum cannot be empty when computing residual.");
  }
  if (stemSum.getSampleRate() != originalMix.getSampleRate()) {
    throw std::invalid_argument("Residual computation requires matching sample rates.");
  }

  const AudioBuffer stemStereo = toStereoBuffer(stemSum);
  const AudioBuffer originalStereo = toStereoBuffer(originalMix);
  const auto alignment = estimateAlignment(stemStereo, originalStereo, maxOffsetSamples);
  AudioBuffer alignedOriginal = alignTargetToReference(stemStereo, originalStereo, alignment.sampleOffset);

  AudioBuffer residual(kDefaultChannels, stemStereo.getNumSamples(), stemStereo.getSampleRate());
  for (int ch = 0; ch < residual.getNumChannels(); ++ch) {
    for (int i = 0; i < residual.getNumSamples(); ++i) {
      const float sample = alignedOriginal.getSample(ch, i) - stemStereo.getSample(ch, i);
      residual.setSample(ch, i, sample);
    }
  }

  sanitizeBuffer(alignedOriginal);
  sanitizeBuffer(residual);
  return ResidualComputation{
      .alignedOriginalMix = alignedOriginal,
      .residual = residual,
      .alignment = alignment,
  };
}

AudioBuffer ResidualBlendProcessor::applyResidualBlend(const AudioBuffer& stemSum,
                                                       const AudioBuffer& residual,
                                                       const double blendPercent,
                                                       const double ceilingDbtp) const {
  if (stemSum.getSampleRate() != residual.getSampleRate()) {
    throw std::invalid_argument("Residual blending requires matching sample rates.");
  }
  if (stemSum.getNumSamples() != residual.getNumSamples()) {
    throw std::invalid_argument("Residual blending requires matching buffer lengths.");
  }

  const AudioBuffer stemStereo = toStereoBuffer(stemSum);
  const AudioBuffer residualStereo = toStereoBuffer(residual);
  AudioBuffer blended(kDefaultChannels, stemStereo.getNumSamples(), stemStereo.getSampleRate());
  const double blendRatio = std::clamp(blendPercent, 0.0, 10.0) / 100.0;

  for (int ch = 0; ch < blended.getNumChannels(); ++ch) {
    for (int i = 0; i < blended.getNumSamples(); ++i) {
      const float sample = stemStereo.getSample(ch, i) +
                           residualStereo.getSample(ch, i) * static_cast<float>(blendRatio);
      blended.setSample(ch, i, sample);
    }
  }

  sanitizeBuffer(blended);

  const double ceiling = dsp::dbToLinear(std::clamp(ceilingDbtp, -3.0, -0.1));
  const double peak = peakLinear(blended);
  if (peak > ceiling && peak > 0.0) {
    blended.applyGain(static_cast<float>(ceiling / peak));
  }

  sanitizeBuffer(blended);
  return blended;
}

} // namespace automix::engine
