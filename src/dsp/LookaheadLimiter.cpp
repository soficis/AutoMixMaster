#include "dsp/LookaheadLimiter.h"

#include <algorithm>
#include <cmath>

namespace automix::dsp {
namespace {

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

} // namespace

void LookaheadLimiter::prepare(const double sampleRate, const int channels, const LookaheadLimiterSettings& settings) {
  sampleRate_ = std::max(8000.0, sampleRate);
  channels_ = std::max(1, channels);
  settings_ = settings;
  lookaheadSamples_ = std::max(1, static_cast<int>(std::round(sampleRate_ * settings_.lookaheadMs * 0.001)));

  const size_t delaySize = static_cast<size_t>(lookaheadSamples_ + 1);
  delayLines_.assign(static_cast<size_t>(channels_), std::vector<float>(delaySize, 0.0f));
  delayWriteIndex_.assign(static_cast<size_t>(channels_), 0u);
  detectorLine_.assign(delaySize, 0.0f);
  detectorWriteIndex_ = 0u;
  smoothedGain_ = 1.0f;

  truePeakDetector_.configure(std::max(2, settings_.truePeakOversampleFactor));
}

void LookaheadLimiter::setSettings(const LookaheadLimiterSettings& settings) {
  settings_ = settings;
  lookaheadSamples_ = std::max(1, static_cast<int>(std::round(sampleRate_ * settings_.lookaheadMs * 0.001)));
  truePeakDetector_.configure(std::max(2, settings_.truePeakOversampleFactor));

  const size_t requiredDelaySize = static_cast<size_t>(lookaheadSamples_ + 1);
  for (auto& delay : delayLines_) {
    if (delay.size() != requiredDelaySize) {
      delay.assign(requiredDelaySize, 0.0f);
    }
  }
  if (detectorLine_.size() != requiredDelaySize) {
    detectorLine_.assign(requiredDelaySize, 0.0f);
  } else {
    std::fill(detectorLine_.begin(), detectorLine_.end(), 0.0f);
  }
  delayWriteIndex_.assign(delayLines_.size(), 0u);
  detectorWriteIndex_ = 0u;
}

void LookaheadLimiter::reset() {
  for (auto& delay : delayLines_) {
    std::fill(delay.begin(), delay.end(), 0.0f);
  }
  std::fill(delayWriteIndex_.begin(), delayWriteIndex_.end(), 0u);
  std::fill(detectorLine_.begin(), detectorLine_.end(), 0.0f);
  detectorWriteIndex_ = 0u;
  smoothedGain_ = 1.0f;
}

int LookaheadLimiter::latencySamples() const { return lookaheadSamples_; }

float LookaheadLimiter::softClipSample(const float input) const {
  if (!settings_.softClipEnabled) {
    return input;
  }
  const double drive = std::max(0.1, settings_.softClipDrive);
  const double normalizer = std::tanh(drive);
  if (std::abs(normalizer) < 1.0e-9) {
    return input;
  }
  const double clipped = std::tanh(static_cast<double>(input) * drive) / normalizer;
  return static_cast<float>(clipped);
}

void LookaheadLimiter::process(engine::AudioBuffer& buffer) {
  if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0) {
    return;
  }
  if (channels_ <= 0 || delayLines_.empty()) {
    prepare(buffer.getSampleRate(), buffer.getNumChannels(), settings_);
  }
  if (buffer.getNumChannels() != channels_) {
    prepare(buffer.getSampleRate(), buffer.getNumChannels(), settings_);
  }

  const float ceilingLinear = static_cast<float>(dbToLinear(settings_.ceilingDb));
  const float attackCoeff =
      static_cast<float>(std::exp(-1.0 / std::max(1.0, sampleRate_ * settings_.attackMs * 0.001)));
  const float releaseCoeff =
      static_cast<float>(std::exp(-1.0 / std::max(1.0, sampleRate_ * settings_.releaseMs * 0.001)));

  for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
    float detectorSamplePeak = 0.0f;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      const float input = softClipSample(buffer.getSample(ch, sample));
      buffer.setSample(ch, sample, input);
      detectorSamplePeak = std::max(detectorSamplePeak, std::abs(input));
    }

    detectorLine_[detectorWriteIndex_] = detectorSamplePeak;
    detectorWriteIndex_ = (detectorWriteIndex_ + 1) % detectorLine_.size();

    float lookaheadPeak = 0.0f;
    for (const float detector : detectorLine_) {
      lookaheadPeak = std::max(lookaheadPeak, detector);
    }

    const float targetGain = lookaheadPeak > ceilingLinear ? (ceilingLinear / std::max(lookaheadPeak, 1.0e-9f)) : 1.0f;
    if (targetGain < smoothedGain_) {
      smoothedGain_ = targetGain + attackCoeff * (smoothedGain_ - targetGain);
    } else {
      smoothedGain_ = targetGain + releaseCoeff * (smoothedGain_ - targetGain);
    }
    smoothedGain_ = std::clamp(smoothedGain_, 0.0f, 1.0f);

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      auto& delay = delayLines_[static_cast<size_t>(ch)];
      size_t& writeIndex = delayWriteIndex_[static_cast<size_t>(ch)];
      const size_t readIndex = (writeIndex + 1) % delay.size();

      const float delayed = delay[readIndex];
      delay[writeIndex] = buffer.getSample(ch, sample);
      writeIndex = (writeIndex + 1) % delay.size();

      const float limited = std::clamp(delayed * smoothedGain_, -ceilingLinear, ceilingLinear);
      buffer.setSample(ch, sample, limited);
    }
  }

  if (settings_.truePeakEnabled) {
    const double truePeakDbtp = truePeakDetector_.computeTruePeakDbtp(buffer);
    if (truePeakDbtp > settings_.ceilingDb) {
      const float correction = static_cast<float>(dbToLinear(settings_.ceilingDb - truePeakDbtp));
      buffer.applyGain(correction);
      for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
          buffer.setSample(ch, i, std::clamp(buffer.getSample(ch, i), -ceilingLinear, ceilingLinear));
        }
      }
    }
  }
}

} // namespace automix::dsp
