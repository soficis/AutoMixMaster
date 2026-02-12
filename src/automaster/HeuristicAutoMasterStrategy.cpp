#include "automaster/HeuristicAutoMasterStrategy.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace automix::automaster {
namespace {

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

double linearToDb(const double value) {
  constexpr double minValue = 1.0e-12;
  return 20.0 * std::log10(std::max(value, minValue));
}

void applyGain(engine::AudioBuffer& buffer, const double gainDb) {
  buffer.applyGain(static_cast<float>(dbToLinear(gainDb)));
}

void applyTonalTilt(engine::AudioBuffer& buffer) {
  // Gentle tilt: trim lows very slightly and lift highs slightly.
  float lowL = 0.0f;
  float lowR = 0.0f;
  constexpr float a = 0.01f;

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    const float inL = buffer.getSample(0, i);
    const float inR = buffer.getNumChannels() > 1 ? buffer.getSample(1, i) : inL;

    lowL += a * (inL - lowL);
    lowR += a * (inR - lowR);

    const float highL = inL - lowL;
    const float highR = inR - lowR;

    buffer.setSample(0, i, lowL * 0.98f + highL * 1.02f);
    if (buffer.getNumChannels() > 1) {
      buffer.setSample(1, i, lowR * 0.98f + highR * 1.02f);
    }
  }
}

void applyGlueCompressor(engine::AudioBuffer& buffer, const double thresholdDb, const double ratio) {
  const float threshold = static_cast<float>(dbToLinear(thresholdDb));
  float envelope = 0.0f;
  constexpr float attack = 0.015f;
  constexpr float release = 0.001f;

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    float detector = 0.0f;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      detector = std::max(detector, std::abs(buffer.getSample(ch, i)));
    }

    if (detector > envelope) {
      envelope += (detector - envelope) * attack;
    } else {
      envelope += (detector - envelope) * release;
    }

    float gain = 1.0f;
    if (envelope > threshold) {
      const float over = envelope / threshold;
      const float compressed = std::pow(over, static_cast<float>(1.0 / std::max(1.0, ratio)));
      gain = 1.0f / std::max(compressed, 1.0e-6f);
    }

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      buffer.setSample(ch, i, buffer.getSample(ch, i) * gain);
    }
  }
}

void applyLimiter(engine::AudioBuffer& buffer, const double ceilingDb) {
  const float ceiling = static_cast<float>(dbToLinear(ceilingDb));
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const float sample = buffer.getSample(ch, i);
      const float limited = std::clamp(sample, -ceiling, ceiling);
      buffer.setSample(ch, i, limited);
    }
  }
}

void applyDither(engine::AudioBuffer& buffer, const int bitDepth) {
  if (bitDepth >= 24) {
    return;
  }

  std::mt19937 generator(42u);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  const float lsb = std::pow(2.0f, -static_cast<float>(bitDepth));

  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const float dither = (distribution(generator) + distribution(generator)) * 0.5f * lsb;
      buffer.setSample(ch, i, buffer.getSample(ch, i) + dither);
    }
  }
}

} // namespace

domain::MasterPlan HeuristicAutoMasterStrategy::buildPlan(const domain::MasterPreset preset,
                                                           const engine::AudioBuffer& mixBuffer) const {
  domain::MasterPlan plan;
  plan.preset = preset;

  switch (preset) {
    case domain::MasterPreset::DefaultStreaming:
      plan.targetLufs = -14.0;
      plan.truePeakDbtp = -1.0;
      break;
    case domain::MasterPreset::Broadcast:
      plan.targetLufs = -23.0;
      plan.truePeakDbtp = -1.0;
      break;
    case domain::MasterPreset::Custom:
      break;
  }

  const double currentLufs = measureIntegratedLufs(mixBuffer);
  plan.preGainDb = std::clamp((plan.targetLufs - currentLufs) * 0.7, -6.0, 6.0);
  plan.limiterCeilingDb = plan.truePeakDbtp;

  plan.decisionLog.push_back("Master preset selected: " + domain::toString(plan.preset));
  plan.decisionLog.push_back("Estimated current LUFS: " + std::to_string(currentLufs));
  plan.decisionLog.push_back("Applied pre-gain: " + std::to_string(plan.preGainDb) + " dB");
  plan.decisionLog.push_back("Limiter ceiling set to: " + std::to_string(plan.limiterCeilingDb) + " dBTP");

  return plan;
}

engine::AudioBuffer HeuristicAutoMasterStrategy::applyPlan(const engine::AudioBuffer& mixBuffer,
                                                            const domain::MasterPlan& plan,
                                                            MasteringReport* reportOut) const {
  engine::AudioBuffer mastered = mixBuffer;

  applyGain(mastered, plan.preGainDb);
  if (plan.applyEq) {
    applyTonalTilt(mastered);
  }

  applyGlueCompressor(mastered, plan.glueThresholdDb, plan.glueRatio);
  applyLimiter(mastered, plan.limiterCeilingDb);

  for (int i = 0; i < 2; ++i) {
    const double loudness = measureIntegratedLufs(mastered);
    const double correctionDb = plan.targetLufs - loudness;
    applyGain(mastered, correctionDb);

    const double truePeakDbtp = estimateTruePeakDbtp(mastered, 4);
    if (truePeakDbtp > plan.truePeakDbtp) {
      applyGain(mastered, plan.truePeakDbtp - truePeakDbtp);
    }
  }

  applyLimiter(mastered, plan.limiterCeilingDb);
  applyDither(mastered, plan.ditherBitDepth);

  if (reportOut != nullptr) {
    reportOut->integratedLufs = measureIntegratedLufs(mastered);
    reportOut->truePeakDbtp = estimateTruePeakDbtp(mastered, 4);
  }

  return mastered;
}

double HeuristicAutoMasterStrategy::measureIntegratedLufs(const engine::AudioBuffer& buffer) const {
  if (buffer.getNumSamples() == 0) {
    return -120.0;
  }

  double energy = 0.0;
  const int channels = std::max(1, buffer.getNumChannels());

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    double mono = 0.0;
    for (int ch = 0; ch < channels; ++ch) {
      mono += buffer.getSample(ch, i);
    }
    mono /= channels;
    energy += mono * mono;
  }

  const double meanSquare = energy / static_cast<double>(buffer.getNumSamples());
  // Approximation of LUFS from mean square power.
  return -0.691 + 10.0 * std::log10(std::max(meanSquare, 1.0e-12));
}

double HeuristicAutoMasterStrategy::estimateTruePeakDbtp(const engine::AudioBuffer& buffer,
                                                          const int oversampleFactor) const {
  if (buffer.getNumSamples() <= 1) {
    return -120.0;
  }

  double peak = 0.0;
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples() - 1; ++i) {
      const float a = buffer.getSample(ch, i);
      const float b = buffer.getSample(ch, i + 1);

      peak = std::max(peak, static_cast<double>(std::abs(a)));
      for (int k = 1; k < oversampleFactor; ++k) {
        const float alpha = static_cast<float>(k) / static_cast<float>(oversampleFactor);
        const float interpolated = a + (b - a) * alpha;
        peak = std::max(peak, static_cast<double>(std::abs(interpolated)));
      }
    }
  }

  return linearToDb(peak);
}

} // namespace automix::automaster
