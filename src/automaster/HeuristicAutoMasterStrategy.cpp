#include "automaster/HeuristicAutoMasterStrategy.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "analysis/StemAnalyzer.h"
#include "dsp/DeEsser.h"
#include "dsp/DynamicDeHarshEq.h"
#include "dsp/LookaheadLimiter.h"
#include "dsp/MidSideProcessor.h"
#include "dsp/SoftClipper.h"
#include "dsp/TruePeakDetector.h"
#include "engine/LoudnessMeter.h"

namespace automix::automaster {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDcHighPassHz = 20.0;
constexpr double kTonalTiltDb = 1.0;
constexpr double kLoudnessToleranceLu = 0.5;
constexpr int kMaxLoudnessIterations = 5;

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

void applyGain(engine::AudioBuffer& buffer, const double gainDb) {
  buffer.applyGain(static_cast<float>(dbToLinear(gainDb)));
}

float lowPassCoefficient(const double sampleRate, const double cutoffHz) {
  const double nyquistSafeSampleRate = std::max(8000.0, sampleRate);
  const double normalized = -2.0 * kPi * std::max(1.0, cutoffHz) / nyquistSafeSampleRate;
  return static_cast<float>(std::exp(normalized));
}

void applyDcHighPass(engine::AudioBuffer& buffer, const double cutoffHz) {
  if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0) {
    return;
  }

  const float a = lowPassCoefficient(buffer.getSampleRate(), cutoffHz);
  std::vector<float> lowState(static_cast<size_t>(buffer.getNumChannels()), 0.0f);
  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      const float x = buffer.getSample(ch, i);
      lowState[static_cast<size_t>(ch)] = a * lowState[static_cast<size_t>(ch)] + (1.0f - a) * x;
      buffer.setSample(ch, i, x - lowState[static_cast<size_t>(ch)]);
    }
  }
}

void applyTonalTilt(engine::AudioBuffer& buffer, const double tiltDb) {
  if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0 || std::abs(tiltDb) < 1.0e-6) {
    return;
  }

  const float lowGain = static_cast<float>(dbToLinear(-0.5 * tiltDb));
  const float highGain = static_cast<float>(dbToLinear(0.5 * tiltDb));
  const float a = lowPassCoefficient(buffer.getSampleRate(), 950.0);
  std::vector<float> lowState(static_cast<size_t>(buffer.getNumChannels()), 0.0f);

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      const float x = buffer.getSample(ch, i);
      lowState[static_cast<size_t>(ch)] = a * lowState[static_cast<size_t>(ch)] + (1.0f - a) * x;
      const float low = lowState[static_cast<size_t>(ch)];
      const float high = x - low;
      buffer.setSample(ch, i, low * lowGain + high * highGain);
    }
  }
}

void applyGlueCompressor(engine::AudioBuffer& buffer, const double thresholdDb, const double ratio) {
  if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0) {
    return;
  }

  const float threshold = static_cast<float>(dbToLinear(thresholdDb));
  const float ratioClamped = static_cast<float>(std::clamp(ratio, 1.05, 8.0));
  float envelope = 0.0f;
  const float attackCoeff = static_cast<float>(std::exp(-1.0 / std::max(1.0, buffer.getSampleRate() * 0.020)));
  const float releaseCoeff = static_cast<float>(std::exp(-1.0 / std::max(1.0, buffer.getSampleRate() * 0.180)));

  const int windowSamples = std::max(1, static_cast<int>(buffer.getSampleRate() * 0.030));
  std::vector<float> window(static_cast<size_t>(windowSamples), 0.0f);
  int windowWrite = 0;
  float sumSquares = 0.0f;

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    float detector = 0.0f;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      detector = std::max(detector, std::abs(buffer.getSample(ch, i)));
    }

    const float detectorSq = detector * detector;
    sumSquares += detectorSq - window[static_cast<size_t>(windowWrite)];
    window[static_cast<size_t>(windowWrite)] = detectorSq;
    windowWrite = (windowWrite + 1) % windowSamples;
    const float rmsDetector = std::sqrt(std::max(0.0f, sumSquares / static_cast<float>(windowSamples)));

    if (rmsDetector > envelope) {
      envelope = rmsDetector + attackCoeff * (envelope - rmsDetector);
    } else {
      envelope = rmsDetector + releaseCoeff * (envelope - rmsDetector);
    }

    float gain = 1.0f;
    if (envelope > threshold) {
      const float over = envelope / threshold;
      const float compressed = std::pow(over, 1.0f / ratioClamped);
      gain = 1.0f / std::max(compressed, 1.0e-6f);
    }

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      buffer.setSample(ch, i, buffer.getSample(ch, i) * gain);
    }
  }

  const double makeupDb = std::clamp((ratioClamped - 1.0f) * 0.8, 0.0, 2.5);
  buffer.applyGain(static_cast<float>(dbToLinear(makeupDb)));
}

void applyDither(engine::AudioBuffer& buffer, const int bitDepth, const float clampCeilingLinear) {
  if (bitDepth <= 0 || bitDepth >= 32) {
    return;
  }

  std::mt19937 generator(42u);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  const float lsb = std::pow(2.0f, -static_cast<float>(bitDepth));

  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const float dither = (distribution(generator) + distribution(generator)) * 0.5f * lsb;
      const float output = buffer.getSample(ch, i) + dither;
      buffer.setSample(ch, i, std::clamp(output, -clampCeilingLinear, clampCeilingLinear));
    }
  }
}

double computeMonoCorrelation(const engine::AudioBuffer& buffer) {
  if (buffer.getNumChannels() < 2 || buffer.getNumSamples() == 0) {
    return 1.0;
  }

  double sumL = 0.0;
  double sumR = 0.0;
  double sumLL = 0.0;
  double sumRR = 0.0;
  double sumLR = 0.0;
  const double n = static_cast<double>(buffer.getNumSamples());

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    const double l = buffer.getSample(0, i);
    const double r = buffer.getSample(1, i);
    sumL += l;
    sumR += r;
    sumLL += l * l;
    sumRR += r * r;
    sumLR += l * r;
  }

  const double cov = sumLR - (sumL * sumR) / n;
  const double varL = sumLL - (sumL * sumL) / n;
  const double varR = sumRR - (sumR * sumR) / n;
  if (varL < 1.0e-9 || varR < 1.0e-9) {
    return 1.0;
  }
  return std::clamp(cov / std::sqrt(varL * varR), -1.0, 1.0);
}

} // namespace

domain::MasterPlan HeuristicAutoMasterStrategy::buildPlan(const domain::MasterPreset preset,
                                                           const engine::AudioBuffer& mixBuffer) const {
  domain::MasterPlan plan;
  plan.preset = preset;

  switch (preset) {
    case domain::MasterPreset::DefaultStreaming:
      plan.presetName = "DefaultStreaming";
      plan.targetLufs = -14.0;
      plan.truePeakDbtp = -1.0;
      break;
    case domain::MasterPreset::Broadcast:
      plan.presetName = "Broadcast";
      plan.targetLufs = -23.0;
      plan.truePeakDbtp = -1.0;
      break;
    case domain::MasterPreset::UdioOptimized:
      plan.presetName = "UdioOptimized";
      plan.targetLufs = -14.0;
      plan.truePeakDbtp = -1.0;
      plan.applyEq = true;
      plan.enableDeEsser = true;
      plan.enableDeHarshEq = true;
      plan.enableLowMono = true;
      plan.lowMonoHz = 120.0;
      plan.stereoWidth = 0.95;
      plan.enableSoftClipper = true;
      plan.softClipDrive = 1.2;
      break;
    case domain::MasterPreset::Custom:
      plan.presetName = "Custom";
      break;
  }

  const double currentLufs = measureIntegratedLufs(mixBuffer);
  analysis::StemAnalyzer analyzer;
  const auto mixMetrics = analyzer.analyzeBuffer(mixBuffer);
  plan.preGainDb = std::clamp(plan.targetLufs - currentLufs, -9.0, 9.0);
  plan.limiterCeilingDb = plan.truePeakDbtp;

  if (mixMetrics.artifactProfile.noiseDominance > 0.55) {
    plan.enableDeEsser = true;
    plan.deEsserStrength = std::max(plan.deEsserStrength, 0.45);
  }
  if (mixMetrics.artifactProfile.swirlRisk > 0.55) {
    plan.enableDeHarshEq = true;
    plan.deHarshStrength = std::max(plan.deHarshStrength, 0.45);
  }
  if (mixMetrics.artifactProfile.phaseInstability > 0.45) {
    plan.enableLowMono = true;
    plan.stereoWidth = std::min(plan.stereoWidth, 0.9);
  }

  plan.decisionLog.push_back("Master preset selected: " + plan.presetName);
  plan.decisionLog.push_back("Measured integrated LUFS: " + std::to_string(currentLufs));
  plan.decisionLog.push_back("Applied pre-gain: " + std::to_string(plan.preGainDb) + " dB");
  plan.decisionLog.push_back("Limiter ceiling set to: " + std::to_string(plan.limiterCeilingDb) + " dBTP");

  return plan;
}

engine::AudioBuffer HeuristicAutoMasterStrategy::applyPlan(const engine::AudioBuffer& mixBuffer,
                                                            const domain::MasterPlan& plan,
                                                            MasteringReport* reportOut) const {
  engine::AudioBuffer mastered = mixBuffer;
  std::vector<std::string> activeModules;

  applyDcHighPass(mastered, kDcHighPassHz);
  activeModules.push_back("DcHighPass");
  applyGain(mastered, plan.preGainDb);
  if (plan.applyEq) {
    applyTonalTilt(mastered, kTonalTiltDb);
    activeModules.push_back("TonalTiltEq");
  }

  if (plan.enableDeEsser) {
    dsp::DeEsser deEsser;
    deEsser.process(mastered, plan.deEsserStrength);
    activeModules.push_back("DeEsser");
  }

  if (plan.enableDeHarshEq) {
    dsp::DynamicDeHarshEq deHarsh;
    deHarsh.process(mastered, plan.deHarshStrength);
    activeModules.push_back("DynamicDeHarshEq");
  }

  if (plan.enableLowMono || std::abs(plan.stereoWidth - 1.0) > 1.0e-3) {
    dsp::MidSideProcessor midSide;
    midSide.process(mastered, plan.lowMonoHz, plan.stereoWidth);
    activeModules.push_back("MidSideProcessor");
  }

  applyGlueCompressor(mastered, plan.glueThresholdDb, plan.glueRatio);
  activeModules.push_back("GlueCompressor");

  if (plan.enableSoftClipper) {
    dsp::SoftClipper softClipper;
    softClipper.process(mastered, plan.softClipDrive);
    activeModules.push_back("SoftClipper");
  }

  dsp::LookaheadLimiterSettings limiterSettings;
  limiterSettings.ceilingDb = plan.limiterCeilingDb;
  limiterSettings.lookaheadMs = plan.limiterLookaheadMs;
  limiterSettings.attackMs = plan.limiterAttackMs;
  limiterSettings.releaseMs = plan.limiterReleaseMs;
  limiterSettings.truePeakEnabled = plan.limiterTruePeakEnabled;
  limiterSettings.truePeakOversampleFactor = 4;
  limiterSettings.softClipEnabled = false;

  dsp::LookaheadLimiter limiter;
  limiter.prepare(mastered.getSampleRate(), mastered.getNumChannels(), limiterSettings);
  limiter.process(mastered);
  activeModules.push_back("LookaheadLimiter");

  for (int i = 0; i < kMaxLoudnessIterations; ++i) {
    const double loudness = measureIntegratedLufs(mastered);
    const double error = plan.targetLufs - loudness;
    if (std::abs(error) <= kLoudnessToleranceLu) {
      break;
    }
    applyGain(mastered, std::clamp(error, -2.5, 2.5));

    const double truePeakDbtp = estimateTruePeakDbtp(mastered, 4);
    if (truePeakDbtp > plan.truePeakDbtp) {
      applyGain(mastered, plan.truePeakDbtp - truePeakDbtp);
    }

    limiter.process(mastered);
  }

  if (mastered.getNumChannels() > 1) {
    const double monoCorr = computeMonoCorrelation(mastered);
    if (monoCorr < 0.0) {
      dsp::MidSideProcessor monoSafety;
      monoSafety.process(mastered, plan.lowMonoHz, 0.9);
      limiter.process(mastered);
      activeModules.push_back("MonoCompatibilitySafety");
    }
  }

  const float ceilingLinear = static_cast<float>(dbToLinear(plan.truePeakDbtp));
  applyDither(mastered, plan.ditherBitDepth, ceilingLinear);

  if (reportOut != nullptr) {
    engine::LoudnessMeter meter;
    const auto metrics = meter.analyze(mastered);
    reportOut->integratedLufs = metrics.integratedLufs;
    reportOut->shortTermLufs = metrics.shortTermLufs;
    reportOut->loudnessRange = metrics.loudnessRange;
    reportOut->samplePeakDbfs = metrics.samplePeakDbfs;
    reportOut->truePeakDbtp = estimateTruePeakDbtp(mastered, 4);
    reportOut->crestDb = reportOut->samplePeakDbfs - metrics.integratedLufs;
    reportOut->monoCorrelation = computeMonoCorrelation(mastered);
    reportOut->activeModules = activeModules;
  }

  return mastered;
}

double HeuristicAutoMasterStrategy::measureIntegratedLufs(const engine::AudioBuffer& buffer) const {
  engine::LoudnessMeter meter;
  return meter.computeIntegratedLufs(buffer);
}

double HeuristicAutoMasterStrategy::estimateTruePeakDbtp(const engine::AudioBuffer& buffer,
                                                          const int oversampleFactor) const {
  dsp::TruePeakDetector detector(std::max(2, oversampleFactor));
  return detector.computeTruePeakDbtp(buffer);
}

} // namespace automix::automaster
