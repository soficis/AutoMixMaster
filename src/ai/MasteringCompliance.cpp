#include "ai/MasteringCompliance.h"

#include <algorithm>
#include <cmath>

#include "engine/LoudnessMeter.h"

namespace automix::ai {
namespace {

// Target LUFS convergence tolerance: 0.5 LU is below typical loudness
// measurement uncertainty and small enough that further tightening yields
// negligible perceptual benefit while increasing iteration cost.
constexpr double kLoudnessToleranceLu = 0.5;

// Upper bound on gain-correction iterations: the algorithm converges
// geometrically in practice, so 4 passes are sufficient to reach
// kLoudnessToleranceLu for realistic material without unnecessary CPU use.
constexpr int kMaxCorrectionIterations = 4;

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

double monoCorrelation(const engine::AudioBuffer& buffer) {
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
  if (varL <= 1.0e-9 || varR <= 1.0e-9) {
    return 1.0;
  }
  return std::clamp(cov / std::sqrt(varL * varR), -1.0, 1.0);
}

} // namespace

domain::MasterPlan MasteringCompliance::enforcePlanBounds(const domain::MasterPlan& plan) const {
  domain::MasterPlan output = plan;
  output.targetLufs = std::clamp(output.targetLufs, -30.0, -8.0);
  output.truePeakDbtp = std::clamp(output.truePeakDbtp, -3.0, -0.1);
  output.limiterCeilingDb = std::clamp(output.limiterCeilingDb, -3.0, -0.1);
  output.preGainDb = std::clamp(output.preGainDb, -9.0, 9.0);
  output.glueThresholdDb = std::clamp(output.glueThresholdDb, -36.0, -6.0);
  output.glueRatio = std::clamp(output.glueRatio, 1.1, 6.0);
  return output;
}

engine::AudioBuffer MasteringCompliance::enforceOutput(const engine::AudioBuffer& masteredBuffer,
                                                       const domain::MasterPlan& plan,
                                                       const automaster::HeuristicAutoMasterStrategy& strategy,
                                                       automaster::MasteringReport* reportOut) const {
  engine::AudioBuffer corrected = masteredBuffer;

  for (int i = 0; i < kMaxCorrectionIterations; ++i) {
    const double lufs = strategy.measureIntegratedLufs(corrected);
    const double loudnessError = plan.targetLufs - lufs;
    if (std::abs(loudnessError) > kLoudnessToleranceLu) {
      corrected.applyGain(static_cast<float>(dbToLinear(std::clamp(loudnessError, -2.5, 2.5))));
    }

    const double truePeak = strategy.estimateTruePeakDbtp(corrected, 4);
    if (truePeak > plan.truePeakDbtp) {
      corrected.applyGain(static_cast<float>(dbToLinear(plan.truePeakDbtp - truePeak)));
    }

    if (std::abs(loudnessError) <= kLoudnessToleranceLu && truePeak <= plan.truePeakDbtp) {
      break;
    }
  }

  if (reportOut != nullptr) {
    engine::LoudnessMeter meter;
    const auto metrics = meter.analyze(corrected);
    reportOut->integratedLufs = metrics.integratedLufs;
    reportOut->shortTermLufs = metrics.shortTermLufs;
    reportOut->loudnessRange = metrics.loudnessRange;
    reportOut->samplePeakDbfs = metrics.samplePeakDbfs;
    reportOut->truePeakDbtp = strategy.estimateTruePeakDbtp(corrected, 4);
    reportOut->crestDb = reportOut->samplePeakDbfs - reportOut->integratedLufs;
    reportOut->monoCorrelation = monoCorrelation(corrected);
  }

  return corrected;
}

} // namespace automix::ai
