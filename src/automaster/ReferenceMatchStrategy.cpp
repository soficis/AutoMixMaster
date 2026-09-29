#include "automaster/ReferenceMatchStrategy.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "dsp/TruePeakDetector.h"
#include "engine/LoudnessMeter.h"

namespace automix::automaster {
namespace {

constexpr double kBlockSeconds = 0.4;
constexpr double kAbsoluteGateLufs = -70.0;
constexpr int kTruePeakOversample = 4;
constexpr double kMinTargetLufs = -14.0;
constexpr double kMaxTargetLufs = -7.0;
constexpr double kMaxLimiterCeilingDb = -1.0;
constexpr double kMinGlueRatio = 1.0;
constexpr double kMaxGlueRatio = 4.0;

// Per-400 ms block loudness on the same scale LoudnessMeter uses when
// libebur128 is not linked. Only the BS.1770-5 absolute gate at -70 LUFS is
// applied: a relative gate would discard exactly the quiet blocks whose
// spread the glue-ratio fit is trying to measure.
std::vector<double> blockLoudnessValues(const engine::AudioBuffer& buffer) {
  std::vector<double> values;
  const int channels = buffer.getNumChannels();
  const int blockSamples = std::max(1, static_cast<int>(buffer.getSampleRate() * kBlockSeconds));
  if (channels == 0 || buffer.getNumSamples() < blockSamples) {
    return values;
  }

  for (int offset = 0; offset + blockSamples <= buffer.getNumSamples(); offset += blockSamples) {
    double sumSquares = 0.0;
    for (int i = offset; i < offset + blockSamples; ++i) {
      double mono = 0.0;
      for (int ch = 0; ch < channels; ++ch) {
        mono += buffer.getSample(ch, i);
      }
      mono /= static_cast<double>(channels);
      sumSquares += mono * mono;
    }
    const double meanSquare = sumSquares / static_cast<double>(blockSamples);
    values.push_back(-0.691 + 10.0 * std::log10(std::max(meanSquare, 1.0e-12)));
  }
  return values;
}

double gatedPercentile(const engine::AudioBuffer& buffer, const double fraction) {
  auto values = blockLoudnessValues(buffer);
  if (values.empty()) {
    return -120.0;
  }
  values.erase(std::remove_if(values.begin(), values.end(),
                               [](const double value) { return value < kAbsoluteGateLufs; }),
               values.end());
  if (values.empty()) {
    return -120.0;
  }
  std::sort(values.begin(), values.end());
  const double position = std::clamp(fraction, 0.0, 1.0) * static_cast<double>(values.size() - 1);
  const size_t lowIndex = static_cast<size_t>(std::floor(position));
  const size_t highIndex = static_cast<size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lowIndex);
  return values[lowIndex] * (1.0 - weight) + values[highIndex] * weight;
}

} // namespace

ReferenceProfile measureReferenceProfile(const engine::AudioBuffer& reference) {
  ReferenceProfile profile;
  if (reference.getNumSamples() == 0 || reference.getNumChannels() == 0) {
    return profile;
  }

  engine::LoudnessMeter meter;
  const auto metrics = meter.analyze(reference);
  profile.integratedLufs = metrics.integratedLufs;
  profile.shortTermLufs = metrics.shortTermLufs;

  dsp::TruePeakDetector truePeakDetector(kTruePeakOversample);
  profile.truePeakDbtp = truePeakDetector.computeTruePeakDbtp(reference);
  profile.crestDb = metrics.samplePeakDbfs - metrics.integratedLufs;

  profile.p50BlockLufs = gatedPercentile(reference, 0.50);
  profile.p95BlockLufs = gatedPercentile(reference, 0.95);
  return profile;
}

ReferenceMatchStrategy::ReferenceMatchStrategy(ReferenceProfile profile) : profile_(profile) {}

void ReferenceMatchStrategy::setEnabled(const bool enabled) { enabled_ = enabled; }

bool ReferenceMatchStrategy::isEnabled() const { return enabled_; }

domain::MasterPlan ReferenceMatchStrategy::buildPlan(const domain::MasterPreset preset,
                                                     const engine::AudioBuffer& mixBuffer) const {
  auto plan = heuristic_.buildPlan(preset, mixBuffer);

  if (!enabled_) {
    plan.decisionLog.push_back("Reference match inactive: per-session toggle is OFF (non-default route).");
    return plan;
  }
  if (profile_.p95BlockLufs <= profile_.p50BlockLufs) {
    plan.decisionLog.push_back("Reference match inactive: reference profile has no usable block dynamics.");
    return plan;
  }

  const double targetLufs = std::clamp(profile_.integratedLufs, kMinTargetLufs, kMaxTargetLufs);
  plan.targetLufs = targetLufs;
  plan.preGainDb = std::clamp(targetLufs - heuristic_.measureIntegratedLufs(mixBuffer), -9.0, 9.0);

  plan.limiterCeilingDb = std::min(profile_.truePeakDbtp, kMaxLimiterCeilingDb);
  plan.truePeakDbtp = plan.limiterCeilingDb;

  const double referenceSpread = profile_.p95BlockLufs - profile_.p50BlockLufs;
  const double targetSpread =
      gatedPercentile(mixBuffer, 0.95) - gatedPercentile(mixBuffer, 0.50);
  // A narrower block spread means the mix is more heavily compressed, so a
  // wider target spread than the reference asks for LESS glue, not more.
  plan.glueRatio =
      std::clamp(targetSpread / std::max(referenceSpread, 0.1), kMinGlueRatio, kMaxGlueRatio);

  plan.decisionLog.push_back("Reference match active: target LUFS=" + std::to_string(targetLufs) +
                             " (reference=" + std::to_string(profile_.integratedLufs) + ").");
  plan.decisionLog.push_back("Reference match derived ceiling=" +
                             std::to_string(plan.limiterCeilingDb) + " dBTP, glue ratio=" +
                             std::to_string(plan.glueRatio) + ".");
  return plan;
}

engine::AudioBuffer ReferenceMatchStrategy::applyPlan(const engine::AudioBuffer& mixBuffer,
                                                      const domain::MasterPlan& plan,
                                                      MasteringReport* reportOut) const {
  return heuristic_.applyPlan(mixBuffer, plan, reportOut);
}

} // namespace automix::automaster
