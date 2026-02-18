#include "automix/HeuristicAutoMixStrategy.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "ai/StemRoleClassifierAI.h"

namespace automix::automix {
namespace {

double targetRmsForRole(const domain::StemRole role) {
  switch (role) {
    case domain::StemRole::Vocals:
      return -18.0;
    case domain::StemRole::Bass:
    case domain::StemRole::Kick:
      return -20.0;
    case domain::StemRole::Drums:
      return -21.0;
    case domain::StemRole::Fx:
      return -24.0;
    default:
      return -22.0;
  }
}

double panForRole(const domain::StemRole role, const int ordinal) {
  switch (role) {
    case domain::StemRole::Bass:
    case domain::StemRole::Kick:
    case domain::StemRole::Vocals:
      return 0.0;
    case domain::StemRole::Guitar:
      return ordinal % 2 == 0 ? -0.4 : 0.4;
    case domain::StemRole::Keys:
      return ordinal % 2 == 0 ? -0.25 : 0.25;
    case domain::StemRole::Fx:
      return ordinal % 2 == 0 ? -0.55 : 0.55;
    default:
      return 0.0;
  }
}

double highPassForRole(const domain::StemRole role) {
  switch (role) {
    case domain::StemRole::Bass:
    case domain::StemRole::Kick:
      return 0.0;
    case domain::StemRole::Vocals:
      return 80.0;
    case domain::StemRole::Fx:
      return 160.0;
    default:
      return 110.0;
  }
}

int dominantBand(const analysis::AnalysisResult& metrics) {
  if (metrics.lowEnergy >= metrics.midEnergy && metrics.lowEnergy >= metrics.highEnergy) {
    return 0;
  }
  if (metrics.midEnergy >= metrics.lowEnergy && metrics.midEnergy >= metrics.highEnergy) {
    return 1;
  }
  return 2;
}

double safeMean(const std::vector<double>& values, const size_t count) {
  if (values.empty() || count == 0) {
    return 0.0;
  }
  const auto limit = std::min(count, values.size());
  return std::accumulate(values.begin(), values.begin() + static_cast<long>(limit), 0.0) /
         static_cast<double>(limit);
}

double normalizedTransientScore(const analysis::AnalysisResult& metrics) {
  const double onset = std::log1p(std::max(0.0, metrics.onsetStrength));
  const double flux = std::log1p(std::max(0.0, metrics.spectralFlux));
  return std::clamp((onset + flux) * 0.35, 0.0, 1.0);
}

double lowFrequencyFocus(const analysis::AnalysisResult& metrics) {
  if (metrics.constantQBins.empty()) {
    return metrics.lowEnergy;
  }
  const double lowCqt = safeMean(metrics.constantQBins, 4);
  return std::clamp((metrics.lowEnergy + lowCqt) * 0.5, 0.0, 1.0);
}

double brightnessHint(const analysis::AnalysisResult& metrics) {
  const double mfcc1 = metrics.mfccCoefficients.size() > 1 ? metrics.mfccCoefficients[1] : 0.0;
  const double mfcc2 = metrics.mfccCoefficients.size() > 2 ? metrics.mfccCoefficients[2] : 0.0;
  const double score = 0.5 + 0.05 * (mfcc1 + mfcc2) + metrics.highEnergy * 0.25 - metrics.lowEnergy * 0.15;
  return std::clamp(score, 0.0, 1.0);
}

} // namespace

domain::MixPlan HeuristicAutoMixStrategy::buildPlan(const domain::Session& session,
                                                     const std::vector<analysis::StemAnalysisEntry>& analysisEntries,
                                                     const double dryWet) const {
  domain::MixPlan plan;
  plan.dryWet = std::clamp(dryWet, 0.0, 1.0);
  plan.mixBusHeadroomDb = 6.0;

  std::unordered_map<std::string, analysis::AnalysisResult> analysisByStem;
  std::unordered_map<int, int> dominantBandCounts;
  for (const auto& entry : analysisEntries) {
    analysisByStem.emplace(entry.stemId, entry.metrics);
    dominantBandCounts[dominantBand(entry.metrics)] += 1;
  }

  int stereoOrdinal = 0;
  for (const auto& stem : session.stems) {
    if (!stem.enabled) {
      continue;
    }

    const auto metricsIt = analysisByStem.find(stem.id);
    if (metricsIt == analysisByStem.end()) {
      continue;
    }

    const auto& metrics = metricsIt->second;
    domain::StemRole role = stem.role;
    ai::StemRoleClassifierAI roleClassifier(nullptr);
    if (role == domain::StemRole::Unknown) {
      const auto prediction = roleClassifier.predict(stem.name.empty() ? stem.filePath : stem.name, metrics);
      role = prediction.role;
    }

    const double transientScore = normalizedTransientScore(metrics);
    const double lowFocus = lowFrequencyFocus(metrics);
    const double brightness = brightnessHint(metrics);
    const double crestFactor = std::clamp(metrics.crestFactor, 1.0, 8.0);

    const double targetRms = targetRmsForRole(role) + (transientScore > 0.6 ? -0.4 : 0.0);
    const bool separatedStem = stem.origin == domain::StemOrigin::Separated;
    const double artifactRisk = std::clamp(metrics.artifactRisk, 0.0, 1.0);
    const double baseGainLimit = separatedStem ? 8.0 : 12.0;
    const double gainLimit = std::max(3.0, baseGainLimit - artifactRisk * 4.0);
    double gainDb = std::clamp(targetRms - metrics.rmsDb, -gainLimit, gainLimit);

    const int overlapCount = dominantBandCounts[dominantBand(metrics)];
    if (overlapCount > 1) {
      gainDb -= std::min(3.0, static_cast<double>(overlapCount - 1) * 0.75);
      gainDb = std::clamp(gainDb, -gainLimit, gainLimit);
    }

    if (metrics.artifactProfile.noiseDominance > 0.55 && metrics.highEnergy > 0.5 && gainDb > 0.0) {
      gainDb = std::min(gainDb, 0.0);
    }

    domain::StemMixDecision decision;
    decision.stemId = stem.id;
    decision.gainDb = gainDb;
    decision.pan = panForRole(role, stereoOrdinal++);
    decision.highPassHz = highPassForRole(role);
    if (role != domain::StemRole::Bass && role != domain::StemRole::Kick && lowFocus > 0.65) {
      decision.highPassHz = std::max(decision.highPassHz, 140.0);
    }

    decision.mudCutDb = (metrics.lowEnergy > 0.5 && metrics.midEnergy > 0.4) ? -1.5 : 0.0;
    if (brightness < 0.35 && decision.mudCutDb < 0.0) {
      decision.mudCutDb = -2.0;
    }

    decision.enableCompressor = role == domain::StemRole::Vocals || role == domain::StemRole::Drums;
    decision.compressorThresholdDb = separatedStem ? -12.0 : -18.0;
    decision.compressorRatio = separatedStem ? 1.6 : 2.5;
    decision.compressorReleaseMs = separatedStem ? 170.0 : 80.0;

    if (transientScore > 0.6 || crestFactor > 4.0) {
      decision.compressorRatio = std::max(1.2, decision.compressorRatio - 0.8);
      decision.compressorReleaseMs = std::max(60.0, decision.compressorReleaseMs - 20.0);
    }

    decision.enableExpander = separatedStem && artifactRisk > 0.30;
    decision.expanderThresholdDb = -44.0 + artifactRisk * 8.0;
    decision.expanderRatio = 1.2 + artifactRisk * 0.8;

    if (separatedStem && decision.enableCompressor && artifactRisk > 0.65) {
      decision.enableCompressor = false;
    }
    if (metrics.artifactProfile.phaseInstability > 0.45) {
      decision.pan *= 0.35;
    }
    if (metrics.artifactProfile.swirlRisk > 0.6 && decision.gainDb > 1.5) {
      decision.gainDb = 1.5;
    }

    plan.decisionLog.push_back("Stem '" + stem.name + "' role=" + domain::toString(role) +
                               " origin=" + domain::toString(stem.origin) +
                               " artifactRisk=" + std::to_string(artifactRisk) +
                               " transient=" + std::to_string(transientScore) +
                               " crestFactor=" + std::to_string(crestFactor) +
                               " overlapCount=" + std::to_string(overlapCount) +
                               " swirl=" + std::to_string(metrics.artifactProfile.swirlRisk) +
                               " phaseInstability=" + std::to_string(metrics.artifactProfile.phaseInstability) +
                               " gainDb=" + std::to_string(decision.gainDb) +
                               " pan=" + std::to_string(decision.pan) +
                               " hpHz=" + std::to_string(decision.highPassHz));
    plan.stemDecisions.push_back(decision);
  }

  plan.decisionLog.push_back("Applied global dry/wet: " + std::to_string(plan.dryWet));
  plan.decisionLog.push_back("Reserved mix bus headroom: " + std::to_string(plan.mixBusHeadroomDb) + " dB");
  return plan;
}

} // namespace automix::automix
