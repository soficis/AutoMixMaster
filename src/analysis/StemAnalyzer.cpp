#include "analysis/StemAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <nlohmann/json.hpp>

#include "analysis/ArtifactRiskEstimator.h"
#include "engine/AudioFileIO.h"

namespace automix::analysis {
namespace {

double linearToDb(const double linear) {
  constexpr double minValue = 1.0e-12;
  return 20.0 * std::log10(std::max(linear, minValue));
}

struct OnePoleLowPass {
  float a = 0.0f;
  float z = 0.0f;

  float process(const float input) {
    z += a * (input - z);
    return z;
  }
};

OnePoleLowPass makeLowPass(const double sampleRate, const double cutoff) {
  const double x = std::exp(-2.0 * 3.14159265358979323846 * cutoff / sampleRate);
  OnePoleLowPass filter;
  filter.a = static_cast<float>(1.0 - x);
  return filter;
}

} // namespace

AnalysisResult StemAnalyzer::analyzeBuffer(const engine::AudioBuffer& buffer) const {
  AnalysisResult result;
  if (buffer.getNumChannels() == 0 || buffer.getNumSamples() == 0) {
    return result;
  }

  double peak = 0.0;
  double energy = 0.0;
  const int totalSamples = buffer.getNumSamples();
  int silenceCount = 0;

  OnePoleLowPass lowL = makeLowPass(buffer.getSampleRate(), 200.0);
  OnePoleLowPass lowR = makeLowPass(buffer.getSampleRate(), 200.0);
  OnePoleLowPass midL = makeLowPass(buffer.getSampleRate(), 2000.0);
  OnePoleLowPass midR = makeLowPass(buffer.getSampleRate(), 2000.0);

  double lowEnergy = 0.0;
  double midEnergy = 0.0;
  double highEnergy = 0.0;

  double sumL = 0.0;
  double sumR = 0.0;
  double sumLL = 0.0;
  double sumRR = 0.0;
  double sumLR = 0.0;

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    const float l = buffer.getSample(0, i);
    const float r = buffer.getNumChannels() > 1 ? buffer.getSample(1, i) : l;

    const double mono = 0.5 * (l + r);
    const double absMono = std::abs(mono);

    peak = std::max(peak, absMono);
    energy += mono * mono;
    if (absMono < 0.001) {
      ++silenceCount;
    }

    const float lowSampleL = lowL.process(l);
    const float lowSampleR = lowR.process(r);
    const float midLowL = midL.process(l);
    const float midLowR = midR.process(r);

    const float midBandL = midLowL - lowSampleL;
    const float midBandR = midLowR - lowSampleR;
    const float highBandL = l - midLowL;
    const float highBandR = r - midLowR;

    lowEnergy += lowSampleL * lowSampleL + lowSampleR * lowSampleR;
    midEnergy += midBandL * midBandL + midBandR * midBandR;
    highEnergy += highBandL * highBandL + highBandR * highBandR;

    sumL += l;
    sumR += r;
    sumLL += l * l;
    sumRR += r * r;
    sumLR += l * r;
  }

  const double rms = std::sqrt(energy / std::max(1, totalSamples));
  result.peakDb = linearToDb(peak);
  result.rmsDb = linearToDb(rms);
  result.crestDb = result.peakDb - result.rmsDb;

  const double bandTotal = lowEnergy + midEnergy + highEnergy + 1.0e-12;
  result.lowEnergy = lowEnergy / bandTotal;
  result.midEnergy = midEnergy / bandTotal;
  result.highEnergy = highEnergy / bandTotal;

  result.silenceRatio = static_cast<double>(silenceCount) / std::max(1, buffer.getNumSamples());

  const double n = static_cast<double>(buffer.getNumSamples());
  const double cov = sumLR - (sumL * sumR) / n;
  const double varL = sumLL - (sumL * sumL) / n;
  const double varR = sumRR - (sumR * sumR) / n;
  if (varL > 1.0e-9 && varR > 1.0e-9) {
    result.stereoCorrelation = cov / std::sqrt(varL * varR);
  } else {
    result.stereoCorrelation = 1.0;
  }

  result.stereoWidth = std::clamp((1.0 - result.stereoCorrelation) * 0.5, 0.0, 1.0);
  ArtifactRiskEstimator riskEstimator;
  result.artifactRisk = riskEstimator.estimate(buffer, result);
  return result;
}

std::vector<StemAnalysisEntry> StemAnalyzer::analyzeSession(const domain::Session& session) const {
  engine::AudioFileIO fileIO;
  std::vector<StemAnalysisEntry> entries;
  entries.reserve(session.stems.size());

  for (const auto& stem : session.stems) {
    if (!stem.enabled) {
      continue;
    }

    engine::AudioBuffer buffer = fileIO.readAudioFile(stem.filePath);
    entries.push_back(StemAnalysisEntry{.stemId = stem.id,
                                        .stemName = stem.name,
                                        .metrics = analyzeBuffer(buffer)});
  }

  return entries;
}

std::string StemAnalyzer::toJsonReport(const std::vector<StemAnalysisEntry>& entries) const {
  nlohmann::json report;
  report["stems"] = nlohmann::json::array();

  for (const auto& entry : entries) {
    report["stems"].push_back({{"stemId", entry.stemId},
                                {"stemName", entry.stemName},
                                {"peakDb", entry.metrics.peakDb},
                                {"rmsDb", entry.metrics.rmsDb},
                                {"crestDb", entry.metrics.crestDb},
                                {"lowEnergy", entry.metrics.lowEnergy},
                                {"midEnergy", entry.metrics.midEnergy},
                                {"highEnergy", entry.metrics.highEnergy},
                                {"silenceRatio", entry.metrics.silenceRatio},
                                {"stereoCorrelation", entry.metrics.stereoCorrelation},
                                {"stereoWidth", entry.metrics.stereoWidth},
                                {"artifactRisk", entry.metrics.artifactRisk}});
  }

  return report.dump(2);
}

} // namespace automix::analysis
