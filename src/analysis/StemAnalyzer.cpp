#include "analysis/StemAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <juce_dsp/juce_dsp.h>
#include <nlohmann/json.hpp>

#include "analysis/ArtifactRiskEstimator.h"
#include "engine/AudioFileIO.h"

namespace automix::analysis {
namespace {

constexpr double kEpsilon = 1.0e-12;

double linearToDb(const double linear) { return 20.0 * std::log10(std::max(linear, kEpsilon)); }

double clamp01(const double value) { return std::clamp(value, 0.0, 1.0); }

size_t bandIndexForFrequency(const double hz) {
  if (hz < 60.0) {
    return 0; // sub
  }
  if (hz < 150.0) {
    return 1; // bass
  }
  if (hz < 500.0) {
    return 2; // low-mid
  }
  if (hz < 2000.0) {
    return 3; // high-mid
  }
  if (hz < 6000.0) {
    return 4; // presence
  }
  return 5; // air
}

} // namespace

AnalysisResult StemAnalyzer::analyzeBuffer(const engine::AudioBuffer& buffer) const {
  AnalysisResult result;
  if (buffer.getNumChannels() == 0 || buffer.getNumSamples() == 0) {
    return result;
  }

  const int totalSamples = buffer.getNumSamples();
  double peak = 0.0;
  double monoEnergy = 0.0;
  double monoSum = 0.0;
  int silenceCount = 0;

  double sumL = 0.0;
  double sumR = 0.0;
  double sumLL = 0.0;
  double sumRR = 0.0;
  double sumLR = 0.0;

  for (int i = 0; i < totalSamples; ++i) {
    const double l = buffer.getSample(0, i);
    const double r = buffer.getNumChannels() > 1 ? buffer.getSample(1, i) : l;
    const double mono = 0.5 * (l + r);
    const double absMono = std::abs(mono);

    peak = std::max(peak, absMono);
    monoEnergy += mono * mono;
    monoSum += mono;
    if (absMono < 0.001) {
      ++silenceCount;
    }

    sumL += l;
    sumR += r;
    sumLL += l * l;
    sumRR += r * r;
    sumLR += l * r;
  }

  const double monoRms = std::sqrt(monoEnergy / std::max(1, totalSamples));
  result.peakDb = linearToDb(peak);
  result.rmsDb = linearToDb(monoRms);
  result.crestDb = result.peakDb - result.rmsDb;
  result.silenceRatio = static_cast<double>(silenceCount) / std::max(1, totalSamples);
  result.dcOffset = monoSum / std::max(1, totalSamples);

  const double n = static_cast<double>(totalSamples);
  const double cov = sumLR - (sumL * sumR) / n;
  const double varL = sumLL - (sumL * sumL) / n;
  const double varR = sumRR - (sumR * sumR) / n;
  if (varL > 1.0e-9 && varR > 1.0e-9) {
    result.stereoCorrelation = cov / std::sqrt(varL * varR);
  } else {
    result.stereoCorrelation = 1.0;
  }
  result.stereoCorrelation = std::clamp(result.stereoCorrelation, -1.0, 1.0);
  result.stereoWidth = clamp01((1.0 - result.stereoCorrelation) * 0.5);

  const double leftRms = std::sqrt(std::max(varL / n, 0.0));
  const double rightRms = std::sqrt(std::max(varR / n, 0.0));
  result.channelBalanceDb = linearToDb(leftRms + kEpsilon) - linearToDb(rightRms + kEpsilon);

  // FFT-based analysis for spectral bands and MIR-style descriptors.
  constexpr int fftOrder = 11;          // 2048
  constexpr int fftSize = 1 << fftOrder;
  const int hopSize = fftSize / 2;
  const int nyquistBins = fftSize / 2;

  juce::dsp::FFT fft(fftOrder);
  juce::dsp::WindowingFunction<float> window(static_cast<size_t>(fftSize),
                                             juce::dsp::WindowingFunction<float>::hann,
                                             false);
  std::vector<float> fftData(static_cast<size_t>(fftSize * 2), 0.0f);
  std::vector<double> previousMagnitude(static_cast<size_t>(nyquistBins + 1), 0.0);
  std::vector<double> bandEnergy(6, 0.0);

  double weightedFreqSum = 0.0;
  double weightedFreqSquaredSum = 0.0;
  double totalMagnitude = 0.0;
  double logMagnitudeSum = 0.0;
  double spectralFluxSum = 0.0;
  int frameCount = 0;

  for (int start = 0; start < totalSamples; start += hopSize) {
    std::fill(fftData.begin(), fftData.end(), 0.0f);

    for (int i = 0; i < fftSize; ++i) {
      const int sampleIndex = start + i;
      if (sampleIndex >= totalSamples) {
        break;
      }
      const float l = buffer.getSample(0, sampleIndex);
      const float r = buffer.getNumChannels() > 1 ? buffer.getSample(1, sampleIndex) : l;
      fftData[static_cast<size_t>(i)] = 0.5f * (l + r);
    }

    window.multiplyWithWindowingTable(fftData.data(), static_cast<size_t>(fftSize));
    fft.performRealOnlyForwardTransform(fftData.data());

    double frameFlux = 0.0;
    for (int bin = 1; bin < nyquistBins; ++bin) {
      const double re = fftData[static_cast<size_t>(bin * 2)];
      const double im = fftData[static_cast<size_t>(bin * 2 + 1)];
      const double magnitude = std::sqrt(re * re + im * im);
      const double power = magnitude * magnitude;
      const double hz = static_cast<double>(bin) * buffer.getSampleRate() / static_cast<double>(fftSize);

      totalMagnitude += magnitude;
      weightedFreqSum += hz * magnitude;
      weightedFreqSquaredSum += hz * hz * magnitude;
      logMagnitudeSum += std::log(magnitude + kEpsilon);
      bandEnergy[bandIndexForFrequency(hz)] += power;

      if (frameCount > 0) {
        const double delta = std::max(0.0, magnitude - previousMagnitude[static_cast<size_t>(bin)]);
        frameFlux += delta * delta;
      }
      previousMagnitude[static_cast<size_t>(bin)] = magnitude;
    }

    spectralFluxSum += frameFlux;
    ++frameCount;
    if (start + fftSize >= totalSamples) {
      break;
    }
  }

  const double bandTotal = std::accumulate(bandEnergy.begin(), bandEnergy.end(), 0.0) + kEpsilon;
  result.subEnergy = bandEnergy[0] / bandTotal;
  result.bassEnergy = bandEnergy[1] / bandTotal;
  result.lowMidEnergy = bandEnergy[2] / bandTotal;
  result.highMidEnergy = bandEnergy[3] / bandTotal;
  result.presenceEnergy = bandEnergy[4] / bandTotal;
  result.airEnergy = bandEnergy[5] / bandTotal;
  result.lowEnergy = result.subEnergy + result.bassEnergy;
  result.midEnergy = result.lowMidEnergy + result.highMidEnergy;
  result.highEnergy = result.presenceEnergy + result.airEnergy;

  if (totalMagnitude > kEpsilon) {
    result.spectralCentroidHz = weightedFreqSum / totalMagnitude;
    const double meanSquared = weightedFreqSquaredSum / totalMagnitude;
    const double variance = std::max(0.0, meanSquared - result.spectralCentroidHz * result.spectralCentroidHz);
    result.spectralSpreadHz = std::sqrt(variance);
  }

  const int magnitudeBins = std::max(1, frameCount * (nyquistBins - 1));
  const double geometricMean = std::exp(logMagnitudeSum / static_cast<double>(magnitudeBins));
  const double arithmeticMean = totalMagnitude / static_cast<double>(magnitudeBins) + kEpsilon;
  result.spectralFlatness = clamp01(geometricMean / arithmeticMean);
  result.spectralFlux = spectralFluxSum / static_cast<double>(std::max(1, frameCount));

  ArtifactRiskEstimator riskEstimator;
  result.artifactProfile = riskEstimator.profile(buffer, result);
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
    entries.push_back(StemAnalysisEntry{
        .stemId = stem.id,
        .stemName = stem.name,
        .metrics = analyzeBuffer(buffer),
    });
  }

  return entries;
}

std::string StemAnalyzer::toJsonReport(const std::vector<StemAnalysisEntry>& entries) const {
  nlohmann::json report;
  report["stems"] = nlohmann::json::array();

  for (const auto& entry : entries) {
    report["stems"].push_back(
        {{"stemId", entry.stemId},
         {"stemName", entry.stemName},
         {"peakDb", entry.metrics.peakDb},
         {"rmsDb", entry.metrics.rmsDb},
         {"crestDb", entry.metrics.crestDb},
         {"dcOffset", entry.metrics.dcOffset},
         {"lowEnergy", entry.metrics.lowEnergy},
         {"midEnergy", entry.metrics.midEnergy},
         {"highEnergy", entry.metrics.highEnergy},
         {"subEnergy", entry.metrics.subEnergy},
         {"bassEnergy", entry.metrics.bassEnergy},
         {"lowMidEnergy", entry.metrics.lowMidEnergy},
         {"highMidEnergy", entry.metrics.highMidEnergy},
         {"presenceEnergy", entry.metrics.presenceEnergy},
         {"airEnergy", entry.metrics.airEnergy},
         {"spectralCentroidHz", entry.metrics.spectralCentroidHz},
         {"spectralSpreadHz", entry.metrics.spectralSpreadHz},
         {"spectralFlatness", entry.metrics.spectralFlatness},
         {"spectralFlux", entry.metrics.spectralFlux},
         {"silenceRatio", entry.metrics.silenceRatio},
         {"stereoCorrelation", entry.metrics.stereoCorrelation},
         {"stereoWidth", entry.metrics.stereoWidth},
         {"channelBalanceDb", entry.metrics.channelBalanceDb},
         {"artifactRisk", entry.metrics.artifactRisk},
         {"artifactProfile",
          {{"swirlRisk", entry.metrics.artifactProfile.swirlRisk},
           {"smearRisk", entry.metrics.artifactProfile.smearRisk},
           {"noiseDominance", entry.metrics.artifactProfile.noiseDominance},
           {"harmonicity", entry.metrics.artifactProfile.harmonicity},
           {"phaseInstability", entry.metrics.artifactProfile.phaseInstability}}}});
  }

  return report.dump(2);
}

} // namespace automix::analysis
