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

double hzToMel(const double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }

double melToHz(const double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

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

std::vector<double> computeMonoSignal(const engine::AudioBuffer& buffer) {
  std::vector<double> mono(static_cast<size_t>(buffer.getNumSamples()), 0.0);
  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    const double l = buffer.getSample(0, i);
    const double r = buffer.getNumChannels() > 1 ? buffer.getSample(1, i) : l;
    mono[static_cast<size_t>(i)] = 0.5 * (l + r);
  }
  return mono;
}

struct StftSummary {
  std::vector<double> avgMagnitude;
  double fluxMean = 0.0;
  int frameCount = 0;
};

StftSummary runStft(const std::vector<double>& mono,
                   const double sampleRate,
                   const int fftOrder,
                   const int hopSize,
                   const bool halfWaveFlux) {
  StftSummary summary;
  const int fftSize = 1 << fftOrder;
  const int nyquistBins = fftSize / 2;

  juce::dsp::FFT fft(fftOrder);
  juce::dsp::WindowingFunction<float> window(static_cast<size_t>(fftSize),
                                             juce::dsp::WindowingFunction<float>::hann,
                                             false);

  std::vector<float> fftData(static_cast<size_t>(fftSize * 2), 0.0f);
  std::vector<double> previousMagnitude(static_cast<size_t>(nyquistBins + 1), 0.0);
  std::vector<double> accumulatedMagnitude(static_cast<size_t>(nyquistBins + 1), 0.0);

  double fluxSum = 0.0;
  int frames = 0;

  const int totalSamples = static_cast<int>(mono.size());
  for (int start = 0; start < totalSamples; start += hopSize) {
    std::fill(fftData.begin(), fftData.end(), 0.0f);

    for (int i = 0; i < fftSize; ++i) {
      const int sampleIndex = start + i;
      if (sampleIndex >= totalSamples) {
        break;
      }
      fftData[static_cast<size_t>(i)] = static_cast<float>(mono[static_cast<size_t>(sampleIndex)]);
    }

    window.multiplyWithWindowingTable(fftData.data(), static_cast<size_t>(fftSize));
    fft.performRealOnlyForwardTransform(fftData.data());

    double frameFlux = 0.0;
    for (int bin = 1; bin < nyquistBins; ++bin) {
      const double re = fftData[static_cast<size_t>(bin * 2)];
      const double im = fftData[static_cast<size_t>(bin * 2 + 1)];
      const double magnitude = std::sqrt(re * re + im * im);
      accumulatedMagnitude[static_cast<size_t>(bin)] += magnitude;

      if (frames > 0) {
        const double delta = magnitude - previousMagnitude[static_cast<size_t>(bin)];
        const double selected = halfWaveFlux ? std::max(0.0, delta) : std::abs(delta);
        frameFlux += selected;
      }
      previousMagnitude[static_cast<size_t>(bin)] = magnitude;
    }

    fluxSum += frameFlux;
    ++frames;
    if (start + fftSize >= totalSamples) {
      break;
    }
  }

  if (frames > 0) {
    for (double& value : accumulatedMagnitude) {
      value /= static_cast<double>(frames);
    }
  }

  summary.avgMagnitude = std::move(accumulatedMagnitude);
  summary.fluxMean = fluxSum / static_cast<double>(std::max(1, frames));
  summary.frameCount = frames;
  return summary;
}

std::vector<std::vector<double>> buildMelFilterBank(const int numBands,
                                                    const int fftSize,
                                                    const double sampleRate,
                                                    const double minHz,
                                                    const double maxHz) {
  const int nyquistBins = fftSize / 2;
  std::vector<std::vector<double>> filters(static_cast<size_t>(numBands),
                                           std::vector<double>(static_cast<size_t>(nyquistBins + 1), 0.0));

  const double melMin = hzToMel(minHz);
  const double melMax = hzToMel(maxHz);

  std::vector<double> melPoints(static_cast<size_t>(numBands + 2), 0.0);
  for (int i = 0; i < numBands + 2; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(numBands + 1);
    melPoints[static_cast<size_t>(i)] = melMin + t * (melMax - melMin);
  }

  std::vector<int> bins(static_cast<size_t>(numBands + 2), 0);
  for (int i = 0; i < numBands + 2; ++i) {
    const double hz = melToHz(melPoints[static_cast<size_t>(i)]);
    const double bin = std::floor((static_cast<double>(fftSize) + 1.0) * hz / sampleRate);
    bins[static_cast<size_t>(i)] = std::clamp(static_cast<int>(bin), 0, nyquistBins);
  }

  for (int m = 1; m <= numBands; ++m) {
    const int left = bins[static_cast<size_t>(m - 1)];
    const int center = bins[static_cast<size_t>(m)];
    const int right = bins[static_cast<size_t>(m + 1)];

    for (int k = left; k < center; ++k) {
      const double denom = std::max(1, center - left);
      filters[static_cast<size_t>(m - 1)][static_cast<size_t>(k)] = (static_cast<double>(k - left)) / denom;
    }
    for (int k = center; k < right; ++k) {
      const double denom = std::max(1, right - center);
      filters[static_cast<size_t>(m - 1)][static_cast<size_t>(k)] = (static_cast<double>(right - k)) / denom;
    }
  }

  return filters;
}

std::vector<double> dctType2(const std::vector<double>& values, const int coeffCount) {
  std::vector<double> output(static_cast<size_t>(coeffCount), 0.0);
  const int n = static_cast<int>(values.size());
  if (n == 0) {
    return output;
  }

  for (int k = 0; k < coeffCount; ++k) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      const double angle = (3.14159265358979323846 / static_cast<double>(n)) * (static_cast<double>(i) + 0.5) *
                           static_cast<double>(k);
      sum += values[static_cast<size_t>(i)] * std::cos(angle);
    }
    output[static_cast<size_t>(k)] = sum;
  }

  return output;
}

std::vector<double> computeConstantQ(const std::vector<double>& avgMagnitude,
                                     const double sampleRate,
                                     const int fftSize,
                                     const int binsPerOctave,
                                     const int totalBins,
                                     const double minHz) {
  std::vector<double> cqt(static_cast<size_t>(totalBins), 0.0);
  if (avgMagnitude.empty()) {
    return cqt;
  }

  const int nyquistBin = static_cast<int>(avgMagnitude.size()) - 1;
  const double q = std::pow(2.0, 1.0 / static_cast<double>(binsPerOctave));
  const double halfBandwidth = std::sqrt(q);

  for (int i = 0; i < totalBins; ++i) {
    const double centerHz = minHz * std::pow(q, static_cast<double>(i));
    const double lowHz = centerHz / halfBandwidth;
    const double highHz = centerHz * halfBandwidth;

    const int lowBin = std::clamp(static_cast<int>(std::floor(lowHz * fftSize / sampleRate)), 1, nyquistBin);
    const int highBin = std::clamp(static_cast<int>(std::ceil(highHz * fftSize / sampleRate)), lowBin, nyquistBin);

    double sum = 0.0;
    for (int bin = lowBin; bin <= highBin; ++bin) {
      sum += avgMagnitude[static_cast<size_t>(bin)];
    }

    cqt[static_cast<size_t>(i)] = sum / static_cast<double>(std::max(1, highBin - lowBin + 1));
  }

  const double total = std::accumulate(cqt.begin(), cqt.end(), 0.0);
  if (total > kEpsilon) {
    for (double& value : cqt) {
      value /= total;
    }
  }

  return cqt;
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
  result.crestFactor = peak / std::max(monoRms, kEpsilon);
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

  const auto monoSignal = computeMonoSignal(buffer);

  // Base STFT used for spectral-balance, centroid/spread/flatness and flux.
  constexpr int kBaseFftOrder = 11; // 2048
  constexpr int kBaseFftSize = 1 << kBaseFftOrder;
  const auto baseSummary = runStft(monoSignal, buffer.getSampleRate(), kBaseFftOrder, kBaseFftSize / 2, false);

  std::vector<double> bandEnergy(6, 0.0);
  double weightedFreqSum = 0.0;
  double weightedFreqSquaredSum = 0.0;
  double totalMagnitude = 0.0;
  double logMagnitudeSum = 0.0;
  int magnitudeBins = 0;

  for (size_t bin = 1; bin < baseSummary.avgMagnitude.size(); ++bin) {
    const double magnitude = baseSummary.avgMagnitude[bin];
    const double power = magnitude * magnitude;
    const double hz = static_cast<double>(bin) * buffer.getSampleRate() / static_cast<double>(kBaseFftSize);

    totalMagnitude += magnitude;
    weightedFreqSum += hz * magnitude;
    weightedFreqSquaredSum += hz * hz * magnitude;
    logMagnitudeSum += std::log(magnitude + kEpsilon);
    bandEnergy[bandIndexForFrequency(hz)] += power;
    ++magnitudeBins;
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

  const double geometricMean = std::exp(logMagnitudeSum / static_cast<double>(std::max(1, magnitudeBins)));
  const double arithmeticMean = totalMagnitude / static_cast<double>(std::max(1, magnitudeBins)) + kEpsilon;
  result.spectralFlatness = clamp01(geometricMean / arithmeticMean);
  result.spectralFlux = baseSummary.fluxMean;

  // Transient-focused STFT for onset strength (short window).
  constexpr int kTransientFftOrder = 8; // 256
  constexpr int kTransientFftSize = 1 << kTransientFftOrder;
  const auto transientSummary = runStft(monoSignal, buffer.getSampleRate(), kTransientFftOrder, kTransientFftSize / 2, true);
  result.onsetStrength = transientSummary.fluxMean;

  // Tonal-focused STFT for MFCC and Constant-Q proxy (long window).
  constexpr int kTonalFftOrder = 12; // 4096
  constexpr int kTonalFftSize = 1 << kTonalFftOrder;
  const auto tonalSummary = runStft(monoSignal, buffer.getSampleRate(), kTonalFftOrder, kTonalFftSize / 2, false);

  const auto melFilters = buildMelFilterBank(26,
                                             kTonalFftSize,
                                             buffer.getSampleRate(),
                                             20.0,
                                             std::max(2000.0, buffer.getSampleRate() * 0.5));
  std::vector<double> melEnergies(melFilters.size(), 0.0);
  for (size_t m = 0; m < melFilters.size(); ++m) {
    double energy = 0.0;
    for (size_t bin = 1; bin < tonalSummary.avgMagnitude.size(); ++bin) {
      energy += melFilters[m][bin] * tonalSummary.avgMagnitude[bin] * tonalSummary.avgMagnitude[bin];
    }
    melEnergies[m] = std::log(energy + kEpsilon);
  }

  result.mfccCoefficients = dctType2(melEnergies, 13);
  result.constantQBins = computeConstantQ(tonalSummary.avgMagnitude,
                                          buffer.getSampleRate(),
                                          kTonalFftSize,
                                          12,
                                          24,
                                          55.0);

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
         {"crestFactor", entry.metrics.crestFactor},
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
         {"onsetStrength", entry.metrics.onsetStrength},
         {"mfccCoefficients", entry.metrics.mfccCoefficients},
         {"constantQBins", entry.metrics.constantQBins},
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
