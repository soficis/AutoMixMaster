#include "ai/StemSeparator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "ai/OnnxModelInference.h"
#include "domain/StemOrigin.h"
#include "domain/StemRole.h"
#include "engine/AudioFileIO.h"
#include "util/WavWriter.h"

namespace automix::ai {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kStemCount = 4;
constexpr int kBassStem = 0;
constexpr int kVocalsStem = 1;
constexpr int kDrumsStem = 2;
constexpr int kMusicStem = 3;

double clampSample(const double value) {
  return std::clamp(value, -1.0, 1.0);
}

double clamp01(const double value) {
  return std::clamp(value, 0.0, 1.0);
}

double lowPassAlpha(const double sampleRate, const double cutoffHz) {
  const double clampedCutoff = std::clamp(cutoffHz, 20.0, sampleRate * 0.45);
  return 1.0 - std::exp(-2.0 * kPi * clampedCutoff / sampleRate);
}

void applyOnePoleLowPass(engine::AudioBuffer& buffer, const double cutoffHz) {
  if (buffer.getNumChannels() <= 0 || buffer.getNumSamples() <= 0) {
    return;
  }
  const double alpha = lowPassAlpha(buffer.getSampleRate(), cutoffHz);

  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    double state = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      state += alpha * (static_cast<double>(buffer.getSample(ch, i)) - state);
      buffer.setSample(ch, i, static_cast<float>(state));
    }
  }
}

void applyOnePoleHighPass(engine::AudioBuffer& buffer, const double cutoffHz) {
  if (buffer.getNumChannels() <= 0 || buffer.getNumSamples() <= 0) {
    return;
  }

  const double alpha = lowPassAlpha(buffer.getSampleRate(), cutoffHz);
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    double lowState = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const double x = buffer.getSample(ch, i);
      lowState += alpha * (x - lowState);
      buffer.setSample(ch, i, static_cast<float>(x - lowState));
    }
  }
}

engine::AudioBuffer makeResidual(const engine::AudioBuffer& source,
                                 const engine::AudioBuffer& bass,
                                 const engine::AudioBuffer& vocals,
                                 const engine::AudioBuffer& drums) {
  engine::AudioBuffer residual(source.getNumChannels(), source.getNumSamples(), source.getSampleRate());
  for (int ch = 0; ch < source.getNumChannels(); ++ch) {
    for (int i = 0; i < source.getNumSamples(); ++i) {
      const double value = source.getSample(ch, i) - bass.getSample(ch, i) - vocals.getSample(ch, i) - drums.getSample(ch, i);
      residual.setSample(ch, i, static_cast<float>(clampSample(value)));
    }
  }
  return residual;
}

domain::Stem makeStem(const std::string& id,
                      const std::string& name,
                      const std::filesystem::path& path,
                      const domain::StemRole role,
                      const double confidence,
                      const double artifactRisk) {
  domain::Stem stem;
  stem.id = id;
  stem.name = name;
  stem.filePath = path.string();
  stem.role = role;
  stem.origin = domain::StemOrigin::Separated;
  stem.enabled = true;
  stem.separationConfidence = clamp01(confidence);
  stem.separationArtifactRisk = clamp01(artifactRisk);
  return stem;
}

std::vector<double> makeHannWindow(const int frameSize) {
  std::vector<double> window(static_cast<size_t>(frameSize), 0.0);
  if (frameSize <= 1) {
    std::fill(window.begin(), window.end(), 1.0);
    return window;
  }
  for (int i = 0; i < frameSize; ++i) {
    window[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos((2.0 * kPi * static_cast<double>(i)) / static_cast<double>(frameSize - 1));
  }
  return window;
}

std::vector<double> extractFrameFeatures(const engine::AudioBuffer& mixBuffer,
                                         const int frameStart,
                                         const int frameSize) {
  double mean = 0.0;
  double rms = 0.0;
  double absMean = 0.0;
  double peak = 0.0;
  double lowEnergy = 0.0;
  double highEnergy = 0.0;
  double flux = 0.0;

  double lowState = 0.0;
  double previous = 0.0;
  const double alpha = lowPassAlpha(mixBuffer.getSampleRate(), 420.0);

  const int channels = std::max(1, mixBuffer.getNumChannels());
  const int totalSamples = mixBuffer.getNumSamples();
  int validSamples = 0;
  for (int i = 0; i < frameSize; ++i) {
    const int absoluteIndex = frameStart + i;
    double sample = 0.0;
    if (absoluteIndex < totalSamples) {
      for (int ch = 0; ch < channels; ++ch) {
        sample += static_cast<double>(mixBuffer.getSample(ch, absoluteIndex));
      }
      sample /= static_cast<double>(channels);
      ++validSamples;
    }

    const double absSample = std::abs(sample);
    lowState += alpha * (sample - lowState);
    const double highSample = sample - lowState;

    mean += sample;
    rms += sample * sample;
    absMean += absSample;
    peak = std::max(peak, absSample);
    lowEnergy += std::abs(lowState);
    highEnergy += std::abs(highSample);
    flux += std::abs(sample - previous);
    previous = sample;
  }

  const double normalizer = static_cast<double>(std::max(1, validSamples));
  mean /= normalizer;
  rms = std::sqrt(rms / normalizer);
  absMean /= normalizer;
  peak = std::max(peak, 1.0e-9);
  lowEnergy /= normalizer;
  highEnergy /= normalizer;
  flux /= normalizer;
  const double midEnergy = std::max(0.0, absMean - lowEnergy * 0.4 - highEnergy * 0.4);
  const double crest = peak / std::max(1.0e-9, rms);

  std::vector<double> features(27, 0.0);
  features[0] = mean;
  features[1] = rms;
  features[2] = std::log1p(crest);
  features[3] = peak;
  features[4] = lowEnergy;
  features[5] = midEnergy;
  features[6] = highEnergy;
  features[7] = flux;
  features[8] = std::abs(mean);
  features[9] = std::log1p(rms);
  features[10] = std::log1p(lowEnergy);
  features[11] = std::log1p(midEnergy);
  features[12] = std::log1p(highEnergy);
  features[13] = std::log1p(flux);
  features[14] = crest;
  features[15] = highEnergy / std::max(1.0e-9, lowEnergy + midEnergy + highEnergy);
  features[16] = lowEnergy / std::max(1.0e-9, lowEnergy + midEnergy + highEnergy);
  features[17] = midEnergy / std::max(1.0e-9, lowEnergy + midEnergy + highEnergy);
  features[18] = peak;
  features[19] = absMean;
  features[20] = flux / std::max(1.0e-9, absMean);
  features[21] = std::abs(highEnergy - lowEnergy);
  features[22] = std::abs(midEnergy - lowEnergy);
  features[23] = std::abs(midEnergy - highEnergy);
  features[24] = static_cast<double>(validSamples) / static_cast<double>(frameSize);
  features[25] = mixBuffer.getSampleRate() / 48000.0;
  features[26] = static_cast<double>(mixBuffer.getNumChannels()) / 2.0;
  return features;
}

std::optional<double> findOutputValue(const InferenceResult& result, const std::vector<std::string>& keys) {
  for (const auto& key : keys) {
    const auto it = result.outputs.find(key);
    if (it != result.outputs.end()) {
      return it->second;
    }
  }
  return std::nullopt;
}

std::array<double, kStemCount> defaultWeights(const std::vector<double>& features) {
  const double low = features.size() > 4 ? clamp01(features[4]) : 0.25;
  const double mid = features.size() > 5 ? clamp01(features[5]) : 0.25;
  const double high = features.size() > 6 ? clamp01(features[6]) : 0.25;
  const double flux = features.size() > 7 ? clamp01(features[7]) : 0.25;

  std::array<double, kStemCount> weights {
      0.55 + low * 0.45 - high * 0.2,
      0.45 + mid * 0.6 - low * 0.15,
      0.35 + high * 0.4 + flux * 0.25,
      0.25 + mid * 0.25 + high * 0.15,
  };

  for (double& value : weights) {
    value = std::max(0.01, value);
  }
  const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  for (double& value : weights) {
    value /= std::max(1.0e-9, sum);
  }
  return weights;
}

std::array<double, kStemCount> weightsFromInference(const InferenceResult& result,
                                                    const std::array<double, kStemCount>& fallback) {
  auto weights = fallback;
  bool anyExplicitWeight = false;
  const std::array<std::vector<std::string>, kStemCount> keyOptions = {
      std::vector<std::string>{"bass_weight", "stem0_weight", "source0_weight", "mask_bass"},
      std::vector<std::string>{"vocals_weight", "stem1_weight", "source1_weight", "mask_vocals"},
      std::vector<std::string>{"drums_weight", "stem2_weight", "source2_weight", "mask_drums"},
      std::vector<std::string>{"music_weight", "other_weight", "stem3_weight", "source3_weight", "mask_other"},
  };

  for (int index = 0; index < kStemCount; ++index) {
    if (const auto value = findOutputValue(result, keyOptions[static_cast<size_t>(index)]); value.has_value()) {
      weights[static_cast<size_t>(index)] = std::max(0.0, value.value());
      anyExplicitWeight = true;
    }
  }

  if (!anyExplicitWeight) {
    return fallback;
  }

  const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (sum <= 1.0e-9) {
    return fallback;
  }
  for (double& value : weights) {
    value = std::max(0.0, value / sum);
  }
  return weights;
}

struct OverlapAddResult {
  bool success = false;
  bool usedModel = false;
  std::array<engine::AudioBuffer, kStemCount> stems;
  std::array<double, kStemCount> confidence {};
  std::array<double, kStemCount> artifactRisk {};
  std::string logMessage;
};

OverlapAddResult runModelBackedOverlapAdd(const engine::AudioBuffer& mixBuffer, const std::filesystem::path& modelPath) {
  OverlapAddResult result;

  OnnxModelInference inference;
  inference.setExecutionProviderPreference("auto");
  inference.setGraphOptimizationEnabled(true);
  inference.setWarmupEnabled(true);
  inference.setPreferQuantizedVariants(true);

  if (!inference.loadModel(modelPath)) {
    result.logMessage = "Separator model exists but failed to load.";
    return result;
  }

  const int channels = mixBuffer.getNumChannels();
  const int samples = mixBuffer.getNumSamples();
  if (channels <= 0 || samples <= 0) {
    result.logMessage = "Input buffer has no audio data.";
    return result;
  }

  for (auto& stem : result.stems) {
    stem = engine::AudioBuffer(channels, samples, mixBuffer.getSampleRate());
  }

  constexpr int frameSize = 4096;
  constexpr int hopSize = frameSize / 4;
  const auto window = makeHannWindow(frameSize);
  std::vector<double> normalization(static_cast<size_t>(samples), 0.0);
  std::array<double, kStemCount> confidenceAccumulator {};
  std::array<double, kStemCount> confidenceWeight {};
  std::array<double, kStemCount> artifactAccumulator {};
  std::array<double, kStemCount> artifactWeight {};

  int processedFrames = 0;
  int modelFrames = 0;
  for (int frameStart = 0; frameStart < samples; frameStart += hopSize) {
    const auto features = extractFrameFeatures(mixBuffer, frameStart, frameSize);
    const auto fallbackWeights = defaultWeights(features);
    auto weights = fallbackWeights;
    double confidence = 0.45;

    InferenceRequest request;
    request.task = "stem_separation";
    request.features = features;
    const auto inferenceResult = inference.run(request);
    if (inferenceResult.usedModel) {
      weights = weightsFromInference(inferenceResult, fallbackWeights);
      confidence = findOutputValue(inferenceResult, {"confidence", "separator_confidence"}).value_or(0.7);
      result.usedModel = true;
      ++modelFrames;
    }

    const double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weightSum <= 1.0e-9) {
      continue;
    }
    for (double& value : weights) {
      value = std::max(0.0, value / weightSum);
    }

    const double frameArtifactRisk = clamp01(1.0 - *std::max_element(weights.begin(), weights.end()));
    for (int i = 0; i < frameSize; ++i) {
      const int absoluteIndex = frameStart + i;
      if (absoluteIndex >= samples) {
        break;
      }
      const double windowed = window[static_cast<size_t>(i)];
      normalization[static_cast<size_t>(absoluteIndex)] += windowed;

      for (int ch = 0; ch < channels; ++ch) {
        const double sample = static_cast<double>(mixBuffer.getSample(ch, absoluteIndex)) * windowed;
        for (int stemIndex = 0; stemIndex < kStemCount; ++stemIndex) {
          const double current = result.stems[static_cast<size_t>(stemIndex)].getSample(ch, absoluteIndex);
          result.stems[static_cast<size_t>(stemIndex)].setSample(
              ch, absoluteIndex, static_cast<float>(current + sample * weights[static_cast<size_t>(stemIndex)]));
        }
      }
    }

    for (int stemIndex = 0; stemIndex < kStemCount; ++stemIndex) {
      confidenceAccumulator[static_cast<size_t>(stemIndex)] += confidence * weights[static_cast<size_t>(stemIndex)];
      confidenceWeight[static_cast<size_t>(stemIndex)] += weights[static_cast<size_t>(stemIndex)];
      artifactAccumulator[static_cast<size_t>(stemIndex)] += frameArtifactRisk * weights[static_cast<size_t>(stemIndex)];
      artifactWeight[static_cast<size_t>(stemIndex)] += weights[static_cast<size_t>(stemIndex)];
    }

    ++processedFrames;
  }

  if (processedFrames == 0) {
    result.logMessage = "Overlap-add separator could not process any frames.";
    return result;
  }

  for (int i = 0; i < samples; ++i) {
    const double gain = normalization[static_cast<size_t>(i)] > 1.0e-9 ? 1.0 / normalization[static_cast<size_t>(i)] : 0.0;
    for (int stemIndex = 0; stemIndex < kStemCount; ++stemIndex) {
      for (int ch = 0; ch < channels; ++ch) {
        const double value = static_cast<double>(result.stems[static_cast<size_t>(stemIndex)].getSample(ch, i)) * gain;
        result.stems[static_cast<size_t>(stemIndex)].setSample(ch, i, static_cast<float>(value));
      }
    }
  }

  // Lock "music" to residual so the separated stems remain phase-consistent with the source.
  for (int ch = 0; ch < channels; ++ch) {
    for (int i = 0; i < samples; ++i) {
      const double value = static_cast<double>(mixBuffer.getSample(ch, i)) -
                           static_cast<double>(result.stems[kBassStem].getSample(ch, i)) -
                           static_cast<double>(result.stems[kVocalsStem].getSample(ch, i)) -
                           static_cast<double>(result.stems[kDrumsStem].getSample(ch, i));
      result.stems[kMusicStem].setSample(ch, i, static_cast<float>(clampSample(value)));
    }
  }

  for (int stemIndex = 0; stemIndex < kStemCount; ++stemIndex) {
    const double confidenceNorm = std::max(1.0e-9, confidenceWeight[static_cast<size_t>(stemIndex)]);
    const double artifactNorm = std::max(1.0e-9, artifactWeight[static_cast<size_t>(stemIndex)]);
    result.confidence[static_cast<size_t>(stemIndex)] =
        clamp01(confidenceAccumulator[static_cast<size_t>(stemIndex)] / confidenceNorm);
    result.artifactRisk[static_cast<size_t>(stemIndex)] =
        clamp01(artifactAccumulator[static_cast<size_t>(stemIndex)] / artifactNorm);
  }

  result.success = true;
  if (result.usedModel) {
    result.logMessage = "Model-backed overlap-add separation completed (" + std::to_string(modelFrames) + " model frames).";
  } else {
    result.logMessage = "Model loaded but returned no usable frame outputs; used overlap-add fallback weights.";
  }
  return result;
}

OverlapAddResult runDeterministicFallback(const engine::AudioBuffer& mixBuffer) {
  OverlapAddResult result;
  result.success = true;
  result.usedModel = false;

  result.stems[kBassStem] = mixBuffer;
  applyOnePoleLowPass(result.stems[kBassStem], 180.0);

  result.stems[kVocalsStem] = mixBuffer;
  applyOnePoleHighPass(result.stems[kVocalsStem], 180.0);
  applyOnePoleLowPass(result.stems[kVocalsStem], 3500.0);

  result.stems[kDrumsStem] = mixBuffer;
  applyOnePoleHighPass(result.stems[kDrumsStem], 3500.0);

  result.stems[kMusicStem] = makeResidual(mixBuffer, result.stems[kBassStem], result.stems[kVocalsStem], result.stems[kDrumsStem]);

  result.confidence = {0.45, 0.45, 0.45, 0.45};
  result.artifactRisk = {0.58, 0.58, 0.58, 0.58};
  result.logMessage = "No separator model installed; used deterministic frequency splitter.";
  return result;
}

} // namespace

StemSeparator::StemSeparator(std::filesystem::path modelRoot) : modelRoot_(std::move(modelRoot)) {}

std::filesystem::path StemSeparator::resolveModelPath() const {
  std::error_code error;
  const auto preferred = modelRoot_ / "separator.onnx";
  if (std::filesystem::is_regular_file(preferred, error) && !error) {
    return preferred;
  }

  const auto fallback = modelRoot_ / "model.onnx";
  error.clear();
  if (std::filesystem::is_regular_file(fallback, error) && !error) {
    return fallback;
  }

  return {};
}

bool StemSeparator::isModelAvailable() const {
  return !resolveModelPath().empty();
}

StemSeparator::SeparationResult StemSeparator::separate(const std::filesystem::path& mixPath,
                                                        const std::filesystem::path& outputDir) const {
  SeparationResult result;

  try {
    engine::AudioFileIO fileIO;
    auto mixBuffer = fileIO.readAudioFile(mixPath);
    if (mixBuffer.getNumChannels() <= 0 || mixBuffer.getNumSamples() <= 0) {
      result.logMessage = "Input mix has no audio samples.";
      return result;
    }

    std::filesystem::create_directories(outputDir);

    OverlapAddResult separated;
    const auto modelPath = resolveModelPath();
    if (!modelPath.empty()) {
      separated = runModelBackedOverlapAdd(mixBuffer, modelPath);
      if (!separated.success) {
        separated = runDeterministicFallback(mixBuffer);
        separated.logMessage = "Model-backed path failed, fallback used. " + separated.logMessage;
      }
    } else {
      separated = runDeterministicFallback(mixBuffer);
    }

    util::WavWriter writer;
    const auto bassPath = outputDir / "stem_bass.wav";
    const auto vocalsPath = outputDir / "stem_vocals.wav";
    const auto drumsPath = outputDir / "stem_drums.wav";
    const auto musicPath = outputDir / "stem_music.wav";

    writer.write(bassPath, separated.stems[kBassStem], 24);
    writer.write(vocalsPath, separated.stems[kVocalsStem], 24);
    writer.write(drumsPath, separated.stems[kDrumsStem], 24);
    writer.write(musicPath, separated.stems[kMusicStem], 24);

    result.generatedFiles = {bassPath, vocalsPath, drumsPath, musicPath};
    result.stems = {
        makeStem("sep_bass",
                 "Separated Bass",
                 bassPath,
                 domain::StemRole::Bass,
                 separated.confidence[kBassStem],
                 separated.artifactRisk[kBassStem]),
        makeStem("sep_vocals",
                 "Separated Vocals",
                 vocalsPath,
                 domain::StemRole::Vocals,
                 separated.confidence[kVocalsStem],
                 separated.artifactRisk[kVocalsStem]),
        makeStem("sep_drums",
                 "Separated Drums",
                 drumsPath,
                 domain::StemRole::Drums,
                 separated.confidence[kDrumsStem],
                 separated.artifactRisk[kDrumsStem]),
        makeStem("sep_music",
                 "Separated Music",
                 musicPath,
                 domain::StemRole::Music,
                 separated.confidence[kMusicStem],
                 separated.artifactRisk[kMusicStem]),
    };

    result.usedModel = separated.usedModel;
    result.logMessage = separated.logMessage;
    result.success = true;
    return result;
  } catch (const std::exception& error) {
    result.logMessage = std::string("Stem separation failed: ") + error.what();
    return result;
  }
}

} // namespace automix::ai
