#include "ai/StemSeparator.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "ai/OnnxModelInference.h"
#include "domain/StemOrigin.h"
#include "domain/StemRole.h"
#include "engine/AudioFileIO.h"
#include "util/WavWriter.h"

namespace automix::ai {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct ModelVariant {
  int stemCount = 4;
  std::filesystem::path modelPath;
  size_t gpuMemoryBudgetMb = 256;
  int maxStreams = 2;
  bool fromManifest = false;
};

struct OverlapAddResult {
  bool success = false;
  bool usedModel = false;
  int stemCount = 0;
  size_t chunkFrames = 1;
  int streamCount = 1;
  std::vector<domain::StemRole> stemRoles;
  std::vector<engine::AudioBuffer> stems;
  std::vector<double> confidence;
  std::vector<double> artifactRisk;
  std::string logMessage;
};

double clampSample(const double value) {
  return std::clamp(value, -1.0, 1.0);
}

double clamp01(const double value) {
  return std::clamp(value, 0.0, 1.0);
}

std::string titleCase(std::string value) {
  bool makeUpper = true;
  for (auto& c : value) {
    if (!std::isalpha(static_cast<unsigned char>(c))) {
      makeUpper = true;
      continue;
    }
    if (makeUpper) {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      makeUpper = false;
    }
  }
  return value;
}

std::string roleToken(const domain::StemRole role, const int index) {
  auto token = domain::toString(role);
  if (!token.empty() && token != "unknown") {
    return token;
  }
  return "stem" + std::to_string(index + 1);
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
                                 const std::vector<engine::AudioBuffer>& separated,
                                 const std::optional<size_t>& skipIndex = std::nullopt) {
  engine::AudioBuffer residual(source.getNumChannels(), source.getNumSamples(), source.getSampleRate());
  for (int ch = 0; ch < source.getNumChannels(); ++ch) {
    for (int i = 0; i < source.getNumSamples(); ++i) {
      double value = source.getSample(ch, i);
      for (size_t stemIndex = 0; stemIndex < separated.size(); ++stemIndex) {
        if (skipIndex.has_value() && skipIndex.value() == stemIndex) {
          continue;
        }
        value -= separated[stemIndex].getSample(ch, i);
      }
      residual.setSample(ch, i, static_cast<float>(clampSample(value)));
    }
  }
  return residual;
}

engine::AudioBuffer applyBandPass(const engine::AudioBuffer& input,
                                  const double highPassHz,
                                  const double lowPassHz) {
  auto output = input;
  applyOnePoleHighPass(output, highPassHz);
  applyOnePoleLowPass(output, lowPassHz);
  return output;
}

std::vector<domain::StemRole> rolesForStemCount(const int stemCount) {
  if (stemCount <= 2) {
    return {domain::StemRole::Vocals, domain::StemRole::Music};
  }
  if (stemCount >= 6) {
    return {
        domain::StemRole::Bass,
        domain::StemRole::Vocals,
        domain::StemRole::Drums,
        domain::StemRole::Guitar,
        domain::StemRole::Keys,
        domain::StemRole::Fx,
    };
  }
  return {
      domain::StemRole::Bass,
      domain::StemRole::Vocals,
      domain::StemRole::Drums,
      domain::StemRole::Music,
  };
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

std::vector<double> defaultWeights(const std::vector<double>& features,
                                   const std::vector<domain::StemRole>& roles) {
  const double low = features.size() > 4 ? clamp01(features[4]) : 0.25;
  const double mid = features.size() > 5 ? clamp01(features[5]) : 0.25;
  const double high = features.size() > 6 ? clamp01(features[6]) : 0.25;
  const double flux = features.size() > 7 ? clamp01(features[7]) : 0.25;

  std::vector<double> weights(roles.size(), 0.25);
  for (size_t i = 0; i < roles.size(); ++i) {
    switch (roles[i]) {
      case domain::StemRole::Bass:
        weights[i] = 0.55 + low * 0.45 - high * 0.2;
        break;
      case domain::StemRole::Vocals:
        weights[i] = 0.45 + mid * 0.6 - low * 0.15;
        break;
      case domain::StemRole::Drums:
        weights[i] = 0.35 + high * 0.4 + flux * 0.25;
        break;
      case domain::StemRole::Guitar:
        weights[i] = 0.30 + mid * 0.45 + high * 0.15;
        break;
      case domain::StemRole::Keys:
        weights[i] = 0.25 + mid * 0.30 + high * 0.25;
        break;
      case domain::StemRole::Fx:
        weights[i] = 0.25 + high * 0.45 + flux * 0.20;
        break;
      case domain::StemRole::Music:
        weights[i] = 0.25 + mid * 0.25 + high * 0.15;
        break;
      default:
        weights[i] = 0.20 + mid * 0.2;
        break;
    }
    weights[i] = std::max(0.01, weights[i]);
  }

  const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  for (double& value : weights) {
    value /= std::max(1.0e-9, sum);
  }
  return weights;
}

std::vector<double> weightsFromInference(const InferenceResult& result,
                                         const std::vector<double>& fallback,
                                         const std::vector<domain::StemRole>& roles) {
  auto weights = fallback;
  bool anyExplicitWeight = false;

  for (size_t index = 0; index < roles.size(); ++index) {
    std::vector<std::string> keys = {
        "stem" + std::to_string(index) + "_weight",
        "source" + std::to_string(index) + "_weight",
        "mask_" + std::to_string(index),
    };

    switch (roles[index]) {
      case domain::StemRole::Bass:
        keys.push_back("bass_weight");
        keys.push_back("mask_bass");
        break;
      case domain::StemRole::Vocals:
        keys.push_back("vocals_weight");
        keys.push_back("mask_vocals");
        break;
      case domain::StemRole::Drums:
        keys.push_back("drums_weight");
        keys.push_back("mask_drums");
        break;
      case domain::StemRole::Music:
        keys.push_back("music_weight");
        keys.push_back("other_weight");
        break;
      case domain::StemRole::Guitar:
        keys.push_back("guitar_weight");
        break;
      case domain::StemRole::Keys:
        keys.push_back("keys_weight");
        break;
      case domain::StemRole::Fx:
        keys.push_back("fx_weight");
        break;
      default:
        break;
    }

    if (const auto value = findOutputValue(result, keys); value.has_value()) {
      weights[index] = std::max(0.0, value.value());
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

size_t envUnsigned(const char* key, const size_t fallback) {
  const char* raw = std::getenv(key);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  try {
    const auto value = static_cast<size_t>(std::stoull(raw));
    return std::max<size_t>(1, value);
  } catch (...) {
    return fallback;
  }
}

int envInt(const char* key, const int fallback) {
  const char* raw = std::getenv(key);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  try {
    const auto value = static_cast<int>(std::stoi(raw));
    return std::max(1, value);
  } catch (...) {
    return fallback;
  }
}

std::vector<ModelVariant> discoverModelVariants(const std::filesystem::path& modelRoot) {
  std::vector<ModelVariant> variants;

  const auto manifestPath = modelRoot / "separator_pack.json";
  std::error_code error;
  if (std::filesystem::is_regular_file(manifestPath, error) && !error) {
    try {
      std::ifstream in(manifestPath);
      nlohmann::json json;
      in >> json;
      if (json.contains("variants") && json.at("variants").is_array()) {
        for (const auto& entry : json.at("variants")) {
          const int stemCount = entry.value("stemCount", 0);
          const std::string modelFile = entry.value("modelFile", "");
          if (stemCount <= 0 || modelFile.empty()) {
            continue;
          }
          const auto candidatePath = modelRoot / modelFile;
          error.clear();
          if (!std::filesystem::is_regular_file(candidatePath, error) || error) {
            continue;
          }

          ModelVariant variant;
          variant.stemCount = stemCount;
          variant.modelPath = candidatePath;
          variant.gpuMemoryBudgetMb = static_cast<size_t>(entry.value("gpuMemoryBudgetMb", 256));
          variant.maxStreams = entry.value("maxStreams", 2);
          variant.fromManifest = true;
          variants.push_back(variant);
        }
      }
    } catch (...) {
    }
  }

  const std::array<std::pair<int, std::string>, 5> fallbackNames = {
      std::pair<int, std::string>{2, "separator_2stem.onnx"},
      std::pair<int, std::string>{4, "separator_4stem.onnx"},
      std::pair<int, std::string>{4, "separator.onnx"},
      std::pair<int, std::string>{6, "separator_6stem.onnx"},
      std::pair<int, std::string>{4, "model.onnx"},
  };

  for (const auto& [stemCount, fileName] : fallbackNames) {
    const auto candidatePath = modelRoot / fileName;
    error.clear();
    if (!std::filesystem::is_regular_file(candidatePath, error) || error) {
      continue;
    }

    const bool duplicate = std::any_of(variants.begin(), variants.end(), [&](const ModelVariant& variant) {
      return variant.modelPath == candidatePath;
    });
    if (duplicate) {
      continue;
    }

    ModelVariant variant;
    variant.stemCount = stemCount;
    variant.modelPath = candidatePath;
    variant.gpuMemoryBudgetMb = stemCount >= 6 ? 384 : (stemCount <= 2 ? 192 : 256);
    variant.maxStreams = stemCount >= 6 ? 3 : 2;
    variant.fromManifest = false;
    variants.push_back(variant);
  }

  std::sort(variants.begin(), variants.end(), [](const ModelVariant& a, const ModelVariant& b) {
    if (a.stemCount != b.stemCount) {
      return a.stemCount < b.stemCount;
    }
    if (a.fromManifest != b.fromManifest) {
      return a.fromManifest > b.fromManifest;
    }
    return a.modelPath.string() < b.modelPath.string();
  });

  std::vector<ModelVariant> uniqueByStemCount;
  std::unordered_set<int> seen;
  for (const auto& variant : variants) {
    if (seen.insert(variant.stemCount).second) {
      uniqueByStemCount.push_back(variant);
    }
  }

  return uniqueByStemCount;
}

std::optional<ModelVariant> pickModelVariant(const std::vector<ModelVariant>& variants,
                                             const engine::AudioBuffer& mixBuffer,
                                             const StemSeparator::SeparationOptions& options) {
  if (variants.empty()) {
    return std::nullopt;
  }

  if (options.targetStemCount.has_value()) {
    const int requested = std::clamp(options.targetStemCount.value(), 2, 6);
    auto best = variants.front();
    int bestDistance = std::abs(best.stemCount - requested);
    for (const auto& variant : variants) {
      const int distance = std::abs(variant.stemCount - requested);
      if (distance < bestDistance || (distance == bestDistance && variant.stemCount > best.stemCount)) {
        best = variant;
        bestDistance = distance;
      }
    }
    return best;
  }

  const auto features = extractFrameFeatures(mixBuffer, 0, std::min(4096, std::max(1024, mixBuffer.getNumSamples())));
  const double complexity = clamp01(features[6] * 0.40 + features[7] * 0.30 + features[15] * 0.30);

  const auto findByStemCount = [&](const int stemCount) -> std::optional<ModelVariant> {
    const auto it = std::find_if(variants.begin(), variants.end(), [&](const ModelVariant& variant) {
      return variant.stemCount == stemCount;
    });
    if (it == variants.end()) {
      return std::nullopt;
    }
    return *it;
  };

  if (complexity > 0.65) {
    if (const auto v = findByStemCount(6); v.has_value()) {
      return v;
    }
  }

  if (complexity < 0.35) {
    if (const auto v = findByStemCount(2); v.has_value()) {
      return v;
    }
  }

  if (const auto v = findByStemCount(4); v.has_value()) {
    return v;
  }

  return variants.back();
}

size_t resolveMemoryBudgetMb(const ModelVariant& variant,
                             const StemSeparator::SeparationOptions& options) {
  if (options.gpuMemoryBudgetMb.has_value()) {
    return std::max<size_t>(64, options.gpuMemoryBudgetMb.value());
  }
  return std::max<size_t>(64, envUnsigned("AUTOMIX_SEPARATOR_GPU_BUDGET_MB", variant.gpuMemoryBudgetMb));
}

int resolveMaxStreams(const ModelVariant& variant,
                      const StemSeparator::SeparationOptions& options) {
  if (options.maxStreams.has_value()) {
    return std::clamp(options.maxStreams.value(), 1, 8);
  }
  return std::clamp(envInt("AUTOMIX_SEPARATOR_MAX_STREAMS", variant.maxStreams), 1, 8);
}

OverlapAddResult runModelBackedOverlapAdd(const engine::AudioBuffer& mixBuffer,
                                          const ModelVariant& variant,
                                          const StemSeparator::SeparationOptions& options) {
  OverlapAddResult result;
  result.stemCount = variant.stemCount;
  result.stemRoles = rolesForStemCount(variant.stemCount);

  OnnxModelInference inference;
  inference.setExecutionProviderPreference("auto");
  inference.setGraphOptimizationEnabled(true);
  inference.setWarmupEnabled(true);
  inference.setPreferQuantizedVariants(true);

  if (!inference.loadModel(variant.modelPath)) {
    result.logMessage = "Separator model exists but failed to load.";
    return result;
  }

  const int channels = mixBuffer.getNumChannels();
  const int samples = mixBuffer.getNumSamples();
  if (channels <= 0 || samples <= 0) {
    result.logMessage = "Input buffer has no audio data.";
    return result;
  }

  result.stems.reserve(result.stemCount);
  for (int index = 0; index < result.stemCount; ++index) {
    result.stems.emplace_back(channels, samples, mixBuffer.getSampleRate());
  }
  result.confidence.assign(static_cast<size_t>(result.stemCount), 0.0);
  result.artifactRisk.assign(static_cast<size_t>(result.stemCount), 0.0);

  constexpr int frameSize = 4096;
  constexpr int hopSize = frameSize / 4;
  const auto window = makeHannWindow(frameSize);
  std::vector<double> normalization(static_cast<size_t>(samples), 0.0);
  std::vector<double> confidenceAccumulator(static_cast<size_t>(result.stemCount), 0.0);
  std::vector<double> confidenceWeight(static_cast<size_t>(result.stemCount), 0.0);
  std::vector<double> artifactAccumulator(static_cast<size_t>(result.stemCount), 0.0);
  std::vector<double> artifactWeight(static_cast<size_t>(result.stemCount), 0.0);

  const int totalFrames = (samples + hopSize - 1) / hopSize;
  const size_t budgetMb = resolveMemoryBudgetMb(variant, options);
  const int streamCount = resolveMaxStreams(variant, options);
  const size_t bytesPerFrame = static_cast<size_t>(frameSize) * static_cast<size_t>(std::max(1, channels)) *
                               static_cast<size_t>(std::max(1, result.stemCount)) * sizeof(float) * 3;
  const size_t budgetBytes = std::max<size_t>(64, budgetMb) * 1024ull * 1024ull;
  const size_t framesPerChunk = std::max<size_t>(1, budgetBytes / std::max<size_t>(1, bytesPerFrame));
  result.chunkFrames = framesPerChunk;
  result.streamCount = streamCount;

  int processedFrames = 0;
  int modelFrames = 0;
  for (int chunkIndex = 0; chunkIndex * static_cast<int>(framesPerChunk) < totalFrames; ++chunkIndex) {
    const int chunkFrameBegin = chunkIndex * static_cast<int>(framesPerChunk);
    const int chunkFrameEnd = std::min(totalFrames, chunkFrameBegin + static_cast<int>(framesPerChunk));

    for (int frameIndex = chunkFrameBegin; frameIndex < chunkFrameEnd; ++frameIndex) {
      const int frameStart = frameIndex * hopSize;
      const auto features = extractFrameFeatures(mixBuffer, frameStart, frameSize);
      const auto fallbackWeights = defaultWeights(features, result.stemRoles);
      auto weights = fallbackWeights;
      double confidence = 0.45;

      InferenceRequest request;
      request.task = "stem_separation";
      request.features = features;
      request.scalars["target_stems"] = static_cast<double>(result.stemCount);
      request.scalars["chunk_budget_mb"] = static_cast<double>(budgetMb);
      request.scalars["stream_slot"] = static_cast<double>(frameIndex % std::max(1, streamCount));

      const auto inferenceResult = inference.run(request);
      if (inferenceResult.usedModel) {
        weights = weightsFromInference(inferenceResult, fallbackWeights, result.stemRoles);
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
          for (int stemIndex = 0; stemIndex < result.stemCount; ++stemIndex) {
            const double current = result.stems[static_cast<size_t>(stemIndex)].getSample(ch, absoluteIndex);
            result.stems[static_cast<size_t>(stemIndex)].setSample(
                ch, absoluteIndex, static_cast<float>(current + sample * weights[static_cast<size_t>(stemIndex)]));
          }
        }
      }

      for (int stemIndex = 0; stemIndex < result.stemCount; ++stemIndex) {
        confidenceAccumulator[static_cast<size_t>(stemIndex)] += confidence * weights[static_cast<size_t>(stemIndex)];
        confidenceWeight[static_cast<size_t>(stemIndex)] += weights[static_cast<size_t>(stemIndex)];
        artifactAccumulator[static_cast<size_t>(stemIndex)] += frameArtifactRisk * weights[static_cast<size_t>(stemIndex)];
        artifactWeight[static_cast<size_t>(stemIndex)] += weights[static_cast<size_t>(stemIndex)];
      }

      ++processedFrames;
    }
  }

  if (processedFrames == 0) {
    result.logMessage = "Overlap-add separator could not process any frames.";
    return result;
  }

  for (int i = 0; i < samples; ++i) {
    const double gain = normalization[static_cast<size_t>(i)] > 1.0e-9 ? 1.0 / normalization[static_cast<size_t>(i)] : 0.0;
    for (int stemIndex = 0; stemIndex < result.stemCount; ++stemIndex) {
      for (int ch = 0; ch < channels; ++ch) {
        const double value = static_cast<double>(result.stems[static_cast<size_t>(stemIndex)].getSample(ch, i)) * gain;
        result.stems[static_cast<size_t>(stemIndex)].setSample(ch, i, static_cast<float>(value));
      }
    }
  }

  auto musicIt = std::find(result.stemRoles.begin(), result.stemRoles.end(), domain::StemRole::Music);
  if (musicIt != result.stemRoles.end()) {
    const size_t musicIndex = static_cast<size_t>(std::distance(result.stemRoles.begin(), musicIt));
    const auto residual = makeResidual(mixBuffer, result.stems, musicIndex);
    result.stems[musicIndex] = residual;
  } else if (!result.stems.empty()) {
    const size_t catchAll = result.stems.size() - 1;
    const auto residual = makeResidual(mixBuffer, result.stems);
    for (int ch = 0; ch < channels; ++ch) {
      for (int i = 0; i < samples; ++i) {
        const double value = result.stems[catchAll].getSample(ch, i) + residual.getSample(ch, i);
        result.stems[catchAll].setSample(ch, i, static_cast<float>(clampSample(value)));
      }
    }
  }

  for (int stemIndex = 0; stemIndex < result.stemCount; ++stemIndex) {
    const double confidenceNorm = std::max(1.0e-9, confidenceWeight[static_cast<size_t>(stemIndex)]);
    const double artifactNorm = std::max(1.0e-9, artifactWeight[static_cast<size_t>(stemIndex)]);
    result.confidence[static_cast<size_t>(stemIndex)] =
        clamp01(confidenceAccumulator[static_cast<size_t>(stemIndex)] / confidenceNorm);
    result.artifactRisk[static_cast<size_t>(stemIndex)] =
        clamp01(artifactAccumulator[static_cast<size_t>(stemIndex)] / artifactNorm);
  }

  result.success = true;
  if (result.usedModel) {
    result.logMessage = "Model-backed overlap-add separation completed (" + std::to_string(modelFrames) +
                        " model frames, stems=" + std::to_string(result.stemCount) +
                        ", chunk_frames=" + std::to_string(result.chunkFrames) +
                        ", streams=" + std::to_string(result.streamCount) + ").";
  } else {
    result.logMessage = "Model loaded but returned no usable frame outputs; overlap-add fallback weights used (stems=" +
                        std::to_string(result.stemCount) + ").";
  }
  return result;
}

OverlapAddResult runDeterministicFallback(const engine::AudioBuffer& mixBuffer, const int requestedStemCount) {
  OverlapAddResult result;
  result.success = true;
  result.usedModel = false;
  result.stemRoles = rolesForStemCount(requestedStemCount);
  result.stemCount = static_cast<int>(result.stemRoles.size());

  result.stems.reserve(result.stemRoles.size());
  for (size_t i = 0; i < result.stemRoles.size(); ++i) {
    result.stems.emplace_back(mixBuffer.getNumChannels(), mixBuffer.getNumSamples(), mixBuffer.getSampleRate());
  }
  result.confidence.assign(result.stemRoles.size(), 0.45);
  result.artifactRisk.assign(result.stemRoles.size(), 0.58);

  auto setStem = [&](const domain::StemRole role, const engine::AudioBuffer& buffer) {
    const auto it = std::find(result.stemRoles.begin(), result.stemRoles.end(), role);
    if (it == result.stemRoles.end()) {
      return;
    }
    result.stems[static_cast<size_t>(std::distance(result.stemRoles.begin(), it))] = buffer;
  };

  if (result.stemCount <= 2) {
    auto vocals = applyBandPass(mixBuffer, 180.0, 3500.0);
    setStem(domain::StemRole::Vocals, vocals);
    const auto residual = makeResidual(mixBuffer, result.stems, std::nullopt);
    setStem(domain::StemRole::Music, residual);
    result.confidence = {0.50, 0.48};
    result.artifactRisk = {0.46, 0.52};
    result.logMessage = "No separator model installed; used deterministic 2-stem splitter.";
    return result;
  }

  auto bass = mixBuffer;
  applyOnePoleLowPass(bass, 180.0);
  setStem(domain::StemRole::Bass, bass);

  auto vocals = applyBandPass(mixBuffer, 180.0, 3500.0);
  setStem(domain::StemRole::Vocals, vocals);

  auto drums = mixBuffer;
  applyOnePoleHighPass(drums, 3500.0);
  setStem(domain::StemRole::Drums, drums);

  if (result.stemCount >= 6) {
    auto harmonicResidual = makeResidual(mixBuffer, result.stems);
    auto guitar = applyBandPass(harmonicResidual, 220.0, 2500.0);
    auto keys = applyBandPass(harmonicResidual, 500.0, 7000.0);

    setStem(domain::StemRole::Guitar, guitar);
    setStem(domain::StemRole::Keys, keys);

    const auto fxResidual = makeResidual(harmonicResidual, {guitar, keys});
    setStem(domain::StemRole::Fx, fxResidual);

    result.confidence = {0.44, 0.47, 0.43, 0.40, 0.40, 0.38};
    result.artifactRisk = {0.60, 0.55, 0.62, 0.66, 0.66, 0.70};
    result.logMessage = "No separator model installed; used deterministic 6-stem fallback splitter.";
    return result;
  }

  const auto residualMusic = makeResidual(mixBuffer, result.stems);
  setStem(domain::StemRole::Music, residualMusic);
  result.confidence = {0.46, 0.47, 0.45, 0.44};
  result.artifactRisk = {0.57, 0.56, 0.58, 0.59};
  result.logMessage = "No separator model installed; used deterministic 4-stem frequency splitter.";
  return result;
}

engine::AudioBuffer sumStems(const std::vector<engine::AudioBuffer>& stems,
                             const int channels,
                             const int samples,
                             const double sampleRate) {
  engine::AudioBuffer sum(channels, samples, sampleRate);
  for (const auto& stem : stems) {
    for (int ch = 0; ch < channels; ++ch) {
      for (int i = 0; i < samples; ++i) {
        const double value = sum.getSample(ch, i) + stem.getSample(ch, i);
        sum.setSample(ch, i, static_cast<float>(value));
      }
    }
  }
  return sum;
}

double energy(const engine::AudioBuffer& buffer) {
  double total = 0.0;
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const double sample = buffer.getSample(ch, i);
      total += sample * sample;
    }
  }
  return total;
}

double onsetStrength(const engine::AudioBuffer& buffer) {
  if (buffer.getNumSamples() < 2 || buffer.getNumChannels() <= 0) {
    return 0.0;
  }

  double total = 0.0;
  for (int i = 1; i < buffer.getNumSamples(); ++i) {
    double current = 0.0;
    double previous = 0.0;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      current += std::abs(buffer.getSample(ch, i));
      previous += std::abs(buffer.getSample(ch, i - 1));
    }
    current /= static_cast<double>(buffer.getNumChannels());
    previous /= static_cast<double>(buffer.getNumChannels());
    total += std::abs(current - previous);
  }

  return total;
}

double normalizedCorrelation(const engine::AudioBuffer& a, const engine::AudioBuffer& b) {
  const int channels = std::min(a.getNumChannels(), b.getNumChannels());
  const int samples = std::min(a.getNumSamples(), b.getNumSamples());
  if (channels <= 0 || samples <= 0) {
    return 0.0;
  }

  double dot = 0.0;
  double energyA = 0.0;
  double energyB = 0.0;
  for (int ch = 0; ch < channels; ++ch) {
    for (int i = 0; i < samples; ++i) {
      const double sa = a.getSample(ch, i);
      const double sb = b.getSample(ch, i);
      dot += sa * sb;
      energyA += sa * sa;
      energyB += sb * sb;
    }
  }

  if (energyA <= 1.0e-9 || energyB <= 1.0e-9) {
    return 0.0;
  }
  return std::abs(dot / std::sqrt(energyA * energyB));
}

StemSeparator::SeparationQaMetrics computeQaMetrics(const engine::AudioBuffer& source,
                                                    const std::vector<engine::AudioBuffer>& stems) {
  StemSeparator::SeparationQaMetrics metrics;
  if (stems.empty()) {
    return metrics;
  }

  const auto summed = sumStems(stems, source.getNumChannels(), source.getNumSamples(), source.getSampleRate());
  const auto residual = makeResidual(source, {summed});

  const double sourceEnergy = std::max(1.0e-9, energy(source));
  const double residualEnergy = energy(residual);
  metrics.residualDistortion = std::sqrt(residualEnergy / sourceEnergy);

  const double sourceOnset = std::max(1.0e-9, onsetStrength(source));
  const double summedOnset = onsetStrength(summed);
  metrics.transientRetention = std::clamp(summedOnset / sourceOnset, 0.0, 2.0);

  double leakageSum = 0.0;
  int leakagePairs = 0;
  for (size_t i = 0; i < stems.size(); ++i) {
    double maxLeakageForStem = 0.0;
    for (size_t j = 0; j < stems.size(); ++j) {
      if (i == j) {
        continue;
      }
      maxLeakageForStem = std::max(maxLeakageForStem, normalizedCorrelation(stems[i], stems[j]));
    }
    leakageSum += maxLeakageForStem;
    ++leakagePairs;
  }

  metrics.energyLeakage = leakagePairs > 0 ? leakageSum / static_cast<double>(leakagePairs) : 0.0;
  return metrics;
}

domain::Stem makeStem(const int stemIndex,
                      const domain::StemRole role,
                      const std::filesystem::path& path,
                      const double confidence,
                      const double artifactRisk) {
  domain::Stem stem;
  const auto token = roleToken(role, stemIndex);
  stem.id = "sep_" + token;
  stem.name = "Separated " + titleCase(token);
  stem.filePath = path.string();
  stem.role = role;
  stem.origin = domain::StemOrigin::Separated;
  stem.enabled = true;
  stem.separationConfidence = clamp01(confidence);
  stem.separationArtifactRisk = clamp01(artifactRisk);
  return stem;
}

void writeQaBundle(const std::filesystem::path& path,
                   const StemSeparator::SeparationResult& result,
                   const std::vector<domain::StemRole>& roles,
                   const OverlapAddResult& overlap,
                   const std::optional<ModelVariant>& variant) {
  nlohmann::json qa = {
      {"success", result.success},
      {"usedModel", result.usedModel},
      {"stemVariantCount", result.stemVariantCount},
      {"energyLeakage", result.qaMetrics.energyLeakage},
      {"residualDistortion", result.qaMetrics.residualDistortion},
      {"transientRetention", result.qaMetrics.transientRetention},
      {"chunkFrames", overlap.chunkFrames},
      {"streamCount", overlap.streamCount},
      {"log", result.logMessage},
  };

  nlohmann::json stemRoles = nlohmann::json::array();
  for (size_t i = 0; i < roles.size(); ++i) {
    stemRoles.push_back({
        {"index", i},
        {"role", domain::toString(roles[i])},
        {"confidence", i < overlap.confidence.size() ? overlap.confidence[i] : 0.0},
        {"artifactRisk", i < overlap.artifactRisk.size() ? overlap.artifactRisk[i] : 0.0},
    });
  }
  qa["stems"] = stemRoles;

  if (variant.has_value()) {
    qa["modelVariant"] = {
        {"stemCount", variant->stemCount},
        {"modelPath", variant->modelPath.string()},
        {"gpuMemoryBudgetMb", variant->gpuMemoryBudgetMb},
        {"maxStreams", variant->maxStreams},
        {"fromManifest", variant->fromManifest},
    };
  }

  std::ofstream out(path);
  out << qa.dump(2);
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
  const auto variants = discoverModelVariants(modelRoot_);
  if (!variants.empty()) {
    return true;
  }
  return !resolveModelPath().empty();
}

StemSeparator::SeparationResult StemSeparator::separate(const std::filesystem::path& mixPath,
                                                        const std::filesystem::path& outputDir,
                                                        const SeparationOptions& options) const {
  SeparationResult result;

  try {
    engine::AudioFileIO fileIO;
    auto mixBuffer = fileIO.readAudioFile(mixPath);
    if (mixBuffer.getNumChannels() <= 0 || mixBuffer.getNumSamples() <= 0) {
      result.logMessage = "Input mix has no audio samples.";
      return result;
    }

    std::filesystem::create_directories(outputDir);

    auto variants = discoverModelVariants(modelRoot_);
    if (variants.empty()) {
      const auto fallbackModel = resolveModelPath();
      if (!fallbackModel.empty()) {
        variants.push_back(ModelVariant{.stemCount = 4,
                                        .modelPath = fallbackModel,
                                        .gpuMemoryBudgetMb = 256,
                                        .maxStreams = 2,
                                        .fromManifest = false});
      }
    }

    OverlapAddResult separated;
    std::optional<ModelVariant> selectedVariant;
    if (!variants.empty()) {
      selectedVariant = pickModelVariant(variants, mixBuffer, options);
    }

    if (selectedVariant.has_value()) {
      separated = runModelBackedOverlapAdd(mixBuffer, selectedVariant.value(), options);
      if (!separated.success) {
        const int fallbackStemCount = options.targetStemCount.value_or(selectedVariant->stemCount);
        separated = runDeterministicFallback(mixBuffer, fallbackStemCount);
        separated.logMessage = "Model-backed path failed, fallback used. " + separated.logMessage;
      }
    } else {
      separated = runDeterministicFallback(mixBuffer, options.targetStemCount.value_or(4));
    }

    util::WavWriter writer;
    result.generatedFiles.clear();
    result.stems.clear();

    for (int stemIndex = 0; stemIndex < separated.stemCount; ++stemIndex) {
      const auto role = stemIndex < static_cast<int>(separated.stemRoles.size()) ? separated.stemRoles[static_cast<size_t>(stemIndex)]
                                                                                  : domain::StemRole::Unknown;
      const auto token = roleToken(role, stemIndex);
      const auto stemPath = outputDir / ("stem_" + token + ".wav");
      writer.write(stemPath, separated.stems[static_cast<size_t>(stemIndex)], 24);
      result.generatedFiles.push_back(stemPath);

      const double confidence = stemIndex < static_cast<int>(separated.confidence.size())
                                    ? separated.confidence[static_cast<size_t>(stemIndex)]
                                    : 0.45;
      const double artifactRisk = stemIndex < static_cast<int>(separated.artifactRisk.size())
                                      ? separated.artifactRisk[static_cast<size_t>(stemIndex)]
                                      : 0.58;
      result.stems.push_back(makeStem(stemIndex, role, stemPath, confidence, artifactRisk));
    }

    result.usedModel = separated.usedModel;
    result.stemVariantCount = separated.stemCount;
    result.qaMetrics = computeQaMetrics(mixBuffer, separated.stems);
    result.qaReportPath = outputDir / "separation_qa_report.json";
    result.logMessage = separated.logMessage;

    writeQaBundle(result.qaReportPath, result, separated.stemRoles, separated, selectedVariant);

    result.success = true;
    return result;
  } catch (const std::exception& error) {
    result.logMessage = std::string("Stem separation failed: ") + error.what();
    return result;
  }
}

} // namespace automix::ai
