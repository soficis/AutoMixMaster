#include "ai/SeparationRunner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace automix::ai {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct ChunkFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::size_t volume(const std::vector<int64_t>& dims) {
  std::size_t count = 1;
  for (const auto dim : dims) {
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

std::vector<float> encodeInput(const analysis::Spectrogram& spec, const InputLayout layout) {
  const int channels = spec.channels;
  const int bins = spec.freqBins;
  const int frames = spec.frames;
  if (layout == InputLayout::MagnitudeChannels) {
    std::vector<float> magnitude(spec.real.size());
    for (std::size_t i = 0; i < magnitude.size(); ++i) {
      magnitude[i] = std::hypot(spec.real[i], spec.imag[i]);
    }
    return magnitude;
  }
  std::vector<float> data(static_cast<std::size_t>(channels) * static_cast<std::size_t>(bins) *
                          static_cast<std::size_t>(frames) * 2);
  for (int ch = 0; ch < channels; ++ch) {
    for (int bin = 0; bin < bins; ++bin) {
      for (int frame = 0; frame < frames; ++frame) {
        const auto source = spec.index(ch, bin, frame);
        std::size_t target = 0;
        if (layout == InputLayout::FoldedStereo) {
          const auto folded = static_cast<std::size_t>(bin) * static_cast<std::size_t>(channels) +
                              static_cast<std::size_t>(ch);
          target = (static_cast<std::size_t>(frame) * static_cast<std::size_t>(bins * channels) + folded) * 2;
        } else {
          target = source * 2;
        }
        data[target] = spec.real[source];
        data[target + 1] = spec.imag[source];
      }
    }
  }
  return data;
}

analysis::Spectrogram decodeOutputStem(const std::vector<float>& data,
                                       const int stemIndex,
                                       const InputLayout layout,
                                       const analysis::Spectrogram& geometry) {
  analysis::Spectrogram spec;
  spec.channels = geometry.channels;
  spec.freqBins = geometry.freqBins;
  spec.frames = geometry.frames;
  spec.sampleRate = geometry.sampleRate;
  const std::size_t plane = static_cast<std::size_t>(spec.channels) * static_cast<std::size_t>(spec.freqBins) *
                            static_cast<std::size_t>(spec.frames);
  spec.real.assign(plane, 0.0f);
  spec.imag.assign(plane, 0.0f);

  const std::size_t stemOffset = static_cast<std::size_t>(stemIndex) * plane * 2;
  for (int ch = 0; ch < spec.channels; ++ch) {
    for (int bin = 0; bin < spec.freqBins; ++bin) {
      for (int frame = 0; frame < spec.frames; ++frame) {
        const auto target = spec.index(ch, bin, frame);
        std::size_t source = 0;
        if (layout == InputLayout::FoldedStereo) {
          const auto folded = static_cast<std::size_t>(bin) * static_cast<std::size_t>(spec.channels) +
                              static_cast<std::size_t>(ch);
          source = stemOffset + (folded * static_cast<std::size_t>(spec.frames) + static_cast<std::size_t>(frame)) * 2;
        } else {
          source = stemOffset + target * 2;
        }
        spec.real[target] = data[source];
        spec.imag[target] = data[source + 1];
      }
    }
  }
  return spec;
}

// Real mask = clip(estimated magnitude / input magnitude, 0, 1), bin by bin.
analysis::Spectrogram ratioMask(const std::vector<float>& estimate,
                                const std::vector<float>& inputMagnitude,
                                const analysis::Spectrogram& geometry) {
  analysis::Spectrogram mask;
  mask.channels = geometry.channels;
  mask.freqBins = geometry.freqBins;
  mask.frames = geometry.frames;
  mask.sampleRate = geometry.sampleRate;
  mask.real.resize(estimate.size());
  mask.imag.assign(estimate.size(), 0.0f);
  for (std::size_t i = 0; i < estimate.size(); ++i) {
    const float denominator = std::max(inputMagnitude[i], 1.0e-8f);
    mask.real[i] = std::clamp(estimate[i] / denominator, 0.0f, 1.0f);
  }
  return mask;
}

// Crossfade weight of sample `i` within a chunk. Consecutive chunks overlap by
// exactly `overlap` samples, and the fade-in of one chunk and the fade-out of
// the previous are sin^2 / cos^2 of the same phase, so they sum to one there.
double chunkWeight(const int i, const int chunkSamples, const int overlap, const bool fadeIn, const bool fadeOut) {
  if (overlap <= 0) {
    return 1.0;
  }
  if (fadeIn && i < overlap) {
    const double phase = 0.5 * kPi * (static_cast<double>(i) + 0.5) / static_cast<double>(overlap);
    return std::sin(phase) * std::sin(phase);
  }
  const int fromEnd = chunkSamples - 1 - i;
  if (fadeOut && fromEnd < overlap) {
    const int j = overlap - 1 - fromEnd;
    const double phase = 0.5 * kPi * (static_cast<double>(j) + 0.5) / static_cast<double>(overlap);
    return std::cos(phase) * std::cos(phase);
  }
  return 1.0;
}

} // namespace

std::string validateRunnerConfig(const RunnerConfig& config) {
  if (config.chunkSamples <= 0 || config.overlapSamples < 0 || config.overlapSamples >= config.chunkSamples) {
    return "invalid chunking: chunk_samples " + std::to_string(config.chunkSamples) + ", overlap_samples " +
           std::to_string(config.overlapSamples);
  }
  if (config.channels <= 0) {
    return "invalid channel count " + std::to_string(config.channels);
  }
  const auto isResidual = [](const SeparationStem& stem) { return !stem.residualOf.empty(); };
  const auto graphStems = std::count_if(config.stems.begin(), config.stems.end(),
                                        [&](const SeparationStem& stem) { return !isResidual(stem); });
  if (graphStems == 0) {
    return "the stem list names no graph-produced stem";
  }
  if ((config.inputLayout == InputLayout::MagnitudeChannels) != (config.outputMode == OutputMode::RatioMask)) {
    return "input_layout magnitude_channels and output_mode ratio_mask only work together";
  }
  if (config.outputMode == OutputMode::Mask || config.outputMode == OutputMode::RatioMask) {
    if (graphStems != 1) {
      return "mask mode produces exactly one graph stem, but " + std::to_string(graphStems) + " are declared";
    }
    const auto target = std::find_if(config.stems.begin(), config.stems.end(),
                                     [&](const SeparationStem& stem) { return stem.name == config.targetStem; });
    if (target == config.stems.end() || isResidual(*target)) {
      return "mask target stem '" + config.targetStem + "' is not a graph-produced stem in the stem list";
    }
  }
  for (const auto& stem : config.stems) {
    if (!isResidual(stem)) {
      continue;
    }
    const auto source = std::find_if(config.stems.begin(), config.stems.end(),
                                     [&](const SeparationStem& other) { return other.name == stem.residualOf; });
    if (source == config.stems.end() || isResidual(*source)) {
      return "stem '" + stem.name + "' is a residual of '" + stem.residualOf +
             "', which is not a graph-produced stem";
    }
  }
  return {};
}

std::optional<InputLayout> inputLayoutFromString(const std::string& value) {
  if (value == "folded_stereo") {
    return InputLayout::FoldedStereo;
  }
  if (value == "split_channels") {
    return InputLayout::SplitChannels;
  }
  if (value == "magnitude_channels") {
    return InputLayout::MagnitudeChannels;
  }
  return std::nullopt;
}

std::optional<OutputMode> outputModeFromString(const std::string& value) {
  if (value == "direct") {
    return OutputMode::Direct;
  }
  if (value == "mask") {
    return OutputMode::Mask;
  }
  if (value == "ratio_mask") {
    return OutputMode::RatioMask;
  }
  return std::nullopt;
}

std::vector<int64_t> tensorInputDims(const InputLayout layout, const int channels, const int freqBins, const int frames) {
  if (layout == InputLayout::FoldedStereo) {
    return {1, frames, static_cast<int64_t>(freqBins) * channels * 2};
  }
  if (layout == InputLayout::MagnitudeChannels) {
    return {1, channels, freqBins, frames};
  }
  return {1, channels, freqBins, frames, 2};
}

std::vector<int64_t> tensorOutputDims(const InputLayout layout,
                                      const int stemAxis,
                                      const int channels,
                                      const int freqBins,
                                      const int frames) {
  if (layout == InputLayout::FoldedStereo) {
    return {1, stemAxis, static_cast<int64_t>(freqBins) * channels, frames, 2};
  }
  if (layout == InputLayout::MagnitudeChannels) {
    return {1, channels, freqBins, frames};
  }
  return {1, stemAxis, channels, freqBins, frames, 2};
}

int SeparationRunner::chunkCount(const int samples, const RunnerConfig& config) {
  if (samples <= 0) {
    return 0;
  }
  if (samples <= config.chunkSamples) {
    return 1;
  }
  const int stride = config.chunkSamples - config.overlapSamples;
  return 1 + (samples - config.chunkSamples + stride - 1) / stride;
}

SeparationRunner::Result SeparationRunner::separate(const engine::AudioBuffer& mix,
                                                    const ITensorInference& inference,
                                                    const RunnerConfig& config) {
  Result result;
  const auto fail = [&](const std::string& reason) {
    Result failed;
    failed.usedModel = false;
    failed.logMessage = "Tensor separation not used: " + reason;
    return failed;
  };

  if (const auto configError = validateRunnerConfig(config); !configError.empty()) {
    return fail(configError);
  }
  const int samples = mix.getNumSamples();
  if (mix.getNumChannels() <= 0 || samples <= 0) {
    return fail("input mix has no audio samples");
  }
  if (std::abs(mix.getSampleRate() - config.sampleRate) > 1.0e-6) {
    return fail("model expects " + std::to_string(static_cast<int>(config.sampleRate)) +
                " Hz audio but the mix is " + std::to_string(static_cast<int>(mix.getSampleRate())) +
                " Hz; resampling is not implemented");
  }
  if (mix.getNumChannels() != config.channels) {
    return fail("model expects " + std::to_string(config.channels) + "-channel audio but the mix has " +
                std::to_string(mix.getNumChannels()) + " channel(s); channels are never duplicated or mixed down");
  }
  if (!inference.isAvailable()) {
    return fail("tensor inference backend is unavailable");
  }

  const auto inputs = inference.inputSpecs();
  const auto outputs = inference.outputSpecs();
  if (inputs.size() != 1 || outputs.size() != 1) {
    return fail("graph declares " + std::to_string(inputs.size()) + " input(s) and " +
                std::to_string(outputs.size()) + " output(s); this runner binds exactly one of each");
  }
  const auto& inputSpec = inputs.front();
  const auto& outputSpec = outputs.front();

  std::vector<std::size_t> graphStemIndex;  // index into config.stems, in stem-axis order
  for (std::size_t i = 0; i < config.stems.size(); ++i) {
    if (config.stems[i].residualOf.empty()) {
      graphStemIndex.push_back(i);
    }
  }
  const int stemAxis = config.outputMode == OutputMode::Direct ? static_cast<int>(graphStemIndex.size()) : 1;

  const int chunk = config.chunkSamples;
  const int overlap = config.overlapSamples;
  const int stride = chunk - overlap;
  const int channels = config.channels;
  const int totalChunks = chunkCount(samples, config);

  // Output stems accumulate in place. The crossfade weights of overlapping
  // chunks sum to one, so no separate normalisation pass is needed.
  std::vector<engine::AudioBuffer> stemAudio;
  stemAudio.reserve(config.stems.size());
  for (std::size_t i = 0; i < config.stems.size(); ++i) {
    stemAudio.emplace_back(channels, samples, mix.getSampleRate());
  }

  const auto cancelRequested = [&config] { return config.cancelRequested && config.cancelRequested(); };
  const auto cancelled = [&](const int chunksDone) {
    Result stopped;
    stopped.cancelled = true;
    stopped.logMessage = "Tensor separation cancelled after " + std::to_string(chunksDone) + "/" +
                         std::to_string(totalChunks) + " chunk(s); no stems were produced.";
    return stopped;
  };

  try {
    for (int chunkIndex = 0; chunkIndex < totalChunks; ++chunkIndex) {
      if (cancelRequested()) {
        return cancelled(chunkIndex);
      }
      const int start = chunkIndex * stride;
      const int valid = std::min(chunk, samples - start);

      // The graph's geometry is fixed, so a short final chunk is zero-padded to
      // the full chunk length and its output trimmed back to `valid` samples.
      engine::AudioBuffer chunkAudio(channels, chunk, mix.getSampleRate());
      for (int ch = 0; ch < channels; ++ch) {
        const float* source = mix.getReadPointer(ch);
        float* destination = chunkAudio.getWritePointer(ch);
        std::copy(source + start, source + start + valid, destination);
      }

      const auto spectrum = analysis::analyze(chunkAudio, config.stft);
      TensorBinding binding;
      binding.expected.name = inputSpec.name;
      binding.expected.elementType = TensorElementType::Float32;
      binding.expected.dims = tensorInputDims(config.inputLayout, channels, spectrum.freqBins, spectrum.frames);
      if (!shapesMatch(inputSpec, binding.expected)) {
        throw ChunkFailure("input '" + inputSpec.name + "' expects " + describeShape(inputSpec.dims) +
                           " but the STFT of a " + std::to_string(chunk) + "-sample chunk is " +
                           describeShape(binding.expected.dims));
      }
      binding.data = encodeInput(spectrum, config.inputLayout);

      const auto inferred = inference.runCancellable({binding}, config.cancelRequested);
      if (!inferred.usedModel && cancelRequested()) {
        // A terminated run reports failure; the cause is the cancellation.
        return cancelled(chunkIndex);
      }
      if (!inferred.usedModel) {
        throw ChunkFailure("chunk " + std::to_string(chunkIndex + 1) + "/" + std::to_string(totalChunks) +
                           " inference failed: " + inferred.logMessage);
      }
      const auto produced = std::find_if(inferred.outputs.begin(), inferred.outputs.end(),
                                         [&](const Tensor& tensor) { return tensor.spec.name == outputSpec.name; });
      if (produced == inferred.outputs.end()) {
        throw ChunkFailure("graph did not return the declared output '" + outputSpec.name + "'");
      }
      const auto expectedOutputDims =
          tensorOutputDims(config.inputLayout, stemAxis, channels, spectrum.freqBins, spectrum.frames);
      const TensorSpec expectedOutput{outputSpec.name, TensorElementType::Float32, expectedOutputDims};
      if (!shapesMatch(expectedOutput, produced->spec) || produced->data.size() != volume(expectedOutputDims)) {
        throw ChunkFailure("output '" + outputSpec.name + "' has shape " + describeShape(produced->spec.dims) +
                           " with " + std::to_string(produced->data.size()) + " values; expected " +
                           describeShape(expectedOutputDims));
      }

      std::vector<std::optional<engine::AudioBuffer>> chunkStems(config.stems.size());
      if (config.outputMode == OutputMode::RatioMask) {
        const auto mask = ratioMask(produced->data, binding.data, spectrum);
        const auto masked = analysis::complexMultiply(spectrum, mask, config.stft.zeroDc);
        chunkStems[graphStemIndex.front()] = analysis::synthesize(masked, config.stft);
      } else if (config.outputMode == OutputMode::Mask) {
        const auto mask = decodeOutputStem(produced->data, 0, config.inputLayout, spectrum);
        const auto masked = analysis::complexMultiply(spectrum, mask, config.stft.zeroDc);
        chunkStems[graphStemIndex.front()] = analysis::synthesize(masked, config.stft);
      } else {
        for (std::size_t n = 0; n < graphStemIndex.size(); ++n) {
          const auto stemSpec = decodeOutputStem(produced->data, static_cast<int>(n), config.inputLayout, spectrum);
          chunkStems[graphStemIndex[n]] = analysis::synthesize(stemSpec, config.stft);
        }
      }
      for (std::size_t i = 0; i < config.stems.size(); ++i) {
        if (config.stems[i].residualOf.empty()) {
          continue;
        }
        const auto sourceIt = std::find_if(config.stems.begin(), config.stems.end(), [&](const SeparationStem& s) {
          return s.name == config.stems[i].residualOf;
        });
        const auto& source = *chunkStems[static_cast<std::size_t>(std::distance(config.stems.begin(), sourceIt))];
        engine::AudioBuffer residual(channels, chunk, mix.getSampleRate());
        for (int ch = 0; ch < channels; ++ch) {
          for (int s = 0; s < chunk; ++s) {
            residual.setSample(ch, s, chunkAudio.getSample(ch, s) - source.getSample(ch, s));
          }
        }
        chunkStems[i] = std::move(residual);
      }

      for (const auto& stem : chunkStems) {
        if (!stem.has_value() || stem->getNumSamples() != chunk || stem->getNumChannels() != channels) {
          throw ChunkFailure("chunk " + std::to_string(chunkIndex + 1) + " reconstructed " +
                             std::to_string(stem.has_value() ? stem->getNumSamples() : 0) + " samples, expected " +
                             std::to_string(chunk));
        }
      }

      const bool fadeIn = chunkIndex > 0;
      const bool fadeOut = chunkIndex + 1 < totalChunks;
      for (std::size_t stemIndex = 0; stemIndex < chunkStems.size(); ++stemIndex) {
        for (int ch = 0; ch < channels; ++ch) {
          float* destination = stemAudio[stemIndex].getWritePointer(ch) + start;
          for (int s = 0; s < valid; ++s) {
            const double weight = chunkWeight(s, chunk, overlap, fadeIn, fadeOut);
            destination[s] += static_cast<float>(weight * static_cast<double>(chunkStems[stemIndex]->getSample(ch, s)));
          }
        }
      }

      if (config.progressCallback) {
        config.progressCallback(chunkIndex + 1, totalChunks);
      }
    }
  } catch (const std::exception& error) {
    return fail(error.what());
  }

  result.stemAudio = std::move(stemAudio);
  for (const auto& stem : config.stems) {
    result.stemNames.push_back(stem.name);
  }
  result.usedModel = true;
  result.logMessage = "Tensor separation completed (" + std::to_string(totalChunks) + " chunk(s), " +
                      std::to_string(config.stems.size()) + " stem(s)).";
  return result;
}

} // namespace automix::ai
