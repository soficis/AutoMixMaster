#include "engine/OfflineRenderPipeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "domain/MixPlan.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/ResidualBlendProcessor.h"

namespace automix::engine {
namespace {

constexpr double kPi = 3.14159265358979323846;

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

double linearToDb(const double linear) {
  constexpr double minValue = 1.0e-12;
  return 20.0 * std::log10(std::max(linear, minValue));
}

struct HighPassState {
  std::vector<float> previousInput;
  std::vector<float> previousOutput;
};

void applyHighPass(AudioBuffer& buffer, const double cutoffHz) {
  if (cutoffHz <= 0.0) {
    return;
  }

  const double dt = 1.0 / buffer.getSampleRate();
  const double rc = 1.0 / (2.0 * kPi * cutoffHz);
  const float alpha = static_cast<float>(rc / (rc + dt));

  HighPassState state;
  state.previousInput.resize(static_cast<size_t>(buffer.getNumChannels()), 0.0f);
  state.previousOutput.resize(static_cast<size_t>(buffer.getNumChannels()), 0.0f);

  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const float x = buffer.getSample(ch, i);
      const float y = alpha * (state.previousOutput[static_cast<size_t>(ch)] + x - state.previousInput[static_cast<size_t>(ch)]);
      buffer.setSample(ch, i, y);
      state.previousInput[static_cast<size_t>(ch)] = x;
      state.previousOutput[static_cast<size_t>(ch)] = y;
    }
  }
}

void applySimpleCompressor(AudioBuffer& buffer,
                           const double thresholdDb,
                           const double ratio,
                           const double releaseMs) {
  const float threshold = static_cast<float>(dbToLinear(thresholdDb));
  const float ratioClamped = static_cast<float>(std::clamp(ratio, 1.1, 20.0));
  float envelope = 0.0f;
  constexpr float attack = 0.08f;
  const double releaseSamples = std::max(1.0, buffer.getSampleRate() * (releaseMs / 1000.0));
  const float release = static_cast<float>(1.0 / releaseSamples);

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    float detector = 0.0f;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      detector = std::max(detector, std::abs(buffer.getSample(ch, i)));
    }

    if (detector > envelope) {
      envelope += (detector - envelope) * attack;
    } else {
      envelope += (detector - envelope) * release;
    }

    float gain = 1.0f;
    if (envelope > threshold) {
      const float over = envelope - threshold;
      const float compressed = threshold + over / ratioClamped;
      gain = compressed / envelope;
    }

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      buffer.setSample(ch, i, buffer.getSample(ch, i) * gain);
    }
  }
}

void applySimpleExpander(AudioBuffer& buffer, const double thresholdDb, const double ratio) {
  const float threshold = static_cast<float>(dbToLinear(thresholdDb));
  const float ratioClamped = static_cast<float>(std::clamp(ratio, 1.05, 4.0));
  float envelope = 0.0f;
  constexpr float attack = 0.03f;
  constexpr float release = 0.004f;

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    float detector = 0.0f;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      detector = std::max(detector, std::abs(buffer.getSample(ch, i)));
    }

    if (detector > envelope) {
      envelope += (detector - envelope) * attack;
    } else {
      envelope += (detector - envelope) * release;
    }

    float gain = 1.0f;
    if (envelope < threshold && envelope > 1.0e-7f) {
      const float normalized = envelope / threshold;
      gain = std::pow(normalized, ratioClamped - 1.0f);
    }

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      buffer.setSample(ch, i, buffer.getSample(ch, i) * gain);
    }
  }
}

AudioBuffer processStemBuffer(const AudioBuffer& input,
                              const domain::StemMixDecision* decision,
                              const double dryWet,
                              const int outputChannels) {
  AudioBuffer processed = input;

  if (decision != nullptr) {
    applyHighPass(processed, decision->highPassHz);
    if (decision->enableCompressor) {
      applySimpleCompressor(processed,
                            decision->compressorThresholdDb,
                            decision->compressorRatio,
                            decision->compressorReleaseMs);
    }
    if (decision->enableExpander) {
      applySimpleExpander(processed, decision->expanderThresholdDb, decision->expanderRatio);
    }
    const float gainLinear = static_cast<float>(dbToLinear(decision->gainDb));
    processed.applyGain(gainLinear);
  }

  AudioBuffer output(outputChannels, processed.getNumSamples(), processed.getSampleRate());
  const double pan = decision != nullptr ? std::clamp(decision->pan, -1.0, 1.0) : 0.0;
  const float leftPanGain = static_cast<float>(std::cos((pan + 1.0) * (kPi / 4.0)));
  const float rightPanGain = static_cast<float>(std::sin((pan + 1.0) * (kPi / 4.0)));

  for (int i = 0; i < processed.getNumSamples(); ++i) {
    float left = 0.0f;
    float right = 0.0f;

    if (processed.getNumChannels() == 1) {
      const float mono = processed.getSample(0, i);
      left = mono * leftPanGain;
      right = mono * rightPanGain;
    } else {
      left = processed.getSample(0, i) * leftPanGain;
      right = processed.getSample(1, i) * rightPanGain;
    }

    const float dryLeft = input.getSample(0, i);
    const float dryRight = input.getNumChannels() > 1 ? input.getSample(1, i) : dryLeft;

    const float outLeft = dryLeft * static_cast<float>(1.0 - dryWet) + left * static_cast<float>(dryWet);
    const float outRight = dryRight * static_cast<float>(1.0 - dryWet) + right * static_cast<float>(dryWet);

    output.setSample(0, i, outLeft);
    if (outputChannels > 1) {
      output.setSample(1, i, outRight);
    }
  }

  return output;
}

} // namespace

OfflineRenderResult OfflineRenderPipeline::renderRawMix(const domain::Session& session,
                                                        const domain::RenderSettings& settings,
                                                        const ProgressCallback& onProgress,
                                                        const std::atomic_bool* cancelFlag) const {
  OfflineRenderResult result;
  AudioFileIO fileIO;
  AudioResampler resampler;

  std::unordered_map<std::string, domain::StemMixDecision> decisions;
  double dryWet = 1.0;
  double headroomDb = 6.0;

  if (session.mixPlan.has_value()) {
    dryWet = session.mixPlan->dryWet;
    headroomDb = session.mixPlan->mixBusHeadroomDb;
    for (const auto& decision : session.mixPlan->stemDecisions) {
      decisions.emplace(decision.stemId, decision);
    }
  }

  std::vector<AudioBuffer> stemBuffers;
  stemBuffers.reserve(session.stems.size());

  for (size_t i = 0; i < session.stems.size(); ++i) {
    if (cancelFlag != nullptr && cancelFlag->load()) {
      result.cancelled = true;
      result.logs.emplace_back("Render cancelled during import stage.");
      return result;
    }

    const auto& stem = session.stems[i];
    if (!stem.enabled) {
      continue;
    }

    AudioBuffer buffer = fileIO.readAudioFile(stem.filePath);
    if (buffer.getSampleRate() != static_cast<double>(settings.outputSampleRate)) {
      buffer = resampler.resampleLinear(buffer, static_cast<double>(settings.outputSampleRate));
    }

    const auto decisionIt = decisions.find(stem.id);
    const domain::StemMixDecision* decision = decisionIt != decisions.end() ? &decisionIt->second : nullptr;

    stemBuffers.emplace_back(processStemBuffer(buffer, decision, dryWet, 2));

    if (onProgress) {
      onProgress(RenderProgress{.fraction = static_cast<double>(i + 1) / std::max<size_t>(1, session.stems.size()) * 0.5,
                                .stage = "Importing stems"});
    }
  }

  int maxSamples = 0;
  for (const auto& stem : stemBuffers) {
    maxSamples = std::max(maxSamples, stem.getNumSamples());
  }

  if (maxSamples == 0) {
    throw std::runtime_error("No stem audio available to render.");
  }

  result.mixBuffer = AudioBuffer(2, maxSamples, static_cast<double>(settings.outputSampleRate));

  const int blockSize = std::max(1, settings.blockSize);
  const int totalBlocks = (maxSamples + blockSize - 1) / blockSize;

  for (int block = 0; block < totalBlocks; ++block) {
    if (cancelFlag != nullptr && cancelFlag->load()) {
      result.cancelled = true;
      result.logs.emplace_back("Render cancelled during mix stage.");
      return result;
    }

    const int start = block * blockSize;
    const int end = std::min(start + blockSize, maxSamples);

    for (const auto& stem : stemBuffers) {
      for (int sample = start; sample < end; ++sample) {
        if (sample >= stem.getNumSamples()) {
          continue;
        }

        for (int ch = 0; ch < result.mixBuffer.getNumChannels(); ++ch) {
          const float mixed = result.mixBuffer.getSample(ch, sample) + stem.getSample(ch, sample);
          result.mixBuffer.setSample(ch, sample, mixed);
        }
      }
    }

    if (onProgress) {
      const double fraction = 0.5 + (static_cast<double>(block + 1) / std::max(1, totalBlocks) * 0.5);
      onProgress(RenderProgress{.fraction = fraction, .stage = "Summing mix blocks"});
    }
  }

  double peak = 0.0;
  for (int ch = 0; ch < result.mixBuffer.getNumChannels(); ++ch) {
    for (int i = 0; i < result.mixBuffer.getNumSamples(); ++i) {
      peak = std::max(peak, static_cast<double>(std::abs(result.mixBuffer.getSample(ch, i))));
    }
  }

  const double peakTargetLinear = dbToLinear(-std::abs(headroomDb));
  if (peak > peakTargetLinear && peak > 0.0) {
    const float scale = static_cast<float>(peakTargetLinear / peak);
    result.mixBuffer.applyGain(scale);
    result.logs.emplace_back("Applied headroom normalization to respect mix bus headroom.");
  }

  if (session.originalMixPath.has_value() && session.residualBlend > 0.0) {
    try {
      AudioBuffer originalMix = fileIO.readAudioFile(session.originalMixPath.value());
      if (originalMix.getSampleRate() != static_cast<double>(settings.outputSampleRate)) {
        originalMix = resampler.resampleLinear(originalMix, static_cast<double>(settings.outputSampleRate));
      }

      ResidualBlendProcessor residualProcessor;
      const int maxOffsetSamples = std::max(256, settings.outputSampleRate / 4);
      const auto residualResult = residualProcessor.computeResidual(result.mixBuffer, originalMix, maxOffsetSamples);
      result.mixBuffer = residualProcessor.applyResidualBlend(result.mixBuffer,
                                                              residualResult.residual,
                                                              session.residualBlend,
                                                              -1.0);

      result.logs.emplace_back("Residual blend applied at " + std::to_string(session.residualBlend) + "%.");
      result.logs.emplace_back("Original mix alignment offset=" + std::to_string(residualResult.alignment.sampleOffset) +
                               " corr=" + std::to_string(residualResult.alignment.normalizedCorrelation));
    } catch (const std::exception& error) {
      result.logs.emplace_back("Residual blend skipped: " + std::string(error.what()));
    } catch (...) {
      result.logs.emplace_back("Residual blend skipped: unknown error.");
    }
  }

  double finalPeak = 0.0;
  for (int ch = 0; ch < result.mixBuffer.getNumChannels(); ++ch) {
    for (int i = 0; i < result.mixBuffer.getNumSamples(); ++i) {
      finalPeak = std::max(finalPeak, static_cast<double>(std::abs(result.mixBuffer.getSample(ch, i))));
    }
  }

  result.logs.emplace_back("Raw mix render completed.");
  result.logs.emplace_back("Final peak dBFS: " + std::to_string(linearToDb(finalPeak)));
  return result;
}

} // namespace automix::engine
