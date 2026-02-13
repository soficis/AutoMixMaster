#include "engine/OfflineRenderPipeline.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <mutex>
#include <optional>
#include <thread>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "domain/MixPlan.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/ResidualBlendProcessor.h"

namespace automix::engine {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtHalf = 0.7071067811865476;

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

double linearToDb(const double linear) {
  constexpr double minValue = 1.0e-12;
  return 20.0 * std::log10(std::max(linear, minValue));
}

struct BiquadCoefficients {
  float b0 = 1.0f;
  float b1 = 0.0f;
  float b2 = 0.0f;
  float a1 = 0.0f;
  float a2 = 0.0f;
};

struct StemRenderNode {
  AudioBuffer buffer;
  std::string busId;
};

BiquadCoefficients makeHighPass(const double sampleRate, const double cutoffHz, const double q = kSqrtHalf) {
  const double sr = std::max(8000.0, sampleRate);
  const double safeCutoff = std::clamp(cutoffHz, 10.0, sr * 0.45);
  const double w0 = 2.0 * kPi * safeCutoff / sr;
  const double cosW0 = std::cos(w0);
  const double sinW0 = std::sin(w0);
  const double alpha = sinW0 / (2.0 * std::max(0.05, q));
  const double a0 = 1.0 + alpha;

  BiquadCoefficients coeffs;
  coeffs.b0 = static_cast<float>(((1.0 + cosW0) * 0.5) / a0);
  coeffs.b1 = static_cast<float>((-(1.0 + cosW0)) / a0);
  coeffs.b2 = static_cast<float>(((1.0 + cosW0) * 0.5) / a0);
  coeffs.a1 = static_cast<float>((-2.0 * cosW0) / a0);
  coeffs.a2 = static_cast<float>((1.0 - alpha) / a0);
  return coeffs;
}

BiquadCoefficients makePeakingEq(const double sampleRate,
                                 const double centerHz,
                                 const double q,
                                 const double gainDb) {
  const double sr = std::max(8000.0, sampleRate);
  const double safeCenter = std::clamp(centerHz, 20.0, sr * 0.45);
  const double w0 = 2.0 * kPi * safeCenter / sr;
  const double cosW0 = std::cos(w0);
  const double sinW0 = std::sin(w0);
  const double a = std::pow(10.0, gainDb / 40.0);
  const double alpha = sinW0 / (2.0 * std::max(0.1, q));
  const double a0 = 1.0 + alpha / a;

  BiquadCoefficients coeffs;
  coeffs.b0 = static_cast<float>((1.0 + alpha * a) / a0);
  coeffs.b1 = static_cast<float>((-2.0 * cosW0) / a0);
  coeffs.b2 = static_cast<float>((1.0 - alpha * a) / a0);
  coeffs.a1 = static_cast<float>((-2.0 * cosW0) / a0);
  coeffs.a2 = static_cast<float>((1.0 - alpha / a) / a0);
  return coeffs;
}

void applyBiquad(AudioBuffer& buffer, const BiquadCoefficients& coeffs) {
  if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0) {
    return;
  }

  std::vector<float> z1(static_cast<size_t>(buffer.getNumChannels()), 0.0f);
  std::vector<float> z2(static_cast<size_t>(buffer.getNumChannels()), 0.0f);

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      const float x = buffer.getSample(ch, i);
      const float y = coeffs.b0 * x + z1[static_cast<size_t>(ch)];
      z1[static_cast<size_t>(ch)] = coeffs.b1 * x - coeffs.a1 * y + z2[static_cast<size_t>(ch)];
      z2[static_cast<size_t>(ch)] = coeffs.b2 * x - coeffs.a2 * y;
      buffer.setSample(ch, i, y);
    }
  }
}

void applyHighPass(AudioBuffer& buffer, const double cutoffHz) {
  if (cutoffHz <= 0.0) {
    return;
  }
  applyBiquad(buffer, makeHighPass(buffer.getSampleRate(), cutoffHz));
}

void applyMudCut(AudioBuffer& buffer, const double cutDb) {
  if (std::abs(cutDb) < 1.0e-6) {
    return;
  }
  applyBiquad(buffer, makePeakingEq(buffer.getSampleRate(), 320.0, 0.9, cutDb));
}

void applySimpleCompressor(AudioBuffer& buffer,
                           const double thresholdDb,
                           const double ratio,
                           const double attackMs,
                           const double releaseMs) {
  const float threshold = static_cast<float>(dbToLinear(thresholdDb));
  const float ratioClamped = static_cast<float>(std::clamp(ratio, 1.1, 20.0));
  float envelope = 0.0f;
  const float attackCoeff =
      static_cast<float>(std::exp(-1.0 / std::max(1.0, buffer.getSampleRate() * std::max(0.5, attackMs) * 0.001)));
  const float releaseCoeff =
      static_cast<float>(std::exp(-1.0 / std::max(1.0, buffer.getSampleRate() * std::max(3.0, releaseMs) * 0.001)));

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    float detector = 0.0f;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      detector = std::max(detector, std::abs(buffer.getSample(ch, i)));
    }

    if (detector > envelope) {
      envelope = detector + attackCoeff * (envelope - detector);
    } else {
      envelope = detector + releaseCoeff * (envelope - detector);
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
  const float attackCoeff = static_cast<float>(std::exp(-1.0 / std::max(1.0, buffer.getSampleRate() * 0.010)));
  const float releaseCoeff = static_cast<float>(std::exp(-1.0 / std::max(1.0, buffer.getSampleRate() * 0.130)));

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    float detector = 0.0f;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      detector = std::max(detector, std::abs(buffer.getSample(ch, i)));
    }

    if (detector > envelope) {
      envelope = detector + attackCoeff * (envelope - detector);
    } else {
      envelope = detector + releaseCoeff * (envelope - detector);
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
    applyMudCut(processed, decision->mudCutDb);
    if (decision->enableCompressor) {
      applySimpleCompressor(processed,
                            decision->compressorThresholdDb,
                            decision->compressorRatio,
                            12.0,
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

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string defaultBusIdForStem(const domain::Stem& stem) {
  if (stem.busId.has_value() && !stem.busId->empty()) {
    return stem.busId.value();
  }

  switch (stem.role) {
    case domain::StemRole::Drums:
    case domain::StemRole::Kick:
      return "bus_drums";
    case domain::StemRole::Bass:
      return "bus_bass";
    case domain::StemRole::Vocals:
      return "bus_vocals";
    case domain::StemRole::Fx:
      return "bus_fx";
    default:
      return "bus_music";
  }
}

void applyRoleBusProcessing(const std::string& busId, AudioBuffer& buffer) {
  const auto name = toLower(busId);
  if (name.find("drum") != std::string::npos) {
    applySimpleCompressor(buffer, -16.0, 2.0, 15.0, 120.0);
    return;
  }
  if (name.find("vocal") != std::string::npos) {
    applySimpleCompressor(buffer, -18.0, 1.8, 20.0, 180.0);
    return;
  }
  if (name.find("music") != std::string::npos || name.find("instrument") != std::string::npos) {
    applySimpleCompressor(buffer, -20.0, 1.5, 18.0, 140.0);
  }
}

int effectiveThreadCount(const domain::RenderSettings& settings, const int taskCount) {
  if (!settings.preferHardwareAcceleration) {
    return 1;
  }
  const int hardwareThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const int requested = settings.processingThreads > 0 ? settings.processingThreads : hardwareThreads;
  return std::clamp(requested, 1, std::max(1, taskCount));
}

void addBlock(AudioBuffer& destination,
              const AudioBuffer& source,
              const int startSample,
              const int numSamples,
              const int sourceStartSample = -1) {
  const int srcStart = sourceStartSample >= 0 ? sourceStartSample : startSample;
  if (numSamples <= 0) {
    return;
  }

  const int channels = std::min(destination.getNumChannels(), source.getNumChannels());
  for (int ch = 0; ch < channels; ++ch) {
    float* dst = destination.getWritePointer(ch);
    const float* src = source.getReadPointer(ch);
    for (int i = 0; i < numSamples; ++i) {
      dst[startSample + i] += src[srcStart + i];
    }
  }
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
  std::unordered_map<std::string, double> busGainDbById;
  double dryWet = 1.0;
  double headroomDb = 6.0;

  if (session.mixPlan.has_value()) {
    dryWet = session.mixPlan->dryWet;
    headroomDb = session.mixPlan->mixBusHeadroomDb;
    for (const auto& decision : session.mixPlan->stemDecisions) {
      decisions.emplace(decision.stemId, decision);
    }
  }

  for (const auto& bus : session.buses) {
    busGainDbById[bus.id] = bus.gainDb;
  }

  std::vector<size_t> enabledStemIndices;
  enabledStemIndices.reserve(session.stems.size());
  for (size_t i = 0; i < session.stems.size(); ++i) {
    if (session.stems[i].enabled) {
      enabledStemIndices.push_back(i);
    }
  }

  std::vector<std::optional<StemRenderNode>> stemNodeSlots(enabledStemIndices.size());
  std::atomic<size_t> nextStemIndex{0};
  std::atomic<size_t> processedStemCount{0};
  std::mutex errorMutex;
  std::mutex progressMutex;
  std::vector<std::string> importErrors;
  const int stemThreads = effectiveThreadCount(settings, static_cast<int>(enabledStemIndices.size()));

  const auto importWorker = [&]() {
    AudioFileIO workerFileIO;
    AudioResampler workerResampler;
    for (;;) {
      const size_t slot = nextStemIndex.fetch_add(1);
      if (slot >= enabledStemIndices.size()) {
        break;
      }

      if (cancelFlag != nullptr && cancelFlag->load()) {
        continue;
      }

      const size_t stemIndex = enabledStemIndices[slot];
      const auto& stem = session.stems[stemIndex];

      try {
        AudioBuffer buffer = workerFileIO.readAudioFile(stem.filePath);
        if (buffer.getSampleRate() != static_cast<double>(settings.outputSampleRate)) {
          buffer = workerResampler.resampleLinear(buffer, static_cast<double>(settings.outputSampleRate));
        }

        const auto decisionIt = decisions.find(stem.id);
        const domain::StemMixDecision* decision = decisionIt != decisions.end() ? &decisionIt->second : nullptr;
        const std::string busId = defaultBusIdForStem(stem);

        stemNodeSlots[slot] = StemRenderNode{
            .buffer = processStemBuffer(buffer, decision, dryWet, 2),
            .busId = busId,
        };
      } catch (const std::exception& error) {
        std::scoped_lock lock(errorMutex);
        importErrors.emplace_back("Failed to import stem '" + stem.name + "': " + error.what());
      }

      const size_t done = processedStemCount.fetch_add(1) + 1;
      if (onProgress) {
        const double total = static_cast<double>(std::max<size_t>(1, enabledStemIndices.size()));
        const double fraction = static_cast<double>(done) / total * 0.35;
        std::scoped_lock lock(progressMutex);
        onProgress(RenderProgress{.fraction = fraction, .stage = "Importing stems"});
      }
    }
  };

  if (stemThreads > 1) {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(stemThreads));
    for (int t = 0; t < stemThreads; ++t) {
      workers.emplace_back(importWorker);
    }
    for (auto& worker : workers) {
      worker.join();
    }
  } else {
    importWorker();
  }

  if (cancelFlag != nullptr && cancelFlag->load()) {
    result.cancelled = true;
    result.logs.emplace_back("Render cancelled during import stage.");
    return result;
  }

  if (!importErrors.empty()) {
    for (const auto& error : importErrors) {
      result.logs.push_back(error);
    }
    throw std::runtime_error(importErrors.front());
  }

  std::vector<StemRenderNode> stemNodes;
  stemNodes.reserve(stemNodeSlots.size());
  for (auto& slot : stemNodeSlots) {
    if (slot.has_value()) {
      stemNodes.push_back(std::move(slot.value()));
    }
  }

  int maxSamples = 0;
  for (const auto& node : stemNodes) {
    maxSamples = std::max(maxSamples, node.buffer.getNumSamples());
  }

  if (maxSamples == 0) {
    throw std::runtime_error("No stem audio available to render.");
  }

  std::unordered_set<std::string> busIds;
  for (const auto& node : stemNodes) {
    busIds.insert(node.busId);
  }
  if (busIds.empty()) {
    busIds.insert("bus_music");
  }

  std::unordered_map<std::string, AudioBuffer> busBuffers;
  for (const auto& busId : busIds) {
    busBuffers.emplace(busId, AudioBuffer(2, maxSamples, static_cast<double>(settings.outputSampleRate)));
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

    for (const auto& node : stemNodes) {
      auto busIt = busBuffers.find(node.busId);
      if (busIt == busBuffers.end()) {
        continue;
      }
      auto& busBuffer = busIt->second;
      if (start >= node.buffer.getNumSamples()) {
        continue;
      }
      const int blockSamples = std::min(end, node.buffer.getNumSamples()) - start;
      addBlock(busBuffer, node.buffer, start, blockSamples);
    }

    if (onProgress) {
      const double fraction = 0.35 + (static_cast<double>(block + 1) / std::max(1, totalBlocks) * 0.40);
      onProgress(RenderProgress{.fraction = fraction, .stage = "Summing stem buses"});
    }
  }

  for (auto& [busId, busBuffer] : busBuffers) {
    applyRoleBusProcessing(busId, busBuffer);

    const auto busGainIt = busGainDbById.find(busId);
    if (busGainIt != busGainDbById.end() && std::abs(busGainIt->second) > 1.0e-6) {
      busBuffer.applyGain(static_cast<float>(dbToLinear(busGainIt->second)));
      result.logs.emplace_back("Applied bus gain " + std::to_string(busGainIt->second) + " dB to '" + busId + "'.");
    }

    addBlock(result.mixBuffer, busBuffer, 0, maxSamples, 0);

    if (onProgress) {
      onProgress(RenderProgress{.fraction = 0.75, .stage = "Mixing role buses"});
    }
  }
  result.logs.emplace_back("Summed stems through " + std::to_string(busBuffers.size()) + " role bus(es).");

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
  if (onProgress) {
    onProgress(RenderProgress{.fraction = 1.0, .stage = "Mix render complete"});
  }
  return result;
}

} // namespace automix::engine
