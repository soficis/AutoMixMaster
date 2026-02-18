#include "engine/LoudnessMeter.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef ENABLE_LIBEBUR128
#include <ebur128.h>
#endif

namespace automix::engine {
namespace {

double linearToDb(const double linear) {
  constexpr double minValue = 1.0e-12;
  return 20.0 * std::log10(std::max(linear, minValue));
}

double fallbackIntegratedLufs(const AudioBuffer& buffer) {
  if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0) {
    return -120.0;
  }

  double sum = 0.0;
  const int channels = buffer.getNumChannels();
  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    double mono = 0.0;
    for (int ch = 0; ch < channels; ++ch) {
      mono += buffer.getSample(ch, i);
    }
    mono /= static_cast<double>(channels);
    sum += mono * mono;
  }

  const double meanSquare = sum / static_cast<double>(std::max(1, buffer.getNumSamples()));
  return -0.691 + 10.0 * std::log10(std::max(meanSquare, 1.0e-12));
}

} // namespace

LoudnessMetrics LoudnessMeter::analyze(const AudioBuffer& buffer, const size_t chunkSize) const {
  LoudnessMetrics metrics;
  metrics.samplePeakDbfs = computeSamplePeakDbfs(buffer);

  if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0) {
    return metrics;
  }

#ifdef ENABLE_LIBEBUR128
  const int channelCount = std::clamp(buffer.getNumChannels(), 1, 2);
  ebur128_state* state =
      ebur128_init(static_cast<unsigned int>(channelCount),
                   static_cast<unsigned long>(std::max(1.0, buffer.getSampleRate())),
                   EBUR128_MODE_I | EBUR128_MODE_S | EBUR128_MODE_LRA);
  if (state == nullptr) {
    metrics.integratedLufs = fallbackIntegratedLufs(buffer);
    metrics.shortTermLufs = metrics.integratedLufs;
    return metrics;
  }

  ebur128_set_channel(state, 0, EBUR128_LEFT);
  if (channelCount > 1) {
    ebur128_set_channel(state, 1, EBUR128_RIGHT);
  }

  const size_t framesPerChunk = std::max<size_t>(1, chunkSize);
  std::vector<double> interleaved;
  interleaved.resize(framesPerChunk * static_cast<size_t>(channelCount));

  for (int offset = 0; offset < buffer.getNumSamples(); offset += static_cast<int>(framesPerChunk)) {
    const int frames = std::min(static_cast<int>(framesPerChunk), buffer.getNumSamples() - offset);
    for (int frame = 0; frame < frames; ++frame) {
      for (int ch = 0; ch < channelCount; ++ch) {
        interleaved[static_cast<size_t>(frame * channelCount + ch)] = buffer.getSample(ch, offset + frame);
      }
    }
    ebur128_add_frames_double(state, interleaved.data(), static_cast<size_t>(frames));
  }

  double integrated = -120.0;
  if (ebur128_loudness_global(state, &integrated) == EBUR128_SUCCESS && std::isfinite(integrated)) {
    metrics.integratedLufs = integrated;
  } else {
    metrics.integratedLufs = fallbackIntegratedLufs(buffer);
  }

  double shortTerm = -120.0;
  if (ebur128_loudness_shortterm(state, &shortTerm) == EBUR128_SUCCESS && std::isfinite(shortTerm)) {
    metrics.shortTermLufs = shortTerm;
  } else {
    metrics.shortTermLufs = metrics.integratedLufs;
  }

  double lra = 0.0;
  if (ebur128_loudness_range(state, &lra) == EBUR128_SUCCESS && std::isfinite(lra)) {
    metrics.loudnessRange = lra;
  }

  ebur128_destroy(&state);
  return metrics;
#else
  metrics.integratedLufs = fallbackIntegratedLufs(buffer);
  metrics.shortTermLufs = metrics.integratedLufs;
  metrics.loudnessRange = 0.0;
  return metrics;
#endif
}

double LoudnessMeter::computeIntegratedLufs(const AudioBuffer& buffer, const size_t chunkSize) const {
  return analyze(buffer, chunkSize).integratedLufs;
}

double LoudnessMeter::computeShortTermLufs(const AudioBuffer& buffer, const size_t chunkSize) const {
  return analyze(buffer, chunkSize).shortTermLufs;
}

double LoudnessMeter::computeLoudnessRange(const AudioBuffer& buffer, const size_t chunkSize) const {
  return analyze(buffer, chunkSize).loudnessRange;
}

double LoudnessMeter::computeSamplePeakDbfs(const AudioBuffer& buffer) const {
  double peak = 0.0;
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      peak = std::max(peak, static_cast<double>(std::abs(buffer.getSample(ch, i))));
    }
  }
  return linearToDb(peak);
}

} // namespace automix::engine
