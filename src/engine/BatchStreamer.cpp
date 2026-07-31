#include "engine/BatchStreamer.h"

#include <algorithm>
#include <cstddef>

namespace automix::engine {

BatchStreamer::BatchStreamer(const int chunkSizeSamples)
    : chunkSizeSamples_(validateChunkSize(chunkSizeSamples)) {}

int BatchStreamer::validateChunkSize(const int chunkSizeSamples) const noexcept {
  return std::max(1, chunkSizeSamples);
}

int BatchStreamer::getChunkSize() const noexcept {
  return chunkSizeSamples_;
}

void BatchStreamer::processChunk(
    AudioBuffer& buffer,
    const int startSample,
    const int numSamples,
    const std::function<void(AudioBuffer&, int, int)>& processor) {
  if (!processor) {
    return;
  }
  if (numSamples <= 0) {
    return;
  }
  const int totalSamples = buffer.getNumSamples();
  if (startSample < 0 || startSample >= totalSamples) {
    return;
  }
  const int clampedNumSamples = std::min(numSamples, totalSamples - startSample);
  if (clampedNumSamples <= 0) {
    return;
  }
  processor(buffer, startSample, clampedNumSamples);
}

void BatchStreamer::processFull(
    AudioBuffer& buffer,
    const std::function<void(AudioBuffer&, int, int)>& processor) {
  if (!processor) {
    return;
  }
  const int totalSamples = buffer.getNumSamples();
  if (totalSamples <= 0) {
    return;
  }

  int offset = 0;
  while (offset < totalSamples) {
    const int remaining = totalSamples - offset;
    const int chunkSize = std::min(chunkSizeSamples_, remaining);
    processor(buffer, offset, chunkSize);
    offset += chunkSize;
  }
}

} // namespace automix::engine
