#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>

#include "engine/AudioBuffer.h"

namespace automix::engine {

class BatchStreamer final {
 public:
  explicit BatchStreamer(int chunkSizeSamples = 4096);

  void processChunk(AudioBuffer& buffer,
                    int startSample,
                    int numSamples,
                    const std::function<void(AudioBuffer&, int, int)>& processor);

  void processFull(AudioBuffer& buffer,
                   const std::function<void(AudioBuffer&, int, int)>& processor);

  [[nodiscard]] int getChunkSize() const noexcept;

 private:
  int validateChunkSize(int chunkSizeSamples) const noexcept;

  int chunkSizeSamples_;
};

} // namespace automix::engine
