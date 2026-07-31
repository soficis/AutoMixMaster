#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "engine/AudioBuffer.h"

namespace automix::engine {

class AudioBufferPool final : public std::enable_shared_from_this<AudioBufferPool> {
 public:
  static std::shared_ptr<AudioBufferPool> create(size_t maxCachedBuffers = 128);

  ~AudioBufferPool();

  AudioBufferPool(const AudioBufferPool&) = delete;
  AudioBufferPool& operator=(const AudioBufferPool&) = delete;

  std::shared_ptr<AudioBuffer> acquire(int channels, int samples, double sampleRate);

  void clear();

  [[nodiscard]] size_t available() const;
  [[nodiscard]] size_t totalAllocated() const;

 private:
  struct Slot {
    AudioBuffer buffer;
    bool inUse = false;
  };

  explicit AudioBufferPool(size_t maxCachedBuffers);

  static void deleter(AudioBuffer* buffer, std::shared_ptr<AudioBufferPool> self);
  void releaseUnsafe(AudioBuffer* buffer) noexcept;

  std::vector<std::unique_ptr<Slot>> slots_;
  std::unordered_map<AudioBuffer*, Slot*> slotMap_;
  std::vector<Slot*> freeSlots_;
  size_t maxCachedBuffers_;
  mutable std::mutex mutex_;
  std::atomic<size_t> totalAllocated_{0};
};

} // namespace automix::engine
