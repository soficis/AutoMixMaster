#include "engine/AudioBufferPool.h"

#include <algorithm>
#include <utility>

namespace automix::engine {

std::shared_ptr<AudioBufferPool> AudioBufferPool::create(const size_t maxCachedBuffers) {
  return std::shared_ptr<AudioBufferPool>(new AudioBufferPool(maxCachedBuffers));
}

AudioBufferPool::AudioBufferPool(const size_t maxCachedBuffers)
    : maxCachedBuffers_(std::max<size_t>(maxCachedBuffers, size_t{1})) {}

AudioBufferPool::~AudioBufferPool() = default;

void AudioBufferPool::deleter(AudioBuffer* const buffer,
                               const std::shared_ptr<AudioBufferPool> self) {
  std::scoped_lock lock(self->mutex_);
  self->releaseUnsafe(buffer);
}

void AudioBufferPool::releaseUnsafe(AudioBuffer* const buffer) noexcept {
  const auto it = slotMap_.find(buffer);
  if (it == slotMap_.end()) {
    return;
  }

  auto* slot = it->second;
  if (!slot->inUse) {
    return;
  }
  slot->buffer.clear();
  slot->inUse = false;
  freeSlots_.push_back(slot);
}

std::shared_ptr<AudioBuffer> AudioBufferPool::acquire(const int channels,
                                                       const int samples,
                                                       const double sampleRate) {
  Slot* slot = nullptr;
  {
    std::scoped_lock lock(mutex_);

    for (auto it = freeSlots_.begin(); it != freeSlots_.end(); ++it) {
      auto* candidate = *it;
      if (candidate->buffer.getNumChannels() == channels &&
          candidate->buffer.getNumSamples() >= samples) {
        slot = candidate;
        freeSlots_.erase(it);
        break;
      }
    }

    if (slot == nullptr) {
      const int allocSamples = std::max(samples, 1024);
      auto newSlot = std::make_unique<Slot>();
      newSlot->buffer = AudioBuffer(channels, allocSamples, sampleRate);
      newSlot->inUse = false;

      slot = newSlot.get();
      slotMap_[&slot->buffer] = slot;
      slots_.push_back(std::move(newSlot));
    }

    slot->inUse = true;
  }

  totalAllocated_.fetch_add(1, std::memory_order_relaxed);

  auto self = shared_from_this();
  return std::shared_ptr<AudioBuffer>(
      &slot->buffer,
      [self](AudioBuffer* buf) mutable {
        deleter(buf, std::move(self));
      });
}

void AudioBufferPool::clear() {
  std::scoped_lock lock(mutex_);

  // Destroy only free (not in-use) slots to avoid dangling pointers
  for (auto it = slots_.begin(); it != slots_.end();) {
    auto* slot = it->get();
    if (!slot->inUse) {
      slotMap_.erase(&slot->buffer);
      it = slots_.erase(it);
    } else {
      ++it;
    }
  }
  freeSlots_.clear();
  totalAllocated_.store(0, std::memory_order_relaxed);
}

size_t AudioBufferPool::available() const {
  std::scoped_lock lock(mutex_);
  return freeSlots_.size();
}

size_t AudioBufferPool::totalAllocated() const {
  return totalAllocated_.load(std::memory_order_relaxed);
}

} // namespace automix::engine
