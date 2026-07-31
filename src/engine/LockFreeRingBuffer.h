#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>

namespace automix::engine {

template <typename T, size_t Capacity>
class LockFreeRingBuffer {
  static_assert(Capacity > 0 && (Capacity & (Capacity - 1)) == 0,
                "LockFreeRingBuffer capacity must be a power of two");

 public:
  LockFreeRingBuffer() noexcept = default;

  LockFreeRingBuffer(const LockFreeRingBuffer&) = delete;
  LockFreeRingBuffer& operator=(const LockFreeRingBuffer&) = delete;

  ~LockFreeRingBuffer() noexcept { clear(); }

  bool tryPush(T value) noexcept(std::is_nothrow_move_constructible_v<T>) {
    const auto h = head_.load(std::memory_order_acquire);
    const auto t = tail_.load(std::memory_order_relaxed);
    if (static_cast<size_t>(t - h) >= Capacity) {
      return false;
    }
    ::new (slotPtr(static_cast<size_t>(t) & kMask_)) T(std::move(value));
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }

  bool tryPop(T& value) noexcept(std::is_nothrow_move_assignable_v<T>) {
    const auto h = head_.load(std::memory_order_relaxed);
    const auto t = tail_.load(std::memory_order_acquire);
    if (h >= t) {
      return false;
    }
    auto* ptr = slotPtr(static_cast<size_t>(h) & kMask_);
    value = std::move(*ptr);
    ptr->~T();
    head_.store(h + 1, std::memory_order_release);
    return true;
  }

  [[nodiscard]] size_t size() const noexcept {
    const auto h = head_.load(std::memory_order_acquire);
    const auto t = tail_.load(std::memory_order_acquire);
    return static_cast<size_t>(static_cast<long long>(t) - static_cast<long long>(h));
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  [[nodiscard]] bool full() const noexcept { return size() >= Capacity; }

  [[nodiscard]] static constexpr size_t capacity() noexcept { return Capacity; }

  void clear() noexcept {
    auto h = head_.load(std::memory_order_relaxed);
    const auto t = tail_.load(std::memory_order_acquire);
    while (h < t) {
      auto* ptr = slotPtr(static_cast<size_t>(h) & kMask_);
      ptr->~T();
      ++h;
    }
    head_.store(t, std::memory_order_release);
  }

 private:
  static constexpr size_t kMask_ = Capacity - 1;
  static constexpr size_t kAlign_ = alignof(T) > alignof(max_align_t) ? alignof(T) : alignof(max_align_t);

  // Aligned storage with proper alignment for T
  alignas(kAlign_) unsigned char storage_[sizeof(T) * Capacity]{};

  std::atomic<long long> head_{0};
  std::atomic<long long> tail_{0};

  T* slotPtr(const size_t index) noexcept {
    return static_cast<T*>(static_cast<void*>(storage_ + index * sizeof(T)));
  }
};

} // namespace automix::engine
