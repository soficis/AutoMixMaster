#include "engine/ProgressTracker.h"

#include <algorithm>

namespace automix::engine {

void ProgressTracker::setTotalSteps(const int total) noexcept {
  totalSteps_.store(std::max(0, total), std::memory_order_relaxed);
}

void ProgressTracker::incrementSteps(const int n) noexcept {
  if (n <= 0) {
    return;
  }
  completedSteps_.fetch_add(n, std::memory_order_relaxed);
}

double ProgressTracker::progress() const noexcept {
  const int total = totalSteps_.load(std::memory_order_relaxed);
  if (total <= 0) {
    return 1.0;
  }
  const int completed = completedSteps_.load(std::memory_order_relaxed);
  return std::clamp(static_cast<double>(completed) / static_cast<double>(total),
                    0.0, 1.0);
}

bool ProgressTracker::isComplete() const noexcept {
  const int completed = completedSteps_.load(std::memory_order_relaxed);
  const int total = totalSteps_.load(std::memory_order_relaxed);
  if (total <= 0) {
    return false;
  }
  return completed >= total;
}

void ProgressTracker::reset() noexcept {
  completedSteps_.store(0, std::memory_order_relaxed);
  totalSteps_.store(0, std::memory_order_relaxed);
}

} // namespace automix::engine
