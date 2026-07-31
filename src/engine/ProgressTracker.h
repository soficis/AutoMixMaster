#pragma once

#include <atomic>
#include <cstddef>

namespace automix::engine {

class ProgressTracker final {
 public:
  ProgressTracker() noexcept = default;

  void setTotalSteps(int total) noexcept;

  void incrementSteps(int n = 1) noexcept;

  [[nodiscard]] double progress() const noexcept;

  [[nodiscard]] bool isComplete() const noexcept;

  void reset() noexcept;

 private:
  std::atomic<int> completedSteps_{0};
  std::atomic<int> totalSteps_{0};
};

} // namespace automix::engine
