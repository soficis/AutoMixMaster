#include "engine/TransportController.h"

#include <algorithm>
#include <cmath>

namespace automix::engine {

void TransportController::setTimeline(const int64_t totalSamples, const double sampleRate) {
  {
    std::scoped_lock lock(mutex_);
    totalSamples_ = std::max<int64_t>(0, totalSamples);
    sampleRate_ = std::max(8000.0, sampleRate);
    positionSamples_.store(
        std::clamp(positionSamples_.load(std::memory_order_relaxed), int64_t{0}, totalSamples_),
        std::memory_order_relaxed);
    loopStartSamples_ = std::clamp(loopStartSamples_, int64_t{0}, totalSamples_);
    loopEndSamples_ = std::clamp(loopEndSamples_, int64_t{0}, totalSamples_);
    if (loopEndSamples_ <= loopStartSamples_) {
      loopEnabled_ = false;
    }
    if (totalSamples_ == 0) {
      positionSamples_.store(0, std::memory_order_relaxed);
      loopStartSamples_ = 0;
      loopEndSamples_ = 0;
      loopEnabled_ = false;
      state_ = State::Stopped;
    }
    playing_.store(state_ == State::Playing, std::memory_order_release);
  }
  sendChangeMessage();
}

void TransportController::play() {
  {
    std::scoped_lock lock(mutex_);
    if (totalSamples_ <= 0) {
      return;
    }
    if (positionSamples_.load(std::memory_order_relaxed) >= totalSamples_) {
      positionSamples_.store(0, std::memory_order_relaxed);
    }
    state_ = State::Playing;
    playing_.store(true, std::memory_order_release);
  }
  sendChangeMessage();
}

void TransportController::pause() {
  {
    std::scoped_lock lock(mutex_);
    if (state_ != State::Playing) {
      return;
    }
    state_ = State::Paused;
    playing_.store(false, std::memory_order_release);
  }
  sendChangeMessage();
}

void TransportController::stop() {
  {
    std::scoped_lock lock(mutex_);
    state_ = State::Stopped;
    positionSamples_.store(0, std::memory_order_relaxed);
    playing_.store(false, std::memory_order_release);
  }
  sendChangeMessage();
}

void TransportController::seekToSample(const int64_t samplePosition) {
  {
    std::scoped_lock lock(mutex_);
    positionSamples_.store(
        std::clamp(samplePosition, int64_t{0}, totalSamples_), std::memory_order_relaxed);
    if (loopEnabled_ && loopEndSamples_ > loopStartSamples_ &&
        positionSamples_.load(std::memory_order_relaxed) > loopEndSamples_) {
      positionSamples_.store(loopEndSamples_, std::memory_order_relaxed);
    }
    if (positionSamples_.load(std::memory_order_relaxed) >= totalSamples_ && totalSamples_ > 0 &&
        state_ == State::Playing) {
      state_ = State::Paused;
      playing_.store(false, std::memory_order_release);
    }
  }
  sendChangeMessage();
}

void TransportController::seekToFraction(const double fraction) {
  int64_t samplePosition = 0;
  {
    std::scoped_lock lock(mutex_);
    const double clamped = std::clamp(fraction, 0.0, 1.0);
    samplePosition = static_cast<int64_t>(clamped * static_cast<double>(totalSamples_));
  }
  seekToSample(samplePosition);
}

void TransportController::advance(const int numSamples) {
  bool changed = false;
  {
    std::scoped_lock lock(mutex_);
    if (state_ != State::Playing || totalSamples_ <= 0 || numSamples <= 0) {
      return;
    }
    const auto advanced =
        positionSamples_.load(std::memory_order_relaxed) + static_cast<int64_t>(numSamples);
    if (loopEnabled_ && loopEndSamples_ > loopStartSamples_) {
      if (advanced >= loopEndSamples_) {
        const int64_t loopLength = std::max<int64_t>(1, loopEndSamples_ - loopStartSamples_);
        const int64_t overshoot = advanced - loopEndSamples_;
        positionSamples_.store(loopStartSamples_ + (overshoot % loopLength),
                               std::memory_order_relaxed);
      } else {
        positionSamples_.store(advanced, std::memory_order_relaxed);
      }
    } else {
      positionSamples_.store(std::min(totalSamples_, advanced), std::memory_order_relaxed);
    }
    changed = true;
    if (!loopEnabled_ && positionSamples_.load(std::memory_order_relaxed) >= totalSamples_) {
      state_ = State::Paused;
      playing_.store(false, std::memory_order_release);
    }
  }

  if (changed) {
    sendChangeMessage();
  }
}

void TransportController::setPositionRealtime(const int64_t samplePosition) {
  positionSamples_.store(samplePosition, std::memory_order_relaxed);
}

void TransportController::stopFromAudioThread() {
  positionSamples_.store(0, std::memory_order_relaxed);
  playing_.store(false, std::memory_order_release);
}

void TransportController::setLoopRangeSeconds(const double loopInSeconds,
                                              const double loopOutSeconds,
                                              const bool enabled) {
  {
    std::scoped_lock lock(mutex_);
    const auto toSample = [this](const double seconds) {
      return static_cast<int64_t>(std::llround(std::max(0.0, seconds) * sampleRate_));
    };
    loopStartSamples_ = std::clamp(toSample(loopInSeconds), int64_t{0}, totalSamples_);
    loopEndSamples_ = std::clamp(toSample(loopOutSeconds), int64_t{0}, totalSamples_);
    loopEnabled_ = enabled && loopEndSamples_ > loopStartSamples_;

    if (loopEnabled_) {
      positionSamples_.store(
          std::clamp(positionSamples_.load(std::memory_order_relaxed), loopStartSamples_,
                     loopEndSamples_),
          std::memory_order_relaxed);
    }
  }
  sendChangeMessage();
}

void TransportController::clearLoopRange() {
  {
    std::scoped_lock lock(mutex_);
    loopEnabled_ = false;
    loopStartSamples_ = 0;
    loopEndSamples_ = 0;
  }
  sendChangeMessage();
}

TransportController::State TransportController::state() const {
  std::scoped_lock lock(mutex_);
  return state_;
}

bool TransportController::isPlaying() const {
  return playing_.load(std::memory_order_acquire);
}

int64_t TransportController::positionSamples() const {
  return positionSamples_.load(std::memory_order_relaxed);
}

int64_t TransportController::totalSamples() const {
  std::scoped_lock lock(mutex_);
  return totalSamples_;
}

double TransportController::positionSeconds() const {
  std::scoped_lock lock(mutex_);
  if (sampleRate_ <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(positionSamples_.load(std::memory_order_relaxed)) / sampleRate_;
}

double TransportController::totalSeconds() const {
  std::scoped_lock lock(mutex_);
  if (sampleRate_ <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(totalSamples_) / sampleRate_;
}

double TransportController::progress() const {
  std::scoped_lock lock(mutex_);
  if (totalSamples_ <= 0) {
    return 0.0;
  }
  return std::clamp(
      static_cast<double>(positionSamples_.load(std::memory_order_relaxed)) /
          static_cast<double>(totalSamples_),
      0.0, 1.0);
}

bool TransportController::loopEnabled() const {
  std::scoped_lock lock(mutex_);
  return loopEnabled_;
}

double TransportController::loopInSeconds() const {
  std::scoped_lock lock(mutex_);
  if (sampleRate_ <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(loopStartSamples_) / sampleRate_;
}

double TransportController::loopOutSeconds() const {
  std::scoped_lock lock(mutex_);
  if (sampleRate_ <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(loopEndSamples_) / sampleRate_;
}

double TransportController::loopInProgress() const {
  std::scoped_lock lock(mutex_);
  if (totalSamples_ <= 0) {
    return 0.0;
  }
  return std::clamp(static_cast<double>(loopStartSamples_) / static_cast<double>(totalSamples_), 0.0, 1.0);
}

double TransportController::loopOutProgress() const {
  std::scoped_lock lock(mutex_);
  if (totalSamples_ <= 0) {
    return 0.0;
  }
  return std::clamp(static_cast<double>(loopEndSamples_) / static_cast<double>(totalSamples_), 0.0, 1.0);
}

} // namespace automix::engine
