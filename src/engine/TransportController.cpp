#include "engine/TransportController.h"

#include <algorithm>

namespace automix::engine {

void TransportController::setTimeline(const int64_t totalSamples, const double sampleRate) {
  {
    std::scoped_lock lock(mutex_);
    totalSamples_ = std::max<int64_t>(0, totalSamples);
    sampleRate_ = std::max(8000.0, sampleRate);
    positionSamples_ = std::clamp(positionSamples_, int64_t{0}, totalSamples_);
    if (totalSamples_ == 0) {
      positionSamples_ = 0;
      state_ = State::Stopped;
    }
  }
  sendChangeMessage();
}

void TransportController::play() {
  {
    std::scoped_lock lock(mutex_);
    if (totalSamples_ <= 0) {
      return;
    }
    if (positionSamples_ >= totalSamples_) {
      positionSamples_ = 0;
    }
    state_ = State::Playing;
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
  }
  sendChangeMessage();
}

void TransportController::stop() {
  {
    std::scoped_lock lock(mutex_);
    state_ = State::Stopped;
    positionSamples_ = 0;
  }
  sendChangeMessage();
}

void TransportController::seekToSample(const int64_t samplePosition) {
  {
    std::scoped_lock lock(mutex_);
    positionSamples_ = std::clamp(samplePosition, int64_t{0}, totalSamples_);
    if (positionSamples_ >= totalSamples_ && totalSamples_ > 0 && state_ == State::Playing) {
      state_ = State::Paused;
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
    positionSamples_ = std::min(totalSamples_, positionSamples_ + static_cast<int64_t>(numSamples));
    changed = true;
    if (positionSamples_ >= totalSamples_) {
      state_ = State::Paused;
    }
  }

  if (changed) {
    sendChangeMessage();
  }
}

TransportController::State TransportController::state() const {
  std::scoped_lock lock(mutex_);
  return state_;
}

bool TransportController::isPlaying() const {
  std::scoped_lock lock(mutex_);
  return state_ == State::Playing;
}

int64_t TransportController::positionSamples() const {
  std::scoped_lock lock(mutex_);
  return positionSamples_;
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
  return static_cast<double>(positionSamples_) / sampleRate_;
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
  return std::clamp(static_cast<double>(positionSamples_) / static_cast<double>(totalSamples_), 0.0, 1.0);
}

} // namespace automix::engine
