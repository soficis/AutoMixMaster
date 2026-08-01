#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>

#include <juce_events/juce_events.h>

namespace automix::engine {

class TransportController final : public juce::ChangeBroadcaster {
 public:
  enum class State {
    Stopped,
    Playing,
    Paused,
  };

  void setTimeline(int64_t totalSamples, double sampleRate);
  void play();
  void pause();
  void stop();
  void seekToSample(int64_t samplePosition);
  void seekToFraction(double fraction);
  void advance(int numSamples);

  /// Realtime-safe (audio callback): publishes the playhead without locking and
  /// without posting a change message. The caller must pass a position already
  /// clamped to the playback range.
  void setPositionRealtime(int64_t samplePosition);

  /// Realtime-safe (audio callback): ends playback without locking and without
  /// posting a change message. The play button is reconciled by the periodic
  /// UI timer; message-thread state is re-synced by the next user transport action.
  void stopFromAudioThread();
  void setLoopRangeSeconds(double loopInSeconds, double loopOutSeconds, bool enabled);
  void clearLoopRange();

  [[nodiscard]] State state() const;
  [[nodiscard]] bool isPlaying() const;
  [[nodiscard]] int64_t positionSamples() const;
  [[nodiscard]] int64_t totalSamples() const;
  [[nodiscard]] double positionSeconds() const;
  [[nodiscard]] double totalSeconds() const;
  [[nodiscard]] double progress() const;
  [[nodiscard]] bool loopEnabled() const;
  [[nodiscard]] double loopInSeconds() const;
  [[nodiscard]] double loopOutSeconds() const;
  [[nodiscard]] double loopInProgress() const;
  [[nodiscard]] double loopOutProgress() const;

 private:
  // The audio callback reads/writes only the atomics below (lock-free);
  // the mutex protects message-thread-only state (state_, timeline, loop).
  mutable std::mutex mutex_;
  std::atomic<bool> playing_{false};
  std::atomic<int64_t> positionSamples_{0};
  int64_t totalSamples_ = 0;
  int64_t loopStartSamples_ = 0;
  int64_t loopEndSamples_ = 0;
  bool loopEnabled_ = false;
  double sampleRate_ = 44100.0;
  State state_ = State::Stopped;
};

} // namespace automix::engine
