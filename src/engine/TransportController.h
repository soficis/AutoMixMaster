#pragma once

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

  [[nodiscard]] State state() const;
  [[nodiscard]] bool isPlaying() const;
  [[nodiscard]] int64_t positionSamples() const;
  [[nodiscard]] int64_t totalSamples() const;
  [[nodiscard]] double positionSeconds() const;
  [[nodiscard]] double totalSeconds() const;
  [[nodiscard]] double progress() const;

 private:
  mutable std::mutex mutex_;
  int64_t totalSamples_ = 0;
  int64_t positionSamples_ = 0;
  double sampleRate_ = 44100.0;
  State state_ = State::Stopped;
};

} // namespace automix::engine
