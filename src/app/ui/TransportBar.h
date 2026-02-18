#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"

namespace automix::app {

/// Playback transport controls: play/pause/stop, time display, volume, loop.
class TransportBar final : public juce::Component, private juce::Slider::Listener {
public:
  TransportBar();
  ~TransportBar() override;

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setPlaying(bool playing);
  void setTimeDisplay(double currentSeconds, double totalSeconds);
  void setVolume(double volume);
  void setLoopEnabled(bool enabled);

  // Callbacks
  std::function<void()> onPlay;
  std::function<void()> onPause;
  std::function<void()> onStop;
  std::function<void()> onSkipToStart;
  std::function<void()> onSkipToEnd;
  std::function<void(double)> onVolumeChanged;
  std::function<void()> onLoopToggle;

private:
  void sliderValueChanged(juce::Slider* slider) override;
  static juce::String formatTime(double seconds);

  juce::TextButton skipStartButton_{"|<"};
  juce::TextButton playPauseButton_{"Play"};
  juce::TextButton stopButton_{"Stop"};
  juce::TextButton skipEndButton_{">|"};
  juce::Label timeLabel_;
  juce::Slider volumeSlider_;
  juce::ToggleButton loopToggle_{"Loop"};

  bool isPlaying_ = false;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TransportBar)
};

} // namespace automix::app
