#include "app/ui/TransportBar.h"

#include <cmath>
#include <sstream>

namespace automix::app {

using namespace theme;

TransportBar::TransportBar() {
  setWantsKeyboardFocus(true);

  addAndMakeVisible(skipStartButton_);
  addAndMakeVisible(playPauseButton_);
  addAndMakeVisible(stopButton_);
  addAndMakeVisible(skipEndButton_);
  addAndMakeVisible(timeLabel_);
  addAndMakeVisible(volumeSlider_);
  addAndMakeVisible(loopToggle_);

  // Tooltips
  skipStartButton_.setTooltip("Skip to Start (Home)");
  playPauseButton_.setTooltip("Play / Pause (Space)");
  stopButton_.setTooltip("Stop");
  skipEndButton_.setTooltip("Skip to End (End)");
  loopToggle_.setTooltip("Toggle Loop (L)");
  volumeSlider_.setTooltip("Volume");

  // Enable keyboard focus on buttons
  skipStartButton_.setWantsKeyboardFocus(true);
  playPauseButton_.setWantsKeyboardFocus(true);
  stopButton_.setWantsKeyboardFocus(true);
  skipEndButton_.setWantsKeyboardFocus(true);

  timeLabel_.setText("0:00 / 0:00", juce::dontSendNotification);
  timeLabel_.setFont(typography::body());
  timeLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  timeLabel_.setJustificationType(juce::Justification::centred);

  volumeSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  volumeSlider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
  volumeSlider_.setRange(0.0, 1.0, 0.01);
  volumeSlider_.setValue(1.0, juce::dontSendNotification);
  volumeSlider_.addListener(this);

  skipStartButton_.onClick = [this] {
    if (onSkipToStart)
      onSkipToStart();
  };
  playPauseButton_.onClick = [this] {
    if (isPlaying_) {
      if (onPause)
        onPause();
    } else {
      if (onPlay)
        onPlay();
    }
  };
  stopButton_.onClick = [this] {
    if (onStop)
      onStop();
  };
  skipEndButton_.onClick = [this] {
    if (onSkipToEnd)
      onSkipToEnd();
  };
  loopToggle_.onClick = [this] {
    if (onLoopToggle)
      onLoopToggle();
  };
}

TransportBar::~TransportBar() {
  volumeSlider_.removeListener(this);
}

void TransportBar::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));

  // Top border
  g.setColour(colour(colours::surfaceBorder));
  g.fillRect(0, 0, getWidth(), 1);
}

void TransportBar::resized() {
  auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingMedium));

  // Left: transport buttons
  auto leftArea = area.removeFromLeft(220);
  skipStartButton_.setBounds(leftArea.removeFromLeft(48).reduced(2));
  playPauseButton_.setBounds(leftArea.removeFromLeft(64).reduced(2));
  stopButton_.setBounds(leftArea.removeFromLeft(56).reduced(2));
  skipEndButton_.setBounds(leftArea.removeFromLeft(48).reduced(2));

  // Right: volume + loop
  auto rightArea = area.removeFromRight(200);
  loopToggle_.setBounds(rightArea.removeFromRight(72).reduced(2));
  volumeSlider_.setBounds(rightArea.reduced(2));

  // Center: time display
  timeLabel_.setBounds(area);
}

void TransportBar::setPlaying(bool playing) {
  isPlaying_ = playing;
  playPauseButton_.setButtonText(playing ? "Pause" : "Play");
}

void TransportBar::setTimeDisplay(double currentSeconds, double totalSeconds) {
  timeLabel_.setText(formatTime(currentSeconds) + " / " + formatTime(totalSeconds),
                     juce::dontSendNotification);
}

void TransportBar::setVolume(double volume) {
  volumeSlider_.setValue(volume, juce::dontSendNotification);
}

void TransportBar::setLoopEnabled(bool enabled) {
  loopToggle_.setToggleState(enabled, juce::dontSendNotification);
}

void TransportBar::sliderValueChanged(juce::Slider* slider) {
  if (slider == &volumeSlider_) {
    if (onVolumeChanged)
      onVolumeChanged(volumeSlider_.getValue());
  }
}

juce::String TransportBar::formatTime(double seconds) {
  double clamped = std::max(0.0, seconds);
  int total = static_cast<int>(std::lround(clamped));
  int mins = total / 60;
  int secs = total % 60;
  std::ostringstream out;
  out << mins << ':';
  if (secs < 10)
    out << '0';
  out << secs;
  return juce::String(out.str());
}

bool TransportBar::keyPressed(const juce::KeyPress& key) {
  if (key == juce::KeyPress::spaceKey) {
    if (isPlaying_) {
      if (onPause) onPause();
    } else {
      if (onPlay) onPlay();
    }
    return true;
  }
  if (key == juce::KeyPress::leftKey) {
    if (onSeekRelative) onSeekRelative(-5.0);
    return true;
  }
  if (key == juce::KeyPress::rightKey) {
    if (onSeekRelative) onSeekRelative(5.0);
    return true;
  }
  if (key == juce::KeyPress::upKey) {
    double newVol = std::min(1.0, volumeSlider_.getValue() + 0.05);
    volumeSlider_.setValue(newVol, juce::sendNotificationSync);
    return true;
  }
  if (key == juce::KeyPress::downKey) {
    double newVol = std::max(0.0, volumeSlider_.getValue() - 0.05);
    volumeSlider_.setValue(newVol, juce::sendNotificationSync);
    return true;
  }
  if (key == juce::KeyPress::homeKey) {
    if (onSkipToStart) onSkipToStart();
    return true;
  }
  if (key == juce::KeyPress::endKey) {
    if (onSkipToEnd) onSkipToEnd();
    return true;
  }
  if (key.getTextCharacter() == 'l' || key.getTextCharacter() == 'L') {
    if (onLoopToggle) onLoopToggle();
    return true;
  }
  return false;
}

} // namespace automix::app
