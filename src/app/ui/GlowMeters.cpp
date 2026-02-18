#include "app/ui/GlowMeters.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace automix::app {

using namespace theme;

GlowMeters::GlowMeters() {
  lufsLabel_.setText("LUFS: --", juce::dontSendNotification);
  lufsLabel_.setFont(typography::caption());
  lufsLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  lufsLabel_.setJustificationType(juce::Justification::centredLeft);

  shortTermLabel_.setText("ST: --", juce::dontSendNotification);
  shortTermLabel_.setFont(typography::caption());
  shortTermLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  shortTermLabel_.setJustificationType(juce::Justification::centredLeft);

  truePeakLabel_.setText("TP: --", juce::dontSendNotification);
  truePeakLabel_.setFont(typography::caption());
  truePeakLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  truePeakLabel_.setJustificationType(juce::Justification::centredLeft);

  addAndMakeVisible(lufsLabel_);
  addAndMakeVisible(shortTermLabel_);
  addAndMakeVisible(truePeakLabel_);

  startTimerHz(30);
}

GlowMeters::~GlowMeters() {
  stopTimer();
}

float GlowMeters::dbToNormalized(float db) {
  return std::clamp((db - kMinDb) / (kMaxDb - kMinDb), 0.0f, 1.0f);
}

void GlowMeters::setLevels(float leftDb, float rightDb) {
  targetLeftLevel_ = leftDb;
  targetRightLevel_ = rightDb;
}

void GlowMeters::setPeaks(float leftPeakDb, float rightPeakDb) {
  targetLeftPeak_ = leftPeakDb;
  targetRightPeak_ = rightPeakDb;
}

void GlowMeters::setLufs(double integrated, double shortTerm) {
  integratedLufs_ = integrated;
  shortTermLufs_ = shortTerm;
}

void GlowMeters::setTruePeak(double truePeakDbtp) {
  truePeakDbtp_ = truePeakDbtp;
}

void GlowMeters::timerCallback() {
  // Smooth ballistics: fast attack, slow release
  constexpr float attackCoeff = 0.6f;
  constexpr float releaseCoeff = 0.05f;

  auto smooth = [](float current, float target, float attack, float release) -> float {
    if (target > current)
      return current + (target - current) * attack;
    return current + (target - current) * release;
  };

  leftLevel_ = smooth(leftLevel_, targetLeftLevel_, attackCoeff, releaseCoeff);
  rightLevel_ = smooth(rightLevel_, targetRightLevel_, attackCoeff, releaseCoeff);

  // Peak hold
  if (targetLeftPeak_ > leftPeak_) {
    leftPeak_ = targetLeftPeak_;
    leftPeakHold_ = kPeakHoldFrames;
  } else if (leftPeakHold_ > 0) {
    --leftPeakHold_;
  } else {
    leftPeak_ = smooth(leftPeak_, targetLeftPeak_, 0.01f, 0.02f);
  }

  if (targetRightPeak_ > rightPeak_) {
    rightPeak_ = targetRightPeak_;
    rightPeakHold_ = kPeakHoldFrames;
  } else if (rightPeakHold_ > 0) {
    --rightPeakHold_;
  } else {
    rightPeak_ = smooth(rightPeak_, targetRightPeak_, 0.01f, 0.02f);
  }

  // Update text labels
  std::ostringstream lufsText;
  lufsText.precision(1);
  lufsText << std::fixed << "LUFS: " << integratedLufs_;
  lufsLabel_.setText(lufsText.str(), juce::dontSendNotification);

  std::ostringstream stText;
  stText.precision(1);
  stText << std::fixed << "ST: " << shortTermLufs_;
  shortTermLabel_.setText(stText.str(), juce::dontSendNotification);

  std::ostringstream tpText;
  tpText.precision(1);
  tpText << std::fixed << "TP: " << truePeakDbtp_ << " dBTP";
  truePeakLabel_.setText(tpText.str(), juce::dontSendNotification);

  repaint();
}

void GlowMeters::drawMeter(juce::Graphics& g, juce::Rectangle<float> bounds, float levelDb,
                            float peakDb) const {
  // Background
  g.setColour(colour(colours::meterBackground));
  g.fillRoundedRectangle(bounds, metrics::cornerRadiusSmall);

  float normalizedLevel = dbToNormalized(levelDb);
  float fillHeight = bounds.getHeight() * normalizedLevel;
  auto fillBounds = bounds.withTop(bounds.getBottom() - fillHeight);

  // Color gradient: green -> yellow -> red
  juce::ColourGradient gradient(colour(colours::meterHigh), fillBounds.getTopLeft(),
                                colour(colours::meterLow), fillBounds.getBottomLeft(), false);
  gradient.addColour(0.6, colour(colours::meterMid));
  g.setGradientFill(gradient);
  g.fillRoundedRectangle(fillBounds, metrics::cornerRadiusSmall);

  // Peak hold indicator
  float normalizedPeak = dbToNormalized(peakDb);
  if (normalizedPeak > 0.01f) {
    float peakY = bounds.getBottom() - bounds.getHeight() * normalizedPeak;
    juce::Colour peakColour =
        normalizedPeak > 0.9f ? colour(colours::meterHigh)
                              : (normalizedPeak > 0.6f ? colour(colours::meterMid) : colour(colours::meterLow));
    g.setColour(peakColour);
    g.fillRect(bounds.getX(), peakY - 1.0f, bounds.getWidth(), 2.0f);
  }
}

void GlowMeters::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));

  auto area = getLocalBounds().toFloat().reduced(metrics::paddingSmall);

  // Labels at bottom
  float labelHeight = 54.0f;
  auto meterArea = area.withTrimmedBottom(labelHeight);

  // Two meter bars side by side
  float meterWidth = std::min(24.0f, meterArea.getWidth() * 0.3f);
  float gap = 4.0f;
  float totalMeterWidth = meterWidth * 2.0f + gap;
  float meterX = meterArea.getCentreX() - totalMeterWidth * 0.5f;

  auto leftBounds =
      juce::Rectangle<float>(meterX, meterArea.getY(), meterWidth, meterArea.getHeight());
  auto rightBounds = juce::Rectangle<float>(meterX + meterWidth + gap, meterArea.getY(), meterWidth,
                                            meterArea.getHeight());

  drawMeter(g, leftBounds, leftLevel_, leftPeak_);
  drawMeter(g, rightBounds, rightLevel_, rightPeak_);

  // L/R labels
  g.setColour(colour(colours::textMuted));
  g.setFont(typography::caption());
  g.drawText("L", leftBounds.withHeight(14.0f).translated(0.0f, -14.0f), juce::Justification::centred);
  g.drawText("R", rightBounds.withHeight(14.0f).translated(0.0f, -14.0f), juce::Justification::centred);
}

void GlowMeters::resized() {
  auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingSmall));
  auto labelArea = area.removeFromBottom(54);

  lufsLabel_.setBounds(labelArea.removeFromTop(18));
  shortTermLabel_.setBounds(labelArea.removeFromTop(18));
  truePeakLabel_.setBounds(labelArea.removeFromTop(18));
}

} // namespace automix::app
