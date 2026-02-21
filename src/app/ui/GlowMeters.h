#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"

namespace automix::app {

/// Stereo level meters with LUFS readout, true peak, and color gradient.
class GlowMeters final : public juce::Component, private juce::Timer {
public:
  GlowMeters();
  ~GlowMeters() override;

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setLevels(float leftDb, float rightDb);
  void setPeaks(float leftPeakDb, float rightPeakDb);
  void setLufs(double integrated, double shortTerm);
  void setTruePeak(double truePeakDbtp);

private:
  void timerCallback() override;
  void drawMeter(juce::Graphics& g, juce::Rectangle<float> bounds, float levelDb, float peakDb) const;
  static float dbToNormalized(float db);

  // Current smoothed display values
  float leftLevel_ = -60.0f;
  float rightLevel_ = -60.0f;
  float leftPeak_ = -60.0f;
  float rightPeak_ = -60.0f;
  int leftPeakHold_ = 0;
  int rightPeakHold_ = 0;

  // Target values (set from audio thread)
  float targetLeftLevel_ = -60.0f;
  float targetRightLevel_ = -60.0f;
  float targetLeftPeak_ = -60.0f;
  float targetRightPeak_ = -60.0f;

  double integratedLufs_ = -70.0;
  double shortTermLufs_ = -70.0;
  double truePeakDbtp_ = -70.0;

  // Previous rendered values for dirty-checking
  float lastRenderedLeft_ = -60.0f;
  float lastRenderedRight_ = -60.0f;

  // Pre-allocated string buffers
  juce::String lufsText_;
  juce::String stText_;
  juce::String tpText_;

  juce::Label lufsLabel_;
  juce::Label shortTermLabel_;
  juce::Label truePeakLabel_;

  static constexpr float kMinDb = -60.0f;
  static constexpr float kMaxDb = 6.0f;
  static constexpr int kPeakHoldFrames = 40;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GlowMeters)
};

} // namespace automix::app
