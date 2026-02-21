#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace automix::app {

/// Custom LookAndFeel for AutoMixMaster.
/// Applies the dark audio-app theme defined in Theme.h to all standard JUCE widgets.
class AutoMixLookAndFeel final : public juce::LookAndFeel_V4 {
public:
  AutoMixLookAndFeel();

  // ── Button ──────────────────────────────────────────────────────
  void drawButtonBackground(juce::Graphics& g, juce::Button& button, const juce::Colour& backgroundColour,
                            bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;

  void drawButtonText(juce::Graphics& g, juce::TextButton& button, bool shouldDrawButtonAsHighlighted,
                      bool shouldDrawButtonAsDown) override;

  // ── Rotary Slider ──────────────────────────────────────────────
  void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height, float sliderPosProportional,
                        float rotaryStartAngle, float rotaryEndAngle, juce::Slider& slider) override;

  // ── Linear Slider ─────────────────────────────────────────────
  void drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height, float sliderPos,
                        float minSliderPos, float maxSliderPos, juce::Slider::SliderStyle style,
                        juce::Slider& slider) override;

  // ── ComboBox ───────────────────────────────────────────────────
  void drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown, int buttonX, int buttonY,
                    int buttonW, int buttonH, juce::ComboBox& box) override;

  // ── Label ──────────────────────────────────────────────────────
  void drawLabel(juce::Graphics& g, juce::Label& label) override;

  // ── Progress Bar ───────────────────────────────────────────────
  void drawProgressBar(juce::Graphics& g, juce::ProgressBar& progressBar, int width, int height,
                       double progress, const juce::String& textToShow) override;

  // ── Toggle Button ──────────────────────────────────────────────
  void drawToggleButton(juce::Graphics& g, juce::ToggleButton& button, bool shouldDrawButtonAsHighlighted,
                        bool shouldDrawButtonAsDown) override;

  // ── Focus ──────────────────────────────────────────────────────
  static void drawFocusRing(juce::Graphics& g, juce::Component& component);

  // ── Font ───────────────────────────────────────────────────────
  juce::Font getTextButtonFont(juce::TextButton& button, int buttonHeight) override;
  juce::Font getComboBoxFont(juce::ComboBox& box) override;
  juce::Font getLabelFont(juce::Label& label) override;
};

} // namespace automix::app
