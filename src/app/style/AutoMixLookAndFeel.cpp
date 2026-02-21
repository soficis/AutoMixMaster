#include "AutoMixLookAndFeel.h"
#include "Theme.h"

namespace automix::app {

using namespace theme;

// ─────────────────────────────────────────────────────────────────
// Constructor — set the base colour scheme
// ─────────────────────────────────────────────────────────────────

AutoMixLookAndFeel::AutoMixLookAndFeel() {
  // Base dark colour scheme
  setColourScheme({
      colour(colours::background),   // windowBackground
      colour(colours::surface),      // widgetBackground
      colour(colours::surface),      // menuBackground
      colour(colours::textMuted),    // outline
      colour(colours::text),         // defaultText
      colour(colours::surfaceLight), // defaultFill
      colour(colours::text),         // highlightedText
      colour(colours::primary),      // highlightedFill
      colour(colours::text),         // menuText
  });

  // Additional component colours
  setColour(juce::TextButton::buttonColourId, colour(colours::primary));
  setColour(juce::TextButton::buttonOnColourId, colour(colours::primaryPressed));
  setColour(juce::TextButton::textColourOffId, colour(colours::text));
  setColour(juce::TextButton::textColourOnId, colour(colours::text));

  setColour(juce::ComboBox::backgroundColourId, colour(colours::surface));
  setColour(juce::ComboBox::textColourId, colour(colours::text));
  setColour(juce::ComboBox::outlineColourId, colour(colours::surfaceBorder));
  setColour(juce::ComboBox::arrowColourId, colour(colours::textMuted));

  setColour(juce::Slider::thumbColourId, colour(colours::primary));
  setColour(juce::Slider::trackColourId, colour(colours::surfaceBorder));
  setColour(juce::Slider::rotarySliderFillColourId, colour(colours::primary));
  setColour(juce::Slider::backgroundColourId, colour(colours::surface));

  setColour(juce::Label::textColourId, colour(colours::text));
  setColour(juce::Label::backgroundColourId, juce::Colour(0x00000000));

  setColour(juce::TextEditor::backgroundColourId, colour(colours::surface));
  setColour(juce::TextEditor::textColourId, colour(colours::text));
  setColour(juce::TextEditor::outlineColourId, colour(colours::surfaceBorder));
  setColour(juce::TextEditor::focusedOutlineColourId, colour(colours::primary));

  setColour(juce::ListBox::backgroundColourId, colour(colours::background));
  setColour(juce::ListBox::textColourId, colour(colours::text));

  setColour(juce::ScrollBar::thumbColourId, colour(colours::surfaceLight));

  setColour(juce::PopupMenu::backgroundColourId, colour(colours::surface));
  setColour(juce::PopupMenu::textColourId, colour(colours::text));
  setColour(juce::PopupMenu::highlightedBackgroundColourId, colour(colours::primary));
  setColour(juce::PopupMenu::highlightedTextColourId, colour(colours::text));

  setColour(juce::ProgressBar::backgroundColourId, colour(colours::surface));
  setColour(juce::ProgressBar::foregroundColourId, colour(colours::primary));

  setColour(juce::AlertWindow::backgroundColourId, colour(colours::surface));
  setColour(juce::AlertWindow::textColourId, colour(colours::text));
  setColour(juce::AlertWindow::outlineColourId, colour(colours::surfaceBorder));
}

// ─────────────────────────────────────────────────────────────────
// Button
// ─────────────────────────────────────────────────────────────────

void AutoMixLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                              const juce::Colour& /*backgroundColour*/,
                                              bool shouldDrawButtonAsHighlighted,
                                              bool shouldDrawButtonAsDown) {
  auto bounds = button.getLocalBounds().toFloat().reduced(0.5f);
  auto cornerSize = metrics::cornerRadius;

  juce::Colour bg;
  if (!button.isEnabled()) {
    bg = colour(colours::surfaceBorder);
  } else if (shouldDrawButtonAsDown) {
    bg = colour(colours::primaryPressed);
  } else if (shouldDrawButtonAsHighlighted) {
    bg = colour(colours::primary).interpolatedWith(colour(colours::primaryHover), 0.6f);
  } else if (button.getToggleState()) {
    bg = colour(colours::primary);
  } else {
    bg = colour(colours::primary);
  }

  g.setColour(bg);
  g.fillRoundedRectangle(bounds, cornerSize);

  // Focus ring
  drawFocusRing(g, button);
}

void AutoMixLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button,
                                        bool /*shouldDrawButtonAsHighlighted*/,
                                        bool /*shouldDrawButtonAsDown*/) {
  auto font = getTextButtonFont(button, button.getHeight());
  g.setFont(font);

  auto textColour = button.isEnabled() ? colour(colours::text) : colour(colours::textDisabled);

  g.setColour(textColour);

  auto yIndent = juce::jmin(4, button.proportionOfHeight(0.3f));
  auto leftIndent = juce::jmin(static_cast<int>(metrics::paddingMedium), button.proportionOfWidth(0.1f));
  auto textWidth = button.getWidth() - 2 * leftIndent;
  auto textHeight = button.getHeight() - 2 * yIndent;

  if (textWidth > 0)
    g.drawFittedText(button.getButtonText(), leftIndent, yIndent, textWidth, textHeight,
                     juce::Justification::centred, 2);
}

// ─────────────────────────────────────────────────────────────────
// Rotary Slider
// ─────────────────────────────────────────────────────────────────

void AutoMixLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                          float sliderPosProportional, float rotaryStartAngle,
                                          float rotaryEndAngle, juce::Slider& slider) {
  auto bounds = juce::Rectangle<int>(x, y, width, height).toFloat().reduced(2.0f);
  auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
  auto centreX = bounds.getCentreX();
  auto centreY = bounds.getCentreY();
  auto rx = centreX - radius;
  auto ry = centreY - radius;
  auto rw = radius * 2.0f;
  auto angle = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

  // Background track arc
  juce::Path backgroundArc;
  backgroundArc.addCentredArc(centreX, centreY, radius - 2.0f, radius - 2.0f, 0.0f, rotaryStartAngle,
                              rotaryEndAngle, true);
  g.setColour(colour(colours::surfaceBorder));
  g.strokePath(backgroundArc,
               juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

  if (slider.isEnabled()) {
    // Value arc
    juce::Path valueArc;
    valueArc.addCentredArc(centreX, centreY, radius - 2.0f, radius - 2.0f, 0.0f, rotaryStartAngle, angle,
                           true);
    g.setColour(colour(colours::primary));
    g.strokePath(valueArc,
                 juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
  }

  // Thumb dot
  juce::Point<float> thumbPoint(
      centreX + (radius - 6.0f) * std::cos(angle - juce::MathConstants<float>::halfPi),
      centreY + (radius - 6.0f) * std::sin(angle - juce::MathConstants<float>::halfPi));

  g.setColour(slider.isEnabled() ? colour(colours::text) : colour(colours::textDisabled));
  g.fillEllipse(juce::Rectangle<float>(8.0f, 8.0f).withCentre(thumbPoint));
}

// ─────────────────────────────────────────────────────────────────
// Linear Slider
// ─────────────────────────────────────────────────────────────────

void AutoMixLookAndFeel::drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                                          float sliderPos, float /*minSliderPos*/, float /*maxSliderPos*/,
                                          juce::Slider::SliderStyle style, juce::Slider& slider) {
  if (style == juce::Slider::LinearBarVertical || style == juce::Slider::LinearBar) {
    // Fallback to default for bar style
    LookAndFeel_V4::drawLinearSlider(g, x, y, width, height, sliderPos, 0.0f, 0.0f, style, slider);
    return;
  }

  bool isHorizontal = (style == juce::Slider::LinearHorizontal || style == juce::Slider::TwoValueHorizontal ||
                       style == juce::Slider::ThreeValueHorizontal);

  auto trackWidth = metrics::sliderTrackHeight;

  if (isHorizontal) {
    auto trackY = static_cast<float>(y) + static_cast<float>(height) * 0.5f - trackWidth * 0.5f;

    // Background track
    g.setColour(colour(colours::surfaceBorder));
    g.fillRoundedRectangle(static_cast<float>(x), trackY, static_cast<float>(width), trackWidth,
                           trackWidth * 0.5f);

    // Active track
    if (slider.isEnabled()) {
      g.setColour(colour(colours::primary));
      g.fillRoundedRectangle(static_cast<float>(x), trackY, sliderPos - static_cast<float>(x), trackWidth,
                             trackWidth * 0.5f);
    }

    // Thumb — enlarged on hover
    auto thumbRadius = metrics::sliderThumbRadius;
    if (slider.isMouseOverOrDragging())
      thumbRadius *= 1.2f;
    auto thumbY = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
    g.setColour(slider.isEnabled() ? colour(colours::text) : colour(colours::textDisabled));
    g.fillEllipse(sliderPos - thumbRadius, thumbY - thumbRadius, thumbRadius * 2.0f, thumbRadius * 2.0f);
  } else {
    auto trackX = static_cast<float>(x) + static_cast<float>(width) * 0.5f - trackWidth * 0.5f;

    // Background track
    g.setColour(colour(colours::surfaceBorder));
    g.fillRoundedRectangle(trackX, static_cast<float>(y), trackWidth, static_cast<float>(height),
                           trackWidth * 0.5f);

    // Active track
    if (slider.isEnabled()) {
      g.setColour(colour(colours::primary));
      g.fillRoundedRectangle(trackX, sliderPos, trackWidth, static_cast<float>(y + height) - sliderPos,
                             trackWidth * 0.5f);
    }

    // Thumb — enlarged on hover
    auto thumbRadius = metrics::sliderThumbRadius;
    if (slider.isMouseOverOrDragging())
      thumbRadius *= 1.2f;
    auto thumbX = static_cast<float>(x) + static_cast<float>(width) * 0.5f;
    g.setColour(slider.isEnabled() ? colour(colours::text) : colour(colours::textDisabled));
    g.fillEllipse(thumbX - thumbRadius, sliderPos - thumbRadius, thumbRadius * 2.0f, thumbRadius * 2.0f);
  }
}

// ─────────────────────────────────────────────────────────────────
// ComboBox
// ─────────────────────────────────────────────────────────────────

void AutoMixLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                                      int /*buttonX*/, int /*buttonY*/, int /*buttonW*/, int /*buttonH*/,
                                      juce::ComboBox& box) {
  auto bounds = juce::Rectangle<int>(0, 0, width, height).toFloat();
  auto cornerSize = metrics::cornerRadius;

  g.setColour(colour(colours::surface));
  g.fillRoundedRectangle(bounds, cornerSize);

  g.setColour(box.hasKeyboardFocus(true) ? colour(colours::primary) : colour(colours::surfaceBorder));
  g.drawRoundedRectangle(bounds.reduced(0.5f), cornerSize, metrics::borderWidth);

  // Arrow indicator
  auto arrowZone =
      juce::Rectangle<float>(static_cast<float>(width) - 24.0f, 0.0f, 20.0f, static_cast<float>(height));
  juce::Path arrow;
  auto arrowCentre = arrowZone.getCentre();
  arrow.addTriangle(arrowCentre.x - 4.0f, arrowCentre.y - 2.0f, arrowCentre.x + 4.0f, arrowCentre.y - 2.0f,
                    arrowCentre.x, arrowCentre.y + 3.0f);

  g.setColour(box.isEnabled() ? colour(colours::textMuted) : colour(colours::textDisabled));
  g.fillPath(arrow);
}

// ─────────────────────────────────────────────────────────────────
// Label
// ─────────────────────────────────────────────────────────────────

void AutoMixLookAndFeel::drawLabel(juce::Graphics& g, juce::Label& label) {
  g.fillAll(label.findColour(juce::Label::backgroundColourId));

  if (!label.isBeingEdited()) {
    auto textColour = label.findColour(juce::Label::textColourId);
    auto textArea = getLabelBorderSize(label).subtractedFrom(label.getLocalBounds());

    g.setColour(textColour);
    g.setFont(getLabelFont(label));
    g.drawFittedText(label.getText(), textArea, label.getJustificationType(),
                     juce::jmax(1, static_cast<int>(static_cast<float>(textArea.getHeight()) /
                                                    getLabelFont(label).getHeight())),
                     label.getMinimumHorizontalScale());
  }
}

// ─────────────────────────────────────────────────────────────────
// Progress Bar
// ─────────────────────────────────────────────────────────────────

void AutoMixLookAndFeel::drawProgressBar(juce::Graphics& g, juce::ProgressBar& /*progressBar*/, int width,
                                         int height, double progress, const juce::String& textToShow) {
  auto bounds = juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
  auto cornerSize = metrics::cornerRadiusSmall;

  // Background
  g.setColour(colour(colours::surface));
  g.fillRoundedRectangle(bounds, cornerSize);

  // Fill
  if (progress >= 0.0 && progress <= 1.0) {
    auto fillWidth = static_cast<float>(progress) * bounds.getWidth();
    auto fillBounds = bounds.withWidth(fillWidth);

    juce::ColourGradient gradient(colour(colours::primary), fillBounds.getTopLeft(),
                                  colour(colours::secondary), fillBounds.getTopRight(), false);
    g.setGradientFill(gradient);
    g.fillRoundedRectangle(fillBounds, cornerSize);
  } else {
    // Indeterminate — animated sweep
    auto pos = static_cast<float>(std::fmod(juce::Time::getMillisecondCounter() / 1000.0, 1.0));
    auto sw = bounds.getWidth() * 0.3f;
    auto sx = bounds.getX() + pos * (bounds.getWidth() + sw) - sw;
    g.setColour(colour(colours::primary));
    g.fillRoundedRectangle(juce::jmax(bounds.getX(), sx), bounds.getY(),
                           juce::jmin(sw, bounds.getRight() - sx), bounds.getHeight(), cornerSize);
  }

  // Text overlay
  if (textToShow.isNotEmpty()) {
    g.setColour(colour(colours::text));
    g.setFont(typography::caption());
    g.drawText(textToShow, bounds, juce::Justification::centred, false);
  }
}

// ─────────────────────────────────────────────────────────────────
// Toggle Button
// ─────────────────────────────────────────────────────────────────

void AutoMixLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                                          bool shouldDrawButtonAsHighlighted,
                                          bool /*shouldDrawButtonAsDown*/) {
  auto fontSize = typography::bodySize;
  auto tickWidth = fontSize * 1.2f;

  auto tickBounds = juce::Rectangle<float>(metrics::paddingSmall,
                                           (static_cast<float>(button.getHeight()) - tickWidth) * 0.5f,
                                           tickWidth, tickWidth);

  // Checkbox background
  g.setColour(button.getToggleState() ? colour(colours::primary) : colour(colours::surface));
  g.fillRoundedRectangle(tickBounds, metrics::cornerRadiusSmall);

  // Border
  if (!button.getToggleState()) {
    g.setColour(shouldDrawButtonAsHighlighted ? colour(colours::primaryHover)
                                              : colour(colours::surfaceBorder));
    g.drawRoundedRectangle(tickBounds.reduced(0.5f), metrics::cornerRadiusSmall, metrics::borderWidth);
  }

  // Checkmark
  if (button.getToggleState()) {
    juce::Path tick;
    auto cx = tickBounds.getCentreX();
    auto cy = tickBounds.getCentreY();
    auto s = tickWidth * 0.25f;
    tick.startNewSubPath(cx - s, cy);
    tick.lineTo(cx - s * 0.3f, cy + s * 0.7f);
    tick.lineTo(cx + s, cy - s * 0.6f);

    g.setColour(colour(colours::text));
    g.strokePath(tick,
                 juce::PathStrokeType(2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
  }

  // Label text
  g.setColour(button.isEnabled() ? colour(colours::text) : colour(colours::textDisabled));
  g.setFont(typography::body());

  auto textBounds = button.getLocalBounds()
                        .withTrimmedLeft(static_cast<int>(tickWidth + metrics::paddingMedium))
                        .withTrimmedRight(2);

  g.drawFittedText(button.getButtonText(), textBounds, juce::Justification::centredLeft, 2);
}

// ─────────────────────────────────────────────────────────────────
// Fonts
// ─────────────────────────────────────────────────────────────────

juce::Font AutoMixLookAndFeel::getTextButtonFont(juce::TextButton& /*button*/, int /*buttonHeight*/) {
  return typography::body();
}

juce::Font AutoMixLookAndFeel::getComboBoxFont(juce::ComboBox& /*box*/) {
  return typography::body();
}

juce::Font AutoMixLookAndFeel::getLabelFont(juce::Label& /*label*/) {
  return typography::body();
}

// ─────────────────────────────────────────────────────────────────
// Focus Ring
// ─────────────────────────────────────────────────────────────────

void AutoMixLookAndFeel::drawFocusRing(juce::Graphics& g, juce::Component& component) {
  if (component.hasKeyboardFocus(false)) {
    auto w = static_cast<float>(component.getWidth());
    auto h = static_cast<float>(component.getHeight());
    g.setColour(colour(focus::ringColour));
    g.drawRoundedRectangle(0.5f, 0.5f, w - 1.0f, h - 1.0f,
                           metrics::cornerRadius, focus::ringWidth);
  }
}

} // namespace automix::app
