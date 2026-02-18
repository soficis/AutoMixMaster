#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace automix::app::theme {

// ── Color Palette (Dark Audio Theme) ──────────────────────────────

namespace colours {

// Backgrounds
inline constexpr juce::uint32 background = 0xFF1A1A2E;    // Deep navy
inline constexpr juce::uint32 surface = 0xFF252540;       // Raised panels
inline constexpr juce::uint32 surfaceLight = 0xFF2F2F50;  // Hovered surfaces
inline constexpr juce::uint32 surfaceBorder = 0xFF3A3A5C; // Subtle borders

// Brand / Accent
inline constexpr juce::uint32 primary = 0xFF4361EE;        // Action blue
inline constexpr juce::uint32 primaryHover = 0xFF5A7AFF;   // Blue hover
inline constexpr juce::uint32 primaryPressed = 0xFF3451D1; // Blue active
inline constexpr juce::uint32 secondary = 0xFF2EC4B6;      // Teal accent
inline constexpr juce::uint32 accent = 0xFFFF6B35;         // Warm accent (warnings, highlights)

// Text
inline constexpr juce::uint32 text = 0xFFEAEAEA;         // Primary text
inline constexpr juce::uint32 textMuted = 0xFF9898B0;    // Secondary text
inline constexpr juce::uint32 textDisabled = 0xFF5A5A72; // Disabled text

// Semantic
inline constexpr juce::uint32 error = 0xFFE63946;   // Red
inline constexpr juce::uint32 success = 0xFF06D6A0; // Green
inline constexpr juce::uint32 warning = 0xFFFFD166; // Yellow

// Meters / Levels
inline constexpr juce::uint32 meterLow = 0xFF06D6A0;  // Green
inline constexpr juce::uint32 meterMid = 0xFFFFD166;  // Yellow
inline constexpr juce::uint32 meterHigh = 0xFFE63946; // Red
inline constexpr juce::uint32 meterBackground = 0xFF1A1A2E;

// Transport / Waveform
inline constexpr juce::uint32 playhead = 0xFFFFFFFF;
inline constexpr juce::uint32 waveformFill = 0xFF4361EE;
inline constexpr juce::uint32 waveformOutline = 0xFF5A7AFF;
inline constexpr juce::uint32 selectionFill = 0x404361EE; // Semi-transparent blue

} // namespace colours

// ── Metrics ───────────────────────────────────────────────────────

namespace metrics {

inline constexpr float cornerRadius = 6.0f;
inline constexpr float cornerRadiusSmall = 4.0f;
inline constexpr float cornerRadiusLarge = 10.0f;

inline constexpr float paddingSmall = 4.0f;
inline constexpr float paddingMedium = 8.0f;
inline constexpr float paddingLarge = 16.0f;

inline constexpr float borderWidth = 1.0f;
inline constexpr float borderWidthFocused = 2.0f;

inline constexpr float buttonHeight = 32.0f;
inline constexpr float sliderThumbRadius = 7.0f;
inline constexpr float sliderTrackHeight = 4.0f;

} // namespace metrics

// ── Typography ────────────────────────────────────────────────────

namespace typography {

inline constexpr float headingSize = 20.0f;
inline constexpr float subheadSize = 16.0f;
inline constexpr float bodySize = 14.0f;
inline constexpr float captionSize = 11.0f;
inline constexpr float smallSize = 10.0f;

inline juce::Font heading() {
  return juce::Font(juce::FontOptions{}.withPointHeight(headingSize));
}

inline juce::Font subhead() {
  return juce::Font(juce::FontOptions{}.withPointHeight(subheadSize));
}

inline juce::Font body() {
  return juce::Font(juce::FontOptions{}.withPointHeight(bodySize));
}

inline juce::Font caption() {
  return juce::Font(juce::FontOptions{}.withPointHeight(captionSize));
}

} // namespace typography

// ── Spacing ───────────────────────────────────────────────────────

namespace spacing {

inline constexpr int gapSmall = 4;
inline constexpr int gapMedium = 8;
inline constexpr int gapLarge = 16;
inline constexpr int gapXLarge = 24;

inline constexpr int marginSmall = 4;
inline constexpr int marginMedium = 8;
inline constexpr int marginLarge = 16;

} // namespace spacing

// ── Helpers ───────────────────────────────────────────────────────

inline juce::Colour colour(juce::uint32 argb) {
  return juce::Colour(argb);
}

} // namespace automix::app::theme
