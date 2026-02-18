#include "app/ui/StemPanel.h"

#include <algorithm>

namespace automix::app {

using namespace theme;

juce::Colour StemPanel::colourForRole(domain::StemRole role) {
  switch (role) {
    case domain::StemRole::Vocals:
      return colour(colours::primary);
    case domain::StemRole::Drums:
      return colour(colours::accent);
    case domain::StemRole::Bass:
      return colour(colours::secondary);
    case domain::StemRole::Guitar:
      return colour(colours::success);
    case domain::StemRole::Keys:
      return colour(colours::warning);
    default:
      return colour(colours::textMuted);
  }
}

// ── StemRow ──────────────────────────────────────────────────────────

StemPanel::StemRow::StemRow(const StemDisplay& stem, StemPanel& parent) : stemData(stem), parent_(parent) {
  nameLabel_.setText(stem.name, juce::dontSendNotification);
  nameLabel_.setFont(typography::body());
  nameLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  nameLabel_.setJustificationType(juce::Justification::centredLeft);

  soloButton_.setClickingTogglesState(true);
  soloButton_.setToggleState(stem.solo, juce::dontSendNotification);
  soloButton_.onClick = [this] {
    stemData.solo = soloButton_.getToggleState();
    if (parent_.onSoloChanged)
      parent_.onSoloChanged(stemData.id, stemData.solo);
  };

  muteButton_.setClickingTogglesState(true);
  muteButton_.setToggleState(stem.mute, juce::dontSendNotification);
  muteButton_.onClick = [this] {
    stemData.mute = muteButton_.getToggleState();
    if (parent_.onMuteChanged)
      parent_.onMuteChanged(stemData.id, stemData.mute);
  };

  volumeSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  volumeSlider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
  volumeSlider_.setRange(0.0, 1.5, 0.01);
  volumeSlider_.setValue(static_cast<double>(stem.volume), juce::dontSendNotification);
  volumeSlider_.onValueChange = [this] {
    stemData.volume = static_cast<float>(volumeSlider_.getValue());
    if (parent_.onVolumeChanged)
      parent_.onVolumeChanged(stemData.id, stemData.volume);
  };

  addAndMakeVisible(nameLabel_);
  addAndMakeVisible(soloButton_);
  addAndMakeVisible(muteButton_);
  addAndMakeVisible(volumeSlider_);
}

void StemPanel::StemRow::paint(juce::Graphics& g) {
  // Role color indicator stripe on the left
  auto roleColour = colourForRole(stemData.role);
  g.setColour(roleColour);
  g.fillRect(0, 2, 4, getHeight() - 4);

  // Bottom separator
  g.setColour(colour(colours::surfaceBorder).withAlpha(0.3f));
  g.fillRect(0, getHeight() - 1, getWidth(), 1);
}

void StemPanel::StemRow::resized() {
  auto area = getLocalBounds().reduced(2);
  area.removeFromLeft(8); // space after role indicator

  nameLabel_.setBounds(area.removeFromLeft(std::min(140, area.getWidth() / 3)));
  soloButton_.setBounds(area.removeFromLeft(32).reduced(2));
  muteButton_.setBounds(area.removeFromLeft(32).reduced(2));
  volumeSlider_.setBounds(area.reduced(4, 2));
}

// ── StemPanel ────────────────────────────────────────────────────────

StemPanel::StemPanel() {
  viewport_.setViewedComponent(&rowContainer_, false);
  viewport_.setScrollBarsShown(true, false);
  addAndMakeVisible(viewport_);
}

void StemPanel::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));

  if (rows_.empty()) {
    g.setColour(colour(colours::textMuted));
    g.setFont(typography::body());
    g.drawText("No stems imported", getLocalBounds(), juce::Justification::centred);
  }
}

void StemPanel::resized() {
  viewport_.setBounds(getLocalBounds());

  constexpr int rowHeight = 36;
  int totalHeight = static_cast<int>(rows_.size()) * rowHeight;
  rowContainer_.setBounds(0, 0, viewport_.getMaximumVisibleWidth(), std::max(totalHeight, getHeight()));

  for (int i = 0; i < static_cast<int>(rows_.size()); ++i) {
    rows_[static_cast<size_t>(i)]->setBounds(0, i * rowHeight, rowContainer_.getWidth(), rowHeight);
  }
}

void StemPanel::setStems(const std::vector<StemDisplay>& stems) {
  rowContainer_.removeAllChildren();
  rows_.clear();

  for (const auto& stem : stems) {
    auto row = std::make_unique<StemRow>(stem, *this);
    rowContainer_.addAndMakeVisible(*row);
    rows_.push_back(std::move(row));
  }

  resized();
  repaint();
}

} // namespace automix::app
