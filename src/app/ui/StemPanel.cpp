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

juce::String StemPanel::nameForRole(domain::StemRole role) {
  switch (role) {
    case domain::StemRole::Vocals:
      return "VOX";
    case domain::StemRole::Drums:
      return "DRM";
    case domain::StemRole::Bass:
      return "BAS";
    case domain::StemRole::Guitar:
      return "GTR";
    case domain::StemRole::Keys:
      return "KEY";
    default:
      return "---";
  }
}

// ── StemRow ──────────────────────────────────────────────────────────

StemPanel::StemRow::StemRow(StemPanel& parent) : parent_(parent) {
  roleLabel_.setFont(typography::caption());
  roleLabel_.setJustificationType(juce::Justification::centred);

  nameLabel_.setFont(typography::body());
  nameLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  nameLabel_.setJustificationType(juce::Justification::centredLeft);

  soloButton_.setClickingTogglesState(true);
  soloButton_.setTooltip("Solo");
  soloButton_.setWantsKeyboardFocus(true);
  soloButton_.onClick = [this] {
    stemData.solo = soloButton_.getToggleState();
    if (parent_.onSoloChanged)
      parent_.onSoloChanged(stemData.id, stemData.solo);
    // Sync back to data store
    for (auto& d : parent_.stemData_) {
      if (d.id == stemData.id) {
        d.solo = stemData.solo;
        break;
      }
    }
  };

  muteButton_.setClickingTogglesState(true);
  muteButton_.setTooltip("Mute");
  muteButton_.setWantsKeyboardFocus(true);
  muteButton_.onClick = [this] {
    stemData.mute = muteButton_.getToggleState();
    if (parent_.onMuteChanged)
      parent_.onMuteChanged(stemData.id, stemData.mute);
    for (auto& d : parent_.stemData_) {
      if (d.id == stemData.id) {
        d.mute = stemData.mute;
        break;
      }
    }
  };

  volumeSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  volumeSlider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
  volumeSlider_.setRange(0.0, 1.5, 0.01);
  volumeSlider_.setTooltip("Volume");
  volumeSlider_.onValueChange = [this] {
    stemData.volume = static_cast<float>(volumeSlider_.getValue());
    if (parent_.onVolumeChanged)
      parent_.onVolumeChanged(stemData.id, stemData.volume);
    for (auto& d : parent_.stemData_) {
      if (d.id == stemData.id) {
        d.volume = stemData.volume;
        break;
      }
    }
  };

  addAndMakeVisible(roleLabel_);
  addAndMakeVisible(nameLabel_);
  addAndMakeVisible(soloButton_);
  addAndMakeVisible(muteButton_);
  addAndMakeVisible(volumeSlider_);
}

void StemPanel::StemRow::updateFromDisplay(const StemDisplay& display) {
  stemData = display;
  roleLabel_.setText(nameForRole(display.role), juce::dontSendNotification);
  roleLabel_.setColour(juce::Label::textColourId, colourForRole(display.role));
  nameLabel_.setText(display.name, juce::dontSendNotification);
  soloButton_.setToggleState(display.solo, juce::dontSendNotification);
  muteButton_.setToggleState(display.mute, juce::dontSendNotification);
  volumeSlider_.setValue(static_cast<double>(display.volume), juce::dontSendNotification);
}

void StemPanel::StemRow::paint(juce::Graphics& g) {
  // Role color indicator stripe on the left
  auto roleColour = colourForRole(stemData.role);
  g.setColour(roleColour);
  g.fillRect(0, 2, 4, getHeight() - 4);

  // Hover highlight
  if (isMouseOver()) {
    g.setColour(colour(colours::surfaceLight).withAlpha(0.15f));
    g.fillRect(getLocalBounds());
  }

  // Bottom separator
  g.setColour(colour(colours::surfaceBorder).withAlpha(0.3f));
  g.fillRect(0, getHeight() - 1, getWidth(), 1);
}

void StemPanel::StemRow::resized() {
  auto area = getLocalBounds().reduced(2);
  area.removeFromLeft(6); // space after role indicator

  // Role text label (32px)
  roleLabel_.setBounds(area.removeFromLeft(32));
  area.removeFromLeft(2);

  nameLabel_.setBounds(area.removeFromLeft(std::min(120, area.getWidth() / 3)));

  // Larger solo/mute buttons (36px each for better hit targets)
  soloButton_.setBounds(area.removeFromLeft(36).reduced(2));
  muteButton_.setBounds(area.removeFromLeft(36).reduced(2));

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

  if (stemData_.empty()) {
    g.setColour(colour(colours::textMuted));
    g.setFont(typography::body());
    g.drawText("No stems imported", getLocalBounds(), juce::Justification::centred);
  }
}

void StemPanel::resized() {
  viewport_.setBounds(getLocalBounds());

  int totalHeight = static_cast<int>(stemData_.size()) * kRowHeight;
  rowContainer_.setBounds(0, 0, viewport_.getMaximumVisibleWidth(), std::max(totalHeight, getHeight()));

  rebuildVisibleRows();
}

void StemPanel::setStems(const std::vector<StemDisplay>& stems) {
  stemData_ = stems;

  // Clear existing rows
  rowContainer_.removeAllChildren();
  visibleRows_.clear();
  firstVisibleIndex_ = 0;
  lastVisibleIndex_ = -1;

  // Resize container to fit all rows
  int totalHeight = static_cast<int>(stemData_.size()) * kRowHeight;
  rowContainer_.setBounds(0, 0, viewport_.getMaximumVisibleWidth(), std::max(totalHeight, getHeight()));

  rebuildVisibleRows();
  repaint();
}

std::vector<StemPanel::StemDisplay> StemPanel::getStemDisplays() const {
  return stemData_;
}

void StemPanel::rebuildVisibleRows() {
  if (stemData_.empty()) {
    rowContainer_.removeAllChildren();
    visibleRows_.clear();
    firstVisibleIndex_ = 0;
    lastVisibleIndex_ = -1;
    return;
  }

  auto visibleArea = viewport_.getViewArea();
  int totalStems = static_cast<int>(stemData_.size());

  // Calculate which rows should be visible (with buffer)
  int newFirst = std::max(0, visibleArea.getY() / kRowHeight - kBufferRows);
  int newLast = std::min(totalStems - 1, (visibleArea.getBottom() + kRowHeight - 1) / kRowHeight + kBufferRows);

  if (newFirst == firstVisibleIndex_ && newLast == lastVisibleIndex_
      && static_cast<int>(visibleRows_.size()) == (newLast - newFirst + 1))
    return;

  // Rebuild visible row components
  rowContainer_.removeAllChildren();
  visibleRows_.clear();

  int containerWidth = rowContainer_.getWidth();
  for (int i = newFirst; i <= newLast; ++i) {
    auto row = std::make_unique<StemRow>(*this);
    row->updateFromDisplay(stemData_[static_cast<size_t>(i)]);
    row->setBounds(0, i * kRowHeight, containerWidth, kRowHeight);
    rowContainer_.addAndMakeVisible(*row);
    visibleRows_.push_back(std::move(row));
  }

  firstVisibleIndex_ = newFirst;
  lastVisibleIndex_ = newLast;
}

} // namespace automix::app
