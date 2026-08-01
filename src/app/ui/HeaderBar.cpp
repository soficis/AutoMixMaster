#include "app/ui/HeaderBar.h"

namespace automix::app {

using namespace theme;

HeaderBar::HeaderBar() {
  sessionNameLabel_.setText("Untitled Session", juce::dontSendNotification);
  sessionNameLabel_.setFont(typography::subhead());
  sessionNameLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  sessionNameLabel_.setJustificationType(juce::Justification::centred);

  addAndMakeVisible(sessionNameLabel_);
  addAndMakeVisible(saveButton_);
  addAndMakeVisible(loadButton_);
  addAndMakeVisible(modelsButton_);
  addAndMakeVisible(settingsButton_);
  addAndMakeVisible(profileSelector_);

  saveButton_.setTooltip("Save Session (Ctrl+S)");
  loadButton_.setTooltip("Load Session (Ctrl+O)");
  modelsButton_.setTooltip("Model Manager (Ctrl+K)");
  settingsButton_.setTooltip("Settings");
  profileSelector_.setTooltip("Switch Project Profile");

  saveButton_.setWantsKeyboardFocus(true);
  loadButton_.setWantsKeyboardFocus(true);
  modelsButton_.setWantsKeyboardFocus(true);
  settingsButton_.setWantsKeyboardFocus(true);

  profileSelector_.setTextWhenNoChoicesAvailable("No profiles");

  saveButton_.onClick = [this] {
    if (onSaveSession)
      onSaveSession();
  };
  loadButton_.onClick = [this] {
    if (onLoadSession)
      onLoadSession();
  };
  modelsButton_.onClick = [this] {
    if (onModels)
      onModels();
  };
  settingsButton_.onClick = [this] {
    if (onSettings)
      onSettings();
  };
  profileSelector_.onChange = [this] {
    const auto id = profileSelector_.getSelectedId();
    if (id > 0 && id <= static_cast<int>(profileIds_.size()) && onProfileSelected) {
      onProfileSelected(profileIds_[id - 1]);
    }
  };
}

void HeaderBar::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));

  g.setColour(colour(colours::primary));
  g.setFont(typography::heading());
  g.drawText("AutoMixMaster", 16, 0, 200, getHeight(), juce::Justification::centredLeft);

  g.setColour(colour(colours::surfaceBorder));
  g.fillRect(0, getHeight() - 1, getWidth(), 1);
  g.setColour(colour(colours::background).withAlpha(0.3f));
  g.fillRect(0, getHeight(), getWidth(), 2);
}

void HeaderBar::resized() {
  auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingMedium));

  area.removeFromLeft(200);

  auto rightArea = area.removeFromRight(320);
  settingsButton_.setBounds(rightArea.removeFromRight(80).reduced(2));
  modelsButton_.setBounds(rightArea.removeFromRight(80).reduced(2));
  loadButton_.setBounds(rightArea.removeFromRight(80).reduced(2));
  saveButton_.setBounds(rightArea.reduced(2));

  auto profileArea = area.removeFromLeft(220);
  profileSelector_.setBounds(profileArea.reduced(2));

  sessionNameLabel_.setBounds(area);
}

void HeaderBar::setSessionName(const juce::String& name) {
  sessionNameLabel_.setText(name, juce::dontSendNotification);
}

void HeaderBar::setProfiles(const std::vector<std::pair<juce::String, juce::String>>& profiles,
                            const juce::String& activeId) {
  profileSelector_.clear(juce::dontSendNotification);
  profileIds_.clear();
  int comboId = 1;
  int selectedId = 0;
  for (const auto& [name, id] : profiles) {
    profileSelector_.addItem(name, comboId);
    profileIds_.push_back(id);
    if (id == activeId)
      selectedId = comboId;
    ++comboId;
  }
  if (selectedId > 0)
    profileSelector_.setSelectedId(selectedId, juce::dontSendNotification);
}

} // namespace automix::app
