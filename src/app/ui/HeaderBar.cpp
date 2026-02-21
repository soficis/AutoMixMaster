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

  // Tooltips
  saveButton_.setTooltip("Save Session (Ctrl+S)");
  loadButton_.setTooltip("Load Session (Ctrl+O)");
  modelsButton_.setTooltip("Model Manager (Ctrl+K)");
  settingsButton_.setTooltip("Settings");

  // Keyboard focus
  saveButton_.setWantsKeyboardFocus(true);
  loadButton_.setWantsKeyboardFocus(true);
  modelsButton_.setWantsKeyboardFocus(true);
  settingsButton_.setWantsKeyboardFocus(true);

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
}

void HeaderBar::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));

  // Logo text
  g.setColour(colour(colours::primary));
  g.setFont(typography::heading());
  g.drawText("AutoMixMaster", 16, 0, 200, getHeight(), juce::Justification::centredLeft);

  // Bottom border with subtle shadow
  g.setColour(colour(colours::surfaceBorder));
  g.fillRect(0, getHeight() - 1, getWidth(), 1);
  g.setColour(colour(colours::background).withAlpha(0.3f));
  g.fillRect(0, getHeight(), getWidth(), 2);
}

void HeaderBar::resized() {
  auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingMedium));

  // Logo takes left portion
  area.removeFromLeft(200);

  // Buttons on the right (80px each for better hit targets)
  auto rightArea = area.removeFromRight(340);
  settingsButton_.setBounds(rightArea.removeFromRight(80).reduced(2));
  modelsButton_.setBounds(rightArea.removeFromRight(80).reduced(2));
  loadButton_.setBounds(rightArea.removeFromRight(80).reduced(2));
  saveButton_.setBounds(rightArea.removeFromRight(80).reduced(2));

  // Session name in remaining center space
  sessionNameLabel_.setBounds(area);
}

void HeaderBar::setSessionName(const juce::String& name) {
  sessionNameLabel_.setText(name, juce::dontSendNotification);
}

} // namespace automix::app
