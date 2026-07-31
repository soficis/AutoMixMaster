#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"

namespace automix::app {

/// Application header bar: logo, session name, and utility buttons.
class HeaderBar final : public juce::Component {
public:
  HeaderBar();

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setSessionName(const juce::String& name);
  void setProfiles(const std::vector<std::pair<juce::String, juce::String>>& profiles,
                   const juce::String& activeId);

  // Callbacks for button actions
  std::function<void()> onSaveSession;
  std::function<void()> onLoadSession;
  std::function<void()> onSettings;
  std::function<void()> onModels;
  std::function<void(const juce::String& profileId)> onProfileSelected;

private:
  juce::Label sessionNameLabel_;
  juce::TextButton saveButton_{"Save"};
  juce::TextButton loadButton_{"Load"};
  juce::TextButton modelsButton_{"Models"};
  juce::TextButton settingsButton_{"Settings"};
  juce::ComboBox profileSelector_;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HeaderBar)
};

} // namespace automix::app
