#include "app/ui/ProfileQuickSwitch.h"

namespace automix::app::ui {

// ─────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────

ProfileQuickSwitch::ProfileQuickSwitch(OnProfileChanged onProfileChanged)
    : onProfileChanged_(std::move(onProfileChanged)) {
  setTextWhenNoChoicesAvailable("(No Profiles)");
  setTextWhenNothingSelected("Select Profile");
  setTooltip("Switch Profile");
  onChange = [this] { onSelectionChanged(); };
}

// ─────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────

void ProfileQuickSwitch::setProfiles(const juce::StringArray& profiles) {
  profileNames_ = profiles;
  clear(juce::dontSendNotification);

  for (int i = 0; i < profiles.size(); ++i)
    addItem(profiles[i], i + 1); // 1-based IDs for ComboBox

  // Restore active selection if it still exists
  const int activeIndex = profileNames_.indexOf(activeProfile_);
  if (activeIndex >= 0)
    setSelectedId(activeIndex + 1, juce::dontSendNotification);
  else if (profiles.size() > 0)
    setSelectedId(1, juce::dontSendNotification);
  else
    activeProfile_.clear();
}

void ProfileQuickSwitch::setActiveProfile(const juce::String& profile) {
  activeProfile_ = profile;
  const int index = profileNames_.indexOf(profile);
  if (index >= 0)
    setSelectedId(index + 1, juce::dontSendNotification);
}

juce::String ProfileQuickSwitch::getActiveProfile() const {
  return activeProfile_;
}

// ─────────────────────────────────────────────────────────────────
// Private
// ─────────────────────────────────────────────────────────────────

void ProfileQuickSwitch::onSelectionChanged() {
  const int id = getSelectedId();
  if (id <= 0 || id > profileNames_.size())
    return;

  const juce::String selected = profileNames_[id - 1];
  if (selected == activeProfile_)
    return; // no actual change

  activeProfile_ = selected;

  if (onProfileChanged_)
    onProfileChanged_(activeProfile_);
}

} // namespace automix::app::ui
