#pragma once

#include <functional>

#include <juce_gui_basics/juce_gui_basics.h>

namespace automix::app::ui {

/// Header-bar profile dropdown that fires a callback when the selection changes.
///
/// Wraps a juce::ComboBox with typed profile-name accessors.  The combo
/// is populated with setProfiles() and the active entry is tracked by
/// display name (not by internal combo-box ID).
class ProfileQuickSwitch final : public juce::ComboBox {
public:
  using OnProfileChanged = std::function<void(const juce::String& profileName)>;

  /// @param onProfileChanged  Called when the user selects a different profile.
  explicit ProfileQuickSwitch(OnProfileChanged onProfileChanged = nullptr);

  /// Replace the list of available profiles.
  void setProfiles(const juce::StringArray& profiles);

  /// Mark @p profile as the active one (must be one of the previously set profiles).
  void setActiveProfile(const juce::String& profile);

  /// Return the display name of the currently selected profile.
  [[nodiscard]] juce::String getActiveProfile() const;

private:
  OnProfileChanged onProfileChanged_;

  juce::StringArray profileNames_;
  juce::String activeProfile_;

  void onSelectionChanged();

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ProfileQuickSwitch)
};

} // namespace automix::app::ui
