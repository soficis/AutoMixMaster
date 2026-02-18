#pragma once

#include <string>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"
#include "domain/Stem.h"
#include "domain/StemRole.h"

namespace automix::app {

/// Panel displaying imported stems with per-stem solo/mute/volume controls.
class StemPanel final : public juce::Component {
public:
  struct StemDisplay {
    std::string id;
    std::string name;
    domain::StemRole role = domain::StemRole::Unknown;
    bool solo = false;
    bool mute = false;
    float volume = 1.0f;
    bool enabled = true;
  };

  StemPanel();

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setStems(const std::vector<StemDisplay>& stems);

  // Callbacks
  std::function<void(const std::string& stemId, bool solo)> onSoloChanged;
  std::function<void(const std::string& stemId, bool mute)> onMuteChanged;
  std::function<void(const std::string& stemId, float volume)> onVolumeChanged;

private:
  class StemRow final : public juce::Component {
  public:
    StemRow(const StemDisplay& stem, StemPanel& parent);
    void paint(juce::Graphics& g) override;
    void resized() override;

    StemDisplay stemData;

  private:
    StemPanel& parent_;
    juce::Label nameLabel_;
    juce::TextButton soloButton_{"S"};
    juce::TextButton muteButton_{"M"};
    juce::Slider volumeSlider_;
  };

  static juce::Colour colourForRole(domain::StemRole role);

  std::vector<std::unique_ptr<StemRow>> rows_;
  juce::Viewport viewport_;
  juce::Component rowContainer_;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StemPanel)
};

} // namespace automix::app
