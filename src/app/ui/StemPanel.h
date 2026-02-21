#pragma once

#include <string>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"
#include "domain/Stem.h"
#include "domain/StemRole.h"

namespace automix::app {

/// Panel displaying imported stems with per-stem solo/mute/volume controls.
/// Uses lazy rendering: only creates StemRow components for visible rows.
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
  ~StemPanel() override = default;

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setStems(const std::vector<StemDisplay>& stems);
  std::vector<StemDisplay> getStemDisplays() const;

  // Callbacks
  std::function<void(const std::string& stemId, bool solo)> onSoloChanged;
  std::function<void(const std::string& stemId, bool mute)> onMuteChanged;
  std::function<void(const std::string& stemId, float volume)> onVolumeChanged;

private:
  class StemRow final : public juce::Component {
  public:
    StemRow(StemPanel& parent);
    void paint(juce::Graphics& g) override;
    void resized() override;
    void updateFromDisplay(const StemDisplay& display);

    StemDisplay stemData;

  private:
    StemPanel& parent_;
    juce::Label roleLabel_;
    juce::Label nameLabel_;
    juce::TextButton soloButton_{"S"};
    juce::TextButton muteButton_{"M"};
    juce::Slider volumeSlider_;
  };

  // Custom viewport that notifies parent when scroll area changes
  class LazyViewport final : public juce::Viewport {
  public:
    explicit LazyViewport(StemPanel& owner) : owner_(owner) {}
    void visibleAreaChanged(const juce::Rectangle<int>& /*area*/) override {
      owner_.rebuildVisibleRows();
    }
  private:
    StemPanel& owner_;
  };

  void rebuildVisibleRows();
  static juce::Colour colourForRole(domain::StemRole role);
  static juce::String nameForRole(domain::StemRole role);

  static constexpr int kRowHeight = 40;
  static constexpr int kBufferRows = 2; // extra rows above/below visible area

  std::vector<StemDisplay> stemData_;
  std::vector<std::unique_ptr<StemRow>> visibleRows_;
  int firstVisibleIndex_ = 0;
  int lastVisibleIndex_ = -1;
  LazyViewport viewport_{*this};
  juce::Component rowContainer_;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StemPanel)
};

} // namespace automix::app
