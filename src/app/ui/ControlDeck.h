#pragma once

#include <memory>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"

namespace automix::app {

class StemPanel;
class GlowMeters;

/// Container for the main control area: StemPanel (left), controls (center), GlowMeters (right).
class ControlDeck final : public juce::Component {
public:
  ControlDeck();
  ~ControlDeck() override;

  void paint(juce::Graphics& g) override;
  void resized() override;

  StemPanel& getStemPanel() { return *stemPanel_; }
  GlowMeters& getGlowMeters() { return *glowMeters_; }

  // Center panel action callbacks
  std::function<void()> onImport;
  std::function<void()> onAutoMix;
  std::function<void()> onAutoMaster;
  std::function<void()> onAutoMixMaster; // One-click pipeline: Mix → Master → Export
  std::function<void()> onBatch;
  std::function<void()> onExport;

  // Settings widgets access for F4 wiring
  juce::ComboBox& getRendererBox() { return rendererBox_; }
  juce::ComboBox& getProfileBox() { return profileBox_; }
  juce::ComboBox& getMasterPresetBox() { return masterPresetBox_; }
  juce::ComboBox& getPlatformPresetBox() { return platformPresetBox_; }
  juce::ComboBox& getExportFormatBox() { return exportFormatBox_; }
  juce::ComboBox& getExportModeBox() { return exportModeBox_; }
  juce::Slider& getResidualBlendSlider() { return residualBlendSlider_; }
  juce::ToggleButton& getSeparatedStemsToggle() { return separatedStemsToggle_; }
  juce::ToggleButton& getBatchRecursiveToggle() { return batchRecursiveToggle_; }

private:
  std::unique_ptr<StemPanel> stemPanel_;
  std::unique_ptr<GlowMeters> glowMeters_;

  // Center panel — action buttons
  juce::TextButton importButton_{"Import"};
  juce::TextButton autoMixButton_{"Auto Mix"};
  juce::TextButton autoMasterButton_{"Auto Master"};
  juce::TextButton autoMixMasterButton_{"Mix + Master"};
  juce::TextButton batchButton_{"Batch"};
  juce::TextButton exportButton_{"Export"};

  // Center panel — settings
  juce::Label rendererLabel_{"", "Renderer"};
  juce::ComboBox rendererBox_;
  juce::Label profileLabel_{"", "Profile"};
  juce::ComboBox profileBox_;
  juce::Label masterPresetLabel_{"", "Master"};
  juce::ComboBox masterPresetBox_;
  juce::Label platformPresetLabel_{"", "Platform"};
  juce::ComboBox platformPresetBox_;
  juce::Label exportFormatLabel_{"", "Format"};
  juce::ComboBox exportFormatBox_;
  juce::Label exportModeLabel_{"", "Mode"};
  juce::ComboBox exportModeBox_;
  juce::Label blendLabel_{"", "Residual Blend"};
  juce::Slider residualBlendSlider_;
  juce::ToggleButton separatedStemsToggle_{"AI Stems"};
  juce::ToggleButton batchRecursiveToggle_{"Recursive Batch"};

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ControlDeck)
};

} // namespace automix::app
