#include "app/ui/ControlDeck.h"

#include "app/ui/GlowMeters.h"
#include "app/ui/StemPanel.h"

namespace automix::app {

using namespace theme;

ControlDeck::ControlDeck() {
  stemPanel_ = std::make_unique<StemPanel>();
  glowMeters_ = std::make_unique<GlowMeters>();

  addAndMakeVisible(*stemPanel_);
  addAndMakeVisible(*glowMeters_);

  // Action buttons
  addAndMakeVisible(importButton_);
  addAndMakeVisible(autoMixButton_);
  addAndMakeVisible(autoMasterButton_);
  addAndMakeVisible(autoMixMasterButton_);
  addAndMakeVisible(batchButton_);
  addAndMakeVisible(exportButton_);

  // Tooltips
  importButton_.setTooltip("Import Stems (Ctrl+I)");
  autoMixButton_.setTooltip("Auto Mix (Ctrl+M)");
  autoMasterButton_.setTooltip("Auto Master");
  autoMixMasterButton_.setTooltip("One-click: Auto Mix \xe2\x86\x92 Auto Master \xe2\x86\x92 Export (Ctrl+Shift+M)");
  batchButton_.setTooltip("Batch Process");
  exportButton_.setTooltip("Export (Ctrl+E)");
  separatedStemsToggle_.setTooltip("Use AI stem separation during import");
  batchRecursiveToggle_.setTooltip("Include subfolders when scanning batch input");
  residualBlendSlider_.setTooltip("Control residual audio blend");

  // Keyboard focus on action buttons
  importButton_.setWantsKeyboardFocus(true);
  autoMixButton_.setWantsKeyboardFocus(true);
  autoMasterButton_.setWantsKeyboardFocus(true);
  autoMixMasterButton_.setWantsKeyboardFocus(true);
  batchButton_.setWantsKeyboardFocus(true);
  exportButton_.setWantsKeyboardFocus(true);

  importButton_.onClick = [this] {
    if (onImport)
      onImport();
  };
  autoMixButton_.onClick = [this] {
    if (onAutoMix)
      onAutoMix();
  };
  autoMasterButton_.onClick = [this] {
    if (onAutoMaster)
      onAutoMaster();
  };
  autoMixMasterButton_.onClick = [this] {
    if (onAutoMixMaster)
      onAutoMixMaster();
  };
  batchButton_.onClick = [this] {
    if (onBatch)
      onBatch();
  };
  exportButton_.onClick = [this] {
    if (onExport)
      onExport();
  };

  // Settings labels
  rendererLabel_.setFont(typography::caption());
  rendererLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  rendererLabel_.setJustificationType(juce::Justification::centredLeft);
  profileLabel_.setFont(typography::caption());
  profileLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  profileLabel_.setJustificationType(juce::Justification::centredLeft);
  masterPresetLabel_.setFont(typography::caption());
  masterPresetLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  masterPresetLabel_.setJustificationType(juce::Justification::centredLeft);
  platformPresetLabel_.setFont(typography::caption());
  platformPresetLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  platformPresetLabel_.setJustificationType(juce::Justification::centredLeft);
  exportFormatLabel_.setFont(typography::caption());
  exportFormatLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  exportFormatLabel_.setJustificationType(juce::Justification::centredLeft);
  exportModeLabel_.setFont(typography::caption());
  exportModeLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  exportModeLabel_.setJustificationType(juce::Justification::centredLeft);
  blendLabel_.setFont(typography::caption());
  blendLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  blendLabel_.setJustificationType(juce::Justification::centredLeft);

  residualBlendSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  residualBlendSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 20);
  residualBlendSlider_.setRange(0.0, 10.0, 0.1);
  residualBlendSlider_.setValue(0.0, juce::dontSendNotification);

  addAndMakeVisible(rendererLabel_);
  addAndMakeVisible(rendererBox_);
  addAndMakeVisible(profileLabel_);
  addAndMakeVisible(profileBox_);
  addAndMakeVisible(masterPresetLabel_);
  addAndMakeVisible(masterPresetBox_);
  addAndMakeVisible(platformPresetLabel_);
  addAndMakeVisible(platformPresetBox_);
  addAndMakeVisible(exportFormatLabel_);
  addAndMakeVisible(exportFormatBox_);
  addAndMakeVisible(exportModeLabel_);
  addAndMakeVisible(exportModeBox_);
  addAndMakeVisible(blendLabel_);
  addAndMakeVisible(residualBlendSlider_);
  addAndMakeVisible(separatedStemsToggle_);
  addAndMakeVisible(batchRecursiveToggle_);
}

ControlDeck::~ControlDeck() = default;

void ControlDeck::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::background));

  // Top border
  g.setColour(colour(colours::surfaceBorder));
  g.fillRect(0, 0, getWidth(), 1);
}

void ControlDeck::resized() {
  auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingMedium));

  // Three-column layout using FlexBox
  int stemWidth = std::max(200, area.getWidth() * 35 / 100);
  int meterWidth = std::max(100, area.getWidth() * 15 / 100);

  auto stemArea = area.removeFromLeft(stemWidth);
  area.removeFromLeft(spacing::gapMedium);
  auto meterArea = area.removeFromRight(meterWidth);
  area.removeFromRight(spacing::gapMedium);
  auto centerArea = area;

  stemPanel_->setBounds(stemArea);
  glowMeters_->setBounds(meterArea);

  // Center panel layout (44px action row, proportional widths)
  // Slots: import(1) | autoMix(1) | autoMaster(1) | Mix+Master(2) | batch(1) | export(1) = 7 slots
  auto actionRow = centerArea.removeFromTop(44);
  const int slotW = actionRow.getWidth() / 7;
  importButton_.setBounds(actionRow.removeFromLeft(slotW).reduced(2));
  autoMixButton_.setBounds(actionRow.removeFromLeft(slotW).reduced(2));
  autoMasterButton_.setBounds(actionRow.removeFromLeft(slotW).reduced(2));
  autoMixMasterButton_.setBounds(actionRow.removeFromLeft(slotW * 2).reduced(2));
  batchButton_.setBounds(actionRow.removeFromLeft(slotW).reduced(2));
  exportButton_.setBounds(actionRow.reduced(2));

  centerArea.removeFromTop(spacing::gapSmall);
  auto toggleRow = centerArea.removeFromTop(24);
  separatedStemsToggle_.setBounds(toggleRow.removeFromLeft(140));
  toggleRow.removeFromLeft(spacing::gapSmall);
  batchRecursiveToggle_.setBounds(toggleRow.removeFromLeft(160));
  centerArea.removeFromTop(spacing::gapSmall);

  // Settings rows
  auto settingsRow1 = centerArea.removeFromTop(28);
  rendererLabel_.setBounds(settingsRow1.removeFromLeft(64).reduced(1));
  rendererBox_.setBounds(settingsRow1.removeFromLeft(160).reduced(1));
  profileLabel_.setBounds(settingsRow1.removeFromLeft(50).reduced(1));
  profileBox_.setBounds(settingsRow1.removeFromLeft(140).reduced(1));

  auto settingsRow2 = centerArea.removeFromTop(28);
  masterPresetLabel_.setBounds(settingsRow2.removeFromLeft(50).reduced(1));
  masterPresetBox_.setBounds(settingsRow2.removeFromLeft(140).reduced(1));
  platformPresetLabel_.setBounds(settingsRow2.removeFromLeft(64).reduced(1));
  platformPresetBox_.setBounds(settingsRow2.removeFromLeft(140).reduced(1));

  auto settingsRow3 = centerArea.removeFromTop(28);
  exportFormatLabel_.setBounds(settingsRow3.removeFromLeft(50).reduced(1));
  exportFormatBox_.setBounds(settingsRow3.removeFromLeft(100).reduced(1));
  exportModeLabel_.setBounds(settingsRow3.removeFromLeft(44).reduced(1));
  exportModeBox_.setBounds(settingsRow3.removeFromLeft(100).reduced(1));

  auto settingsRow4 = centerArea.removeFromTop(28);
  blendLabel_.setBounds(settingsRow4.removeFromLeft(96).reduced(1));
  residualBlendSlider_.setBounds(settingsRow4.removeFromLeft(200).reduced(1));
}

} // namespace automix::app
