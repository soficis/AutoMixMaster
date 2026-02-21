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
  autoMixMasterButton_.setTooltip("One-click: Auto Mix -> Auto Master -> Export (Ctrl+Shift+M)");
  batchButton_.setTooltip("Batch Process");
  exportButton_.setTooltip("Export (Ctrl+E)");
  separatedStemsToggle_.setTooltip("Split a single imported full mix into stems using the active Separation model pack.");
  batchRecursiveToggle_.setTooltip("Include subfolders when scanning batch input");
  rendererChainToggle_.setTooltip("Run renderers in a staged chain");
  rendererChainModeBox_.setTooltip("Renderer chain strategy");
  residualBlendSlider_.setTooltip("Control residual audio blend");

  // Button visual hierarchy: primary > secondary > tertiary
  // Primary — Mix+Master is the hero action; filled brand colour.
  autoMixMasterButton_.setColour(juce::TextButton::buttonColourId, colour(colours::primary));
  autoMixMasterButton_.setColour(juce::TextButton::buttonOnColourId, colour(colours::primaryPressed));
  autoMixMasterButton_.setColour(juce::TextButton::textColourOffId, juce::Colours::white);
  autoMixMasterButton_.setColour(juce::TextButton::textColourOnId, juce::Colours::white);

  // Secondary — Import and Export are the gateway/output actions; tinted text on dark surface.
  for (juce::TextButton* btn : {&importButton_, &exportButton_}) {
    btn->setColour(juce::TextButton::buttonColourId, colour(colours::surface));
    btn->setColour(juce::TextButton::buttonOnColourId, colour(colours::surfaceLight));
    btn->setColour(juce::TextButton::textColourOffId, colour(colours::primary));
    btn->setColour(juce::TextButton::textColourOnId, colour(colours::primaryHover));
  }

  // Tertiary — AutoMix, AutoMaster, Batch are advanced ops; visually receded.
  for (juce::TextButton* btn : {&autoMixButton_, &autoMasterButton_, &batchButton_}) {
    btn->setColour(juce::TextButton::buttonColourId, colour(colours::surface));
    btn->setColour(juce::TextButton::buttonOnColourId, colour(colours::surfaceLight));
    btn->setColour(juce::TextButton::textColourOffId, colour(colours::textMuted));
    btn->setColour(juce::TextButton::textColourOnId, colour(colours::text));
  }

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

  // Settings labels — always-visible context row
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

  // Advanced settings labels — hidden until disclosed (except active chain preview)
  exportFormatLabel_.setFont(typography::caption());
  exportFormatLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  exportFormatLabel_.setJustificationType(juce::Justification::centredLeft);
  exportModeLabel_.setFont(typography::caption());
  exportModeLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  exportModeLabel_.setJustificationType(juce::Justification::centredLeft);
  rendererChainModeLabel_.setFont(typography::caption());
  rendererChainModeLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  rendererChainModeLabel_.setJustificationType(juce::Justification::centredLeft);
  rendererChainPreviewLabel_.setFont(typography::caption());
  rendererChainPreviewLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  rendererChainPreviewLabel_.setJustificationType(juce::Justification::centredLeft);
  addAndMakeVisible(rendererChainPreviewLabel_);
  blendLabel_.setFont(typography::caption());
  blendLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  blendLabel_.setJustificationType(juce::Justification::centredLeft);
  separationModelStatusLabel_.setFont(typography::caption());
  separationModelStatusLabel_.setColour(juce::Label::textColourId, colour(colours::warning));
  separationModelStatusLabel_.setJustificationType(juce::Justification::centredLeft);
  separationModelStatusLabel_.setText("Separation model: none", juce::dontSendNotification);

  residualBlendSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  residualBlendSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 20);
  residualBlendSlider_.setRange(0.0, 10.0, 0.1);
  residualBlendSlider_.setValue(0.0, juce::dontSendNotification);
  rendererChainModeBox_.setEnabled(false);

  // Advanced disclosure toggle
  advancedToggle_.setColour(juce::TextButton::buttonColourId, colour(colours::surface));
  advancedToggle_.setColour(juce::TextButton::buttonOnColourId, colour(colours::surfaceLight));
  advancedToggle_.setColour(juce::TextButton::textColourOffId, colour(colours::textMuted));
  advancedToggle_.setColour(juce::TextButton::textColourOnId, colour(colours::text));
  advancedToggle_.onClick = [this] {
    advancedExpanded_ = !advancedExpanded_;
    advancedToggle_.setButtonText(advancedExpanded_ ? "v Advanced" : "> Advanced");
    resized();
    repaint();
  };

  // Always-visible settings
  addAndMakeVisible(rendererLabel_);
  addAndMakeVisible(rendererBox_);
  addAndMakeVisible(profileLabel_);
  addAndMakeVisible(profileBox_);
  addAndMakeVisible(masterPresetLabel_);
  addAndMakeVisible(masterPresetBox_);
  addAndMakeVisible(platformPresetLabel_);
  addAndMakeVisible(platformPresetBox_);
  addAndMakeVisible(separatedStemsToggle_);
  addAndMakeVisible(separationModelStatusLabel_);
  addAndMakeVisible(advancedToggle_);

  // Advanced settings — hidden by default, revealed by advancedToggle_
  addChildComponent(exportFormatLabel_);
  addChildComponent(exportFormatBox_);
  addChildComponent(exportModeLabel_);
  addChildComponent(exportModeBox_);
  addChildComponent(rendererChainToggle_);
  addChildComponent(rendererChainModeLabel_);
  addChildComponent(rendererChainModeBox_);
  addChildComponent(blendLabel_);
  addChildComponent(residualBlendSlider_);
  addChildComponent(batchRecursiveToggle_);
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

  // Three-column layout
  int stemWidth = std::max(200, area.getWidth() * 35 / 100);
  int meterWidth = std::max(100, area.getWidth() * 15 / 100);

  auto stemArea = area.removeFromLeft(stemWidth);
  area.removeFromLeft(spacing::gapMedium);
  auto meterArea = area.removeFromRight(meterWidth);
  area.removeFromRight(spacing::gapMedium);
  auto centerArea = area;

  stemPanel_->setBounds(stemArea);
  glowMeters_->setBounds(meterArea);

  // Action row
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

  // Always-visible context rows: Renderer/Profile and Master/Platform
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
  settingsRow2.removeFromLeft(spacing::gapSmall);
  separatedStemsToggle_.setBounds(settingsRow2.removeFromLeft(180).reduced(1));
  settingsRow2.removeFromLeft(spacing::gapSmall);
  separationModelStatusLabel_.setBounds(settingsRow2.reduced(1));

  auto chainPreviewRow = centerArea.removeFromTop(24);
  rendererChainPreviewLabel_.setBounds(chainPreviewRow.removeFromLeft(640).reduced(1));
  centerArea.removeFromTop(spacing::gapSmall);

  // Advanced disclosure row
  advancedToggle_.setBounds(centerArea.removeFromTop(24).removeFromLeft(100).reduced(1));
  centerArea.removeFromTop(spacing::gapSmall);

  // Advanced settings — shown only when expanded
  exportFormatLabel_.setVisible(advancedExpanded_);
  exportFormatBox_.setVisible(advancedExpanded_);
  exportModeLabel_.setVisible(advancedExpanded_);
  exportModeBox_.setVisible(advancedExpanded_);
  rendererChainToggle_.setVisible(advancedExpanded_);
  rendererChainModeLabel_.setVisible(advancedExpanded_);
  rendererChainModeBox_.setVisible(advancedExpanded_);
  blendLabel_.setVisible(advancedExpanded_);
  residualBlendSlider_.setVisible(advancedExpanded_);
  batchRecursiveToggle_.setVisible(advancedExpanded_);

  if (advancedExpanded_) {
    auto settingsRow3 = centerArea.removeFromTop(28);
    exportFormatLabel_.setBounds(settingsRow3.removeFromLeft(50).reduced(1));
    exportFormatBox_.setBounds(settingsRow3.removeFromLeft(100).reduced(1));
    exportModeLabel_.setBounds(settingsRow3.removeFromLeft(44).reduced(1));
    exportModeBox_.setBounds(settingsRow3.removeFromLeft(100).reduced(1));
    rendererChainToggle_.setBounds(settingsRow3.removeFromLeft(130).reduced(1));
    rendererChainModeLabel_.setBounds(settingsRow3.removeFromLeft(44).reduced(1));
    rendererChainModeBox_.setBounds(settingsRow3.removeFromLeft(170).reduced(1));

    auto settingsRow4 = centerArea.removeFromTop(28);
    blendLabel_.setBounds(settingsRow4.removeFromLeft(96).reduced(1));
    residualBlendSlider_.setBounds(settingsRow4.removeFromLeft(200).reduced(1));

    centerArea.removeFromTop(spacing::gapSmall);
    auto toggleRow = centerArea.removeFromTop(24);
    batchRecursiveToggle_.setBounds(toggleRow.removeFromLeft(160));
  }
}

void ControlDeck::setRendererChainPreviewText(const juce::String& text) {
  rendererChainPreviewLabel_.setText(text, juce::dontSendNotification);
}

void ControlDeck::setSeparationModelStatus(const juce::String& text, const bool ready) {
  separationModelStatusLabel_.setText(text, juce::dontSendNotification);
  separationModelStatusLabel_.setColour(
      juce::Label::textColourId,
      ready ? colour(colours::success) : colour(colours::warning));
}

} // namespace automix::app
