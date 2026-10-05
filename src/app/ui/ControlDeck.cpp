#include "app/ui/ControlDeck.h"

#include "app/style/AutoMixLookAndFeel.h"
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
  autoMasterButton_.setTooltip("Auto Master (Ctrl+Shift+A)");
  autoMixMasterButton_.setTooltip("One-click: Auto Mix -> Auto Master -> Export (Ctrl+Shift+M)");
  batchButton_.setTooltip("Batch Process");
  exportButton_.setTooltip("Export (Ctrl+E)");
  separatedStemsToggle_.setTooltip("Split a single imported full mix into stems using the active Separation model pack.");
  tensorSeparationToggle_.setTooltip(
      "Use a vocal-separation model pack (e.g. BS-RoFormer) when AI Stem Separation runs. Produces vocals plus an "
      "instrumental that is the residual (mix - vocals), not a second separation. Takes minutes on CPU; falls back to "
      "the standard separator if the pack cannot run.");
  tensorSeparationToggle_.setEnabled(false);
  rendererBox_.setTooltip("Engine that renders the exported file. BuiltIn needs no external tools.");
  exportModeBox_.setTooltip("Final renders at full quality. Quick Preview renders faster for checking the result.");
  masterPresetBox_.setTooltip("Mastering target. Default Streaming suits Spotify, Apple Music and YouTube.");
  platformPresetBox_.setTooltip("Loudness target for the platform you will publish to.");

  // Accessibility titles (match the visible labels)
  rendererBox_.setTitle("Renderer");
  profileBox_.setTitle("Profile");
  masterPresetBox_.setTitle("Master preset");
  platformPresetBox_.setTitle("Platform");
  exportFormatBox_.setTitle("Export format");
  exportModeBox_.setTitle("Render mode");
  rendererChainModeBox_.setTitle("Chain");
  residualBlendSlider_.setTitle("Residual blend");
  batchRecursiveToggle_.setTooltip("Include subfolders when scanning batch input");
  rendererChainToggle_.setTooltip("Run renderers in a staged chain");
  rendererChainModeBox_.setTooltip("Renderer chain strategy");
  residualBlendSlider_.setTooltip("Control residual audio blend");

  // Button visual hierarchy: primary > secondary > quiet (see buttonVariant in AutoMixLookAndFeel.h).
  setButtonVariant(autoMixMasterButton_, buttonVariant::primary);
  for (juce::TextButton* btn : {&autoMixButton_, &autoMasterButton_, &batchButton_, &exportButton_})
    setButtonVariant(*btn, buttonVariant::secondary);
  setButtonVariant(importButton_, buttonVariant::primary);

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
  addChildComponent(rendererChainPreviewLabel_);
  blendLabel_.setFont(typography::caption());
  blendLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  blendLabel_.setJustificationType(juce::Justification::centredLeft);
  separationModelStatusLabel_.setFont(typography::caption());
  separationModelStatusLabel_.setColour(juce::Label::textColourId, colour(colours::warning));
  separationModelStatusLabel_.setJustificationType(juce::Justification::centredLeft);
  separationModelStatusLabel_.setText("Model: none installed", juce::dontSendNotification);

  residualBlendSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  residualBlendSlider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 20);
  residualBlendSlider_.setRange(0.0, 10.0, 0.1);
  residualBlendSlider_.setValue(0.0, juce::dontSendNotification);
  rendererChainModeBox_.setEnabled(false);

  // Advanced disclosure toggle
  setButtonVariant(advancedToggle_, buttonVariant::quiet);
  advancedToggle_.onClick = [this] {
    advancedExpanded_ = !advancedExpanded_;
    advancedToggle_.setButtonText(advancedExpanded_ ? "v Advanced" : "> Advanced");
    resized();
    repaint();
  };

  // Always-visible settings
  addAndMakeVisible(profileLabel_);
  addAndMakeVisible(profileBox_);
  addAndMakeVisible(masterPresetLabel_);
  addAndMakeVisible(masterPresetBox_);
  addAndMakeVisible(platformPresetLabel_);
  addAndMakeVisible(platformPresetBox_);
  addAndMakeVisible(separatedStemsToggle_);
  addChildComponent(tensorSeparationToggle_);
  addChildComponent(separationModelStatusLabel_);
  addAndMakeVisible(advancedToggle_);

  // Advanced settings — hidden by default, revealed by advancedToggle_
  addChildComponent(rendererLabel_);
  addChildComponent(rendererBox_);
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

  setHasStems(false);
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
  int stemWidth = juce::jlimit(200, 360, area.getWidth() * 28 / 100);
  int meterWidth = juce::jlimit(110, 180, area.getWidth() * 13 / 100);

  auto stemArea = area.removeFromLeft(stemWidth);
  area.removeFromLeft(spacing::gapMedium);
  auto meterArea = area.removeFromRight(meterWidth);
  area.removeFromRight(spacing::gapMedium);
  auto centerArea = area;

  stemPanel_->setBounds(stemArea);
  glowMeters_->setBounds(meterArea);

  // Action row
  // Slots: import(1) | autoMix(1) | autoMaster(1) | Mix+Master(1.5) | batch(1) | export(1) = 6.5 slots
  auto actionRow = centerArea.removeFromTop(44);
  const int slotW = juce::roundToInt(static_cast<float>(actionRow.getWidth()) / 6.5f);
  const bool compact = slotW < 90;
  for (auto* b : {&importButton_, &autoMixButton_, &autoMasterButton_, &autoMixMasterButton_,
                  &batchButton_, &exportButton_})
    b->getProperties().set("compact", compact);
  importButton_.setBounds(actionRow.removeFromLeft(slotW).reduced(2));
  autoMixButton_.setBounds(actionRow.removeFromLeft(slotW).reduced(2));
  autoMasterButton_.setBounds(actionRow.removeFromLeft(slotW).reduced(2));
  autoMixMasterButton_.setBounds(actionRow.removeFromLeft(juce::roundToInt(static_cast<float>(slotW) * 1.5f)).reduced(2));
  batchButton_.setBounds(actionRow.removeFromLeft(slotW).reduced(2));
  exportButton_.setBounds(actionRow.reduced(2));

  centerArea.removeFromTop(spacing::gapSmall);

  // Always-visible context rows: Profile/Master/Platform, then the separation row
  // Pairs wrap onto a new row when they do not fit; combos shrink toward their minimum first.
  {
    struct Pair {
      juce::Label* label;
      juce::ComboBox* box;
      int labelW, prefW, minW;
    };
    const Pair pairs[] = {{&profileLabel_, &profileBox_, 50, 200, 150},
                          {&masterPresetLabel_, &masterPresetBox_, 50, 170, 130},
                          {&platformPresetLabel_, &platformPresetBox_, 64, 150, 120}};
    const int rowWidth = centerArea.getWidth();
    auto row = centerArea.removeFromTop(28);
    int x = 0;
    for (const auto& p : pairs) {
      const int avail = rowWidth - x - p.labelW;
      if (x > 0 && avail < p.minW) {
        row = centerArea.removeFromTop(28);
        x = 0;
      }
      const int comboW = std::min(p.prefW, std::max(p.minW, rowWidth - x - p.labelW));
      p.label->setBounds(row.getX() + x, row.getY(), p.labelW, row.getHeight());
      p.label->setBounds(p.label->getBounds().reduced(1));
      p.box->setBounds(juce::Rectangle<int>(row.getX() + x + p.labelW, row.getY(), comboW, row.getHeight()).reduced(1));
      x += p.labelW + comboW;
    }
  }

  auto separationRow = centerArea.removeFromTop(28);
  separatedStemsToggle_.setBounds(separationRow.removeFromLeft(180).reduced(1));
  tensorSeparationToggle_.setVisible(separationControlsVisible_);
  separationModelStatusLabel_.setVisible(separationControlsVisible_);
  if (separationControlsVisible_) {
    separationRow.removeFromLeft(16);
    tensorSeparationToggle_.setBounds(separationRow.removeFromLeft(120).reduced(1));
    separationRow.removeFromLeft(spacing::gapSmall);
    separationModelStatusLabel_.setBounds(separationRow.reduced(1));
  }
  centerArea.removeFromTop(spacing::gapSmall);

  // Advanced disclosure row
  advancedToggle_.setBounds(centerArea.removeFromTop(24).removeFromLeft(100).reduced(1));
  centerArea.removeFromTop(spacing::gapSmall);

  // Advanced settings — shown only when expanded
  rendererLabel_.setVisible(advancedExpanded_);
  rendererBox_.setVisible(advancedExpanded_);
  rendererChainPreviewLabel_.setVisible(advancedExpanded_);
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
    auto rendererRow = centerArea.removeFromTop(28);
    rendererLabel_.setBounds(rendererRow.removeFromLeft(64).reduced(1));
    rendererBox_.setBounds(rendererRow.removeFromLeft(160).reduced(1));
    rendererRow.removeFromLeft(spacing::gapSmall);
    rendererChainPreviewLabel_.setBounds(rendererRow.removeFromLeft(640).reduced(1));

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
      ready ? colour(colours::textMuted) : colour(colours::warning));
}

void ControlDeck::setHasStems(const bool hasStems) {
  autoMixButton_.setEnabled(hasStems);
  autoMasterButton_.setEnabled(hasStems);
  autoMixMasterButton_.setEnabled(hasStems);
  exportButton_.setEnabled(hasStems);
  batchButton_.setEnabled(true);
  importButton_.setEnabled(true);
  setButtonVariant(importButton_, hasStems ? buttonVariant::secondary : buttonVariant::primary);
}

void ControlDeck::setSeparationControlsVisible(const bool visible) {
  separationControlsVisible_ = visible;
  resized();
}

} // namespace automix::app
