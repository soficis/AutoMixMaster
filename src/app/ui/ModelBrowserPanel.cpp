#include "app/ui/ModelBrowserPanel.h"

namespace automix::app {

using namespace theme;

// ── ModelRow ───────────────────────────────────────────────────────

ModelBrowserPanel::ModelRow::ModelRow() {
  repoLabel_.setFont(typography::body());
  repoLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  repoLabel_.setJustificationType(juce::Justification::centredLeft);

  detailLabel_.setFont(typography::caption());
  detailLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  detailLabel_.setJustificationType(juce::Justification::centredLeft);

  installButton_.onClick = [this] {
    if (onInstall)
      onInstall(model_.repoId);
  };

  addAndMakeVisible(repoLabel_);
  addAndMakeVisible(detailLabel_);
  addAndMakeVisible(installButton_);
}

void ModelBrowserPanel::ModelRow::paint(juce::Graphics& g) {
  g.setColour(colour(colours::surfaceBorder));
  g.drawHorizontalLine(getHeight() - 1, 0.0f, static_cast<float>(getWidth()));
}

void ModelBrowserPanel::ModelRow::resized() {
  auto area = getLocalBounds().reduced(4);
  installButton_.setBounds(area.removeFromRight(80).reduced(2));
  auto top = area.removeFromTop(area.getHeight() / 2);
  repoLabel_.setBounds(top);
  detailLabel_.setBounds(area);
}

void ModelBrowserPanel::ModelRow::setModel(const ai::HubModelInfo& model) {
  model_ = model;
  repoLabel_.setText(juce::String(model.repoId), juce::dontSendNotification);
  juce::String detail = "rev: " + juce::String(model.revision);
  if (!model.useCase.empty())
    detail += "  use: " + juce::String(model.useCase);
  if (!model.license.empty())
    detail += "  license: " + juce::String(model.license);
  detailLabel_.setText(detail, juce::dontSendNotification);
}

// ── ModelBrowserPanel ──────────────────────────────────────────────

ModelBrowserPanel::ModelBrowserPanel() {
  statusLabel_.setText("Click 'Fetch Catalog' to browse available models", juce::dontSendNotification);
  statusLabel_.setFont(typography::body());
  statusLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  statusLabel_.setJustificationType(juce::Justification::centredLeft);

  fetchButton_.onClick = [this] {
    if (onFetchCatalog)
      onFetchCatalog();
  };
  updatesButton_.onClick = [this] {
    if (onCheckUpdates)
      onCheckUpdates();
  };
  verifyButton_.onClick = [this] {
    if (onVerifyIntegrity)
      onVerifyIntegrity();
  };

  viewport_.setViewedComponent(&rowContainer_, false);
  viewport_.setScrollBarsShown(true, false);

  addAndMakeVisible(statusLabel_);
  addAndMakeVisible(fetchButton_);
  addAndMakeVisible(updatesButton_);
  addAndMakeVisible(verifyButton_);
  addAndMakeVisible(viewport_);
}

void ModelBrowserPanel::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::surface));
}

void ModelBrowserPanel::resized() {
  auto area = getLocalBounds().reduced(static_cast<int>(metrics::paddingMedium));

  auto statusRow = area.removeFromTop(28);
  statusLabel_.setBounds(statusRow);

  area.removeFromTop(4);
  auto buttonRow = area.removeFromTop(32);
  fetchButton_.setBounds(buttonRow.removeFromLeft(120).reduced(2));
  updatesButton_.setBounds(buttonRow.removeFromLeft(120).reduced(2));
  verifyButton_.setBounds(buttonRow.removeFromLeft(80).reduced(2));

  area.removeFromTop(4);
  viewport_.setBounds(area);

  // Resize existing rows to match viewport width
  for (auto& row : modelRows_) {
    auto bounds = row->getBounds();
    row->setBounds(bounds.withWidth(viewport_.getWidth() - viewport_.getScrollBarThickness()));
  }
}

void ModelBrowserPanel::setDiscoveredModels(const std::vector<ai::HubModelInfo>& models) {
  models_ = models;
  rebuildModelRows();
  statusLabel_.setText(juce::String(static_cast<int>(models.size())) + " models found",
                       juce::dontSendNotification);
}

void ModelBrowserPanel::setStatus(const juce::String& status) {
  statusLabel_.setText(status, juce::dontSendNotification);
}

void ModelBrowserPanel::setActionsEnabled(bool enabled) {
  fetchButton_.setEnabled(enabled);
  updatesButton_.setEnabled(enabled);
  verifyButton_.setEnabled(enabled);
  for (auto& row : modelRows_)
    row->setEnabled(enabled);
}

void ModelBrowserPanel::rebuildModelRows() {
  modelRows_.clear();
  rowContainer_.removeAllChildren();

  constexpr int rowHeight = 52;
  const int rowWidth = viewport_.getWidth() - viewport_.getScrollBarThickness();
  int y = 0;

  for (const auto& model : models_) {
    auto row = std::make_unique<ModelRow>();
    row->setModel(model);
    row->onInstall = [this](const std::string& repoId) {
      if (onInstallModel)
        onInstallModel(repoId);
    };
    row->setBounds(0, y, std::max(rowWidth, 300), rowHeight);
    rowContainer_.addAndMakeVisible(row.get());
    modelRows_.push_back(std::move(row));
    y += rowHeight;
  }

  rowContainer_.setSize(std::max(rowWidth, 300), y);
}

} // namespace automix::app
