#include "app/ui/BatchProgressPanel.h"

#include <algorithm>
#include <numeric>

#include <juce_gui_basics/juce_gui_basics.h>

namespace automix::app::ui {

using namespace theme;

// ─────────────────────────────────────────────────────────────────
// Internal canvas — draws the per-item rows inside the viewport.
// ─────────────────────────────────────────────────────────────────

class BatchProgressCanvas final : public juce::Component {
public:
  BatchProgressCanvas(BatchProgressPanel& owner, std::vector<BatchProgressEntry>& itemsRef)
      : owner_(owner), items_(itemsRef) {
    setPaintingIsUnclipped(true);
  }

  void paint(juce::Graphics& g) override {
    using C = BatchProgressPanel;
    const int rowH = C::kRowHeight;
    const int rowG = C::kRowGap;
    const int margin = spacing::marginMedium;
    const int count = static_cast<int>(items_.size());

    for (int i = 0; i < count; ++i) {
      auto row = juce::Rectangle<int>(margin, i * (rowH + rowG),
                                      getWidth() - margin * 2, rowH);
      drawRow(g, row, items_[static_cast<size_t>(i)]);
    }
  }

private:
  BatchProgressPanel& owner_;
  std::vector<BatchProgressEntry>& items_;

  static void drawRow(juce::Graphics& g, juce::Rectangle<int> row, const BatchProgressEntry& entry) {
    auto bounds = row.toFloat().reduced(0.0f, 1.0f);
    auto corner = metrics::cornerRadiusSmall;

    // Background
    g.setColour(colour(colours::surface).withAlpha(0.5f));
    g.fillRoundedRectangle(bounds, corner);

    // Status dot (left)
    auto dotBounds = bounds.removeFromLeft(12.0f).withSizeKeepingCentre(8.0f, 8.0f);
    g.setColour(BatchProgressPanel::colourForStatus(entry.status));
    g.fillEllipse(dotBounds);

    // Item name (~45% width)
    auto nameArea = bounds.removeFromLeft(bounds.getWidth() * 0.45f).reduced(4.0f, 0.0f);
    g.setColour(colour(colours::text));
    g.setFont(typography::body());
    g.drawText(entry.name, nameArea, juce::Justification::centredLeft);

    // Per-item progress bar (~35% width)
    auto barArea = bounds.removeFromLeft(bounds.getWidth() * 0.35f).reduced(2.0f, 6.0f);
    {
      g.setColour(colour(colours::surfaceBorder));
      g.fillRoundedRectangle(barArea.toFloat(), corner);
      if (entry.progress > 0.0) {
        auto fill = barArea.toFloat().withWidth(barArea.toFloat().getWidth()
                                                 * static_cast<float>(entry.progress));
        g.setColour(BatchProgressPanel::colourForStatus(entry.status));
        g.fillRoundedRectangle(fill, corner);
      }
    }

    // Status label (right)
    auto statusArea = bounds.reduced(4.0f, 0.0f);
    g.setColour(colour(colours::textMuted));
    g.setFont(typography::caption());
    g.drawText(entry.statusMessage.isNotEmpty()
                   ? entry.statusMessage
                   : BatchProgressPanel::labelForStatus(entry.status),
               statusArea, juce::Justification::centredRight);
  }
};

// ─────────────────────────────────────────────────────────────────
// Construction / Destruction
// ─────────────────────────────────────────────────────────────────

BatchProgressPanel::BatchProgressPanel() {
  headerLabel_.setText("Batch Progress", juce::dontSendNotification);
  headerLabel_.setFont(typography::subhead());
  headerLabel_.setColour(juce::Label::textColourId, colour(colours::text));
  addAndMakeVisible(headerLabel_);

  summaryLabel_.setFont(typography::caption());
  summaryLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  addAndMakeVisible(summaryLabel_);

  etaLabel_.setFont(typography::caption());
  etaLabel_.setColour(juce::Label::textColourId, colour(colours::textMuted));
  etaLabel_.setJustificationType(juce::Justification::right);
  addAndMakeVisible(etaLabel_);

  canvas_ = std::make_unique<BatchProgressCanvas>(*this, items_);
  viewport_.setViewedComponent(canvas_.get(), false);
  viewport_.setScrollBarsShown(true, false);
  addAndMakeVisible(viewport_);
}

BatchProgressPanel::~BatchProgressPanel() = default;

// ─────────────────────────────────────────────────────────────────
// Paint
// ─────────────────────────────────────────────────────────────────

void BatchProgressPanel::paint(juce::Graphics& g) {
  g.fillAll(colour(colours::background));

  auto area = getLocalBounds();

  // Header background strip
  auto headerArea = area.removeFromTop(kHeaderHeight);
  g.setColour(colour(colours::surface));
  g.fillRect(headerArea);

  // Overall progress bar
  auto barArea = area.removeFromTop(kOverallBarHeight + spacing::gapMedium)
                     .reduced(spacing::marginMedium, 0);
  barArea.removeFromTop(spacing::gapSmall);
  drawOverallBar(g, barArea);
}

void BatchProgressPanel::resized() {
  auto area = getLocalBounds();

  headerLabel_.setBounds(area.removeFromTop(kHeaderHeight).reduced(spacing::marginMedium, 0));

  auto infoArea = area.removeFromTop(20).reduced(spacing::marginMedium, 0);
  summaryLabel_.setBounds(infoArea.removeFromLeft(infoArea.getWidth() / 2));
  etaLabel_.setBounds(infoArea);

  area.removeFromTop(kOverallBarHeight + spacing::gapMedium);

  viewport_.setBounds(area.reduced(spacing::marginMedium, 0));

  const int canvasH = static_cast<int>(items_.size()) * (kRowHeight + kRowGap) + spacing::marginMedium;
  const int scrollBarW = viewport_.getScrollBarThickness();
  canvas_->setBounds(0, 0, viewport_.getWidth() - scrollBarW, std::max(canvasH, area.getHeight()));
}

// ─────────────────────────────────────────────────────────────────
// Model manipulation
// ─────────────────────────────────────────────────────────────────

void BatchProgressPanel::setBatchItems(const std::vector<BatchProgressEntry>& items) {
  items_ = items;
  completedCount_ = 0;
  failedCount_ = 0;
  completedSamples_.clear();

  if (!items_.empty())
    batchStart_ = std::chrono::steady_clock::now();
  else
    batchStart_.reset();

  for (const auto& item : items_) {
    if (item.status == BatchItemStatus::Completed) ++completedCount_;
    else if (item.status == BatchItemStatus::Failed) ++failedCount_;
  }

  rebuildCanvas();
}

void BatchProgressPanel::addBatchItem(const juce::String& name) {
  BatchProgressEntry entry;
  entry.name = name;
  entry.progress = 0.0;
  entry.status = BatchItemStatus::Pending;
  items_.push_back(std::move(entry));

  if (!batchStart_.has_value())
    batchStart_ = std::chrono::steady_clock::now();

  rebuildCanvas();
}

void BatchProgressPanel::updateItemStatus(const int index, const double progress,
                                          const juce::String& status) {
  if (index < 0 || index >= static_cast<int>(items_.size()))
    return;

  auto& item = items_[static_cast<size_t>(index)];
  const auto oldStatus = item.status;

  item.progress = progress;
  if (progress >= 1.0)
    item.status = BatchItemStatus::Completed;
  else
    item.status = BatchItemStatus::Processing;

  if (status.isNotEmpty())
    item.statusMessage = status;

  if (oldStatus != BatchItemStatus::Completed && item.status == BatchItemStatus::Completed) {
    ++completedCount_;
    recordCompletion();
  }

  repaint();
  canvas_->repaint();
}

void BatchProgressPanel::clearBatch() {
  items_.clear();
  completedSamples_.clear();
  completedCount_ = 0;
  failedCount_ = 0;
  batchStart_.reset();
  rebuildCanvas();
}

// ─────────────────────────────────────────────────────────────────
// ETA / Summary
// ─────────────────────────────────────────────────────────────────

void BatchProgressPanel::recordCompletion() {
  const auto now = std::chrono::steady_clock::now();
  if (!batchStart_.has_value())
    return;
  completedSamples_.push_back({*batchStart_, now});
  if (completedSamples_.size() > 50)
    completedSamples_.erase(completedSamples_.begin());
}

juce::String BatchProgressPanel::buildEtaText() const {
  if (completedSamples_.empty() || items_.empty())
    return {};

  const int total = static_cast<int>(items_.size());
  const int done = completedCount_ + failedCount_;
  const int remaining = total - done;
  if (remaining <= 0)
    return "Done";

  auto totalDuration = std::chrono::milliseconds::zero();
  for (const auto& sample : completedSamples_)
    totalDuration += std::chrono::duration_cast<std::chrono::milliseconds>(sample.end - sample.start);

  const auto avgMs = totalDuration.count() / static_cast<double>(completedSamples_.size());
  const auto etaSec = static_cast<int>((avgMs * remaining) / 1000.0);

  if (etaSec < 60)
    return "ETA: " + juce::String(etaSec) + "s";
  if (etaSec < 3600)
    return "ETA: " + juce::String(etaSec / 60) + "m " + juce::String(etaSec % 60) + "s";
  return "ETA: " + juce::String(etaSec / 3600) + "h " + juce::String((etaSec % 3600) / 60) + "m";
}

juce::String BatchProgressPanel::buildSummaryText() const {
  const int total = static_cast<int>(items_.size());
  if (total == 0)
    return "No items";
  return juce::String(completedCount_) + " / " + juce::String(total) + " completed"
         + (failedCount_ > 0 ? " (" + juce::String(failedCount_) + " failed)" : "");
}

// ─────────────────────────────────────────────────────────────────
// Drawing helpers
// ─────────────────────────────────────────────────────────────────

juce::Colour BatchProgressPanel::colourForStatus(const BatchItemStatus status) {
  switch (status) {
    case BatchItemStatus::Pending:    return colour(colours::textDisabled);
    case BatchItemStatus::Processing: return colour(colours::primary);
    case BatchItemStatus::Completed:  return colour(colours::success);
    case BatchItemStatus::Failed:     return colour(colours::error);
  }
  return colour(colours::textDisabled);
}

juce::String BatchProgressPanel::labelForStatus(const BatchItemStatus status) {
  switch (status) {
    case BatchItemStatus::Pending:    return "Pending";
    case BatchItemStatus::Processing: return "Processing";
    case BatchItemStatus::Completed:  return "Complete";
    case BatchItemStatus::Failed:     return "Failed";
  }
  return {};
}

void BatchProgressPanel::drawOverallBar(juce::Graphics& g, juce::Rectangle<int> area) {
  const int total = static_cast<int>(items_.size());
  if (total == 0)
    return;

  const int done = completedCount_ + failedCount_;
  const double fraction = static_cast<double>(done) / static_cast<double>(total);

  auto bounds = area.toFloat();
  auto corner = metrics::cornerRadiusSmall;

  g.setColour(colour(colours::surfaceBorder));
  g.fillRoundedRectangle(bounds, corner);

  if (fraction > 0.0) {
    auto fill = bounds.withWidth(bounds.getWidth() * static_cast<float>(fraction));
    g.setColour(colour(colours::primary));
    g.fillRoundedRectangle(fill, corner);
  }

  g.setColour(colour(colours::text));
  g.setFont(typography::caption());
  g.drawText(juce::String(static_cast<int>(fraction * 100.0)) + "%", bounds, juce::Justification::centred);

  summaryLabel_.setText(buildSummaryText(), juce::dontSendNotification);
  etaLabel_.setText(buildEtaText(), juce::dontSendNotification);
}

// ─────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────

void BatchProgressPanel::rebuildCanvas() {
  // Release old canvas from viewport without double-deleting —
  // we own the lifetime via the unique_ptr.
  if (viewport_.getViewedComponent() != nullptr)
    viewport_.setViewedComponent(nullptr, false);
  canvas_.reset();
  canvas_ = std::make_unique<BatchProgressCanvas>(*this, items_);
  viewport_.setViewedComponent(canvas_.get(), false);
  resized();
  repaint();
}

} // namespace automix::app::ui
