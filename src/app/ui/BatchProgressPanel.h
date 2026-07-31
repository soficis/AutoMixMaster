#pragma once

#include <chrono>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"

namespace automix::app::ui {

/// Status of an individual batch item.
enum class BatchItemStatus {
  Pending,
  Processing,
  Completed,
  Failed
};

/// Data for one batch queue entry.
struct BatchProgressEntry {
  juce::String name;
  double progress = 0.0;          // 0.0 – 1.0
  BatchItemStatus status = BatchItemStatus::Pending;
  juce::String statusMessage;
};

/// Panel that visualises the batch processing queue.
/// Shows an overall progress bar + ETA at the top, then a scrollable
/// list of per-item progress bars.
class BatchProgressPanel final : public juce::Component {
public:
  BatchProgressPanel();
  ~BatchProgressPanel() override;

  void paint(juce::Graphics& g) override;
  void resized() override;

  /// Replace the entire item list.
  void setBatchItems(const std::vector<BatchProgressEntry>& items);

  /// Add a single item at the end.
  void addBatchItem(const juce::String& name);

  /// Update progress / status of item at @p index.
  void updateItemStatus(int index, double progress, const juce::String& status);

  /// Clear all items and reset progress.
  void clearBatch();

private:
  // ── Model ─────────────────────────────────────────────────────
  std::vector<BatchProgressEntry> items_;

  // ── ETA tracking ──────────────────────────────────────────────
  struct ProcessingSample {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
  };
  std::vector<ProcessingSample> completedSamples_;
  std::optional<std::chrono::steady_clock::time_point> batchStart_;
  int completedCount_ = 0;
  int failedCount_ = 0;

  void recordCompletion();
  juce::String buildEtaText() const;
  juce::String buildSummaryText() const;

  // ── Layout helpers ────────────────────────────────────────────
public:
  static juce::Colour colourForStatus(BatchItemStatus status);
  static juce::String labelForStatus(BatchItemStatus status);
private:
  void drawOverallBar(juce::Graphics& g, juce::Rectangle<int> area);
  void rebuildCanvas();

  // ── Child components ──────────────────────────────────────────
  juce::Label headerLabel_;
  juce::Label summaryLabel_;
  juce::Label etaLabel_;
  juce::Viewport viewport_;
  std::unique_ptr<juce::Component> canvas_;  // BatchProgressCanvas instance

  // ── Constants ─────────────────────────────────────────────────
public:
  static constexpr int kHeaderHeight = 24;
  static constexpr int kOverallBarHeight = 20;
  static constexpr int kRowHeight = 36;
  static constexpr int kRowGap = 2;
  static constexpr int kMaxVisibleRows = 10;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BatchProgressPanel)
};

} // namespace automix::app::ui
