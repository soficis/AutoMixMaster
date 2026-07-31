#pragma once

#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"
#include "engine/BatchQueueRunner.h"

namespace automix::app {

enum class TaskState { Idle, Running, Cancelled, Completed, Failed };

/// Represents a single item in the batch job queue.
struct BatchQueueItem {
  juce::String filePath;
  juce::String fileName;
  double progress = 0.0;
  TaskState state = TaskState::Idle;
  int orderIndex = 0;
};

/// Task progress display with state badge, current task, progress bar, cancel button,
/// batch queue visualization, and history.
class TaskCenterPanel final : public juce::Component {
public:
  TaskCenterPanel();

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setCurrentTask(const juce::String& name, const juce::String& status);
  void setProgress(double progress);
  void setCanCancel(bool canCancel);
  void setTaskState(TaskState state);
  void appendHistory(const juce::String& line);
  void clearHistory();

  // Batch queue
  void setQueueItems(const std::vector<BatchQueueItem>& items);
  void setQueueEta(const juce::String& eta);
  void setQueueThroughput(const juce::String& throughput);
  void moveQueueItem(int fromIndex, int toIndex);
  void removeQueueItem(int index);

  // Batch ETA / summary row
  /// Pure ETA estimate: remaining_items * avg_item_seconds where
  /// avg_item_seconds = elapsedSeconds / currentIndex (items completed so far).
  static juce::RelativeTime etaEstimate(int currentIndex, int total, double elapsedSeconds);
  /// Feed real BatchQueueRunner::ProgressDetail data (itemIndex, overallFraction,
  /// completed/failed/totalCount) plus wall-clock elapsed into the summary row.
  void setBatchProgress(const engine::BatchQueueRunner::ProgressDetail& detail, double elapsedSeconds);
  /// Current ETA row text ("—", "ETA mm:ss", or "Done").
  juce::String batchEtaText() const;
  /// Current item-status counts text ("completed/failed/total").
  juce::String batchCountsText() const;

  // Callbacks
  std::function<void()> onCancel;
  std::function<void(int fromIndex, int toIndex)> onQueueItemMoved;
  std::function<void(int index)> onQueueItemRemoved;

private:
  static constexpr int kMaxHistoryLines = 1500;
  static constexpr int kHistoryTrimChunkLines = 300;
  static constexpr int kQueueItemHeight = 24;
  static constexpr int kMaxVisibleQueueItems = 8;

  static juce::Colour stateColour(TaskState state);
  static const char* stateLabel(TaskState state);
  void drawQueueItem(juce::Graphics& g, juce::Rectangle<float> bounds, const BatchQueueItem& item, int index);
  static juce::String formatEta(const juce::RelativeTime& eta);
  void updateBatchSummary();

  juce::Label taskLabel_;
  juce::Label stateBadge_;
  juce::ProgressBar progressBar_;
  juce::Label progressLabel_;
  juce::TextButton copyLogButton_{"Copy Log"};
  juce::TextButton cancelButton_{"Cancel"};
  juce::TextEditor historyEditor_;

  // Batch queue components
  juce::Label queueHeaderLabel_;
  juce::Label queueEtaLabel_;
  juce::Label queueThroughputLabel_;
  std::vector<BatchQueueItem> queueItems_;

  // Batch ETA / summary row components
  juce::Label etaLabel_;
  juce::Label batchCountsLabel_;
  double batchProgressValue_ = 0.0;
  double batchElapsedSeconds_ = 0.0;
  engine::BatchQueueRunner::ProgressDetail batchDetail_;
  juce::ProgressBar batchProgressBar_;

  double progressValue_ = 0.0;
  TaskState currentState_ = TaskState::Idle;
  bool hasHistoryEntries_ = false;
  int historyLineCount_ = 0;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TaskCenterPanel)
};

} // namespace automix::app
