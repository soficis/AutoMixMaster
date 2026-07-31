#pragma once

#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"

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

  double progressValue_ = 0.0;
  TaskState currentState_ = TaskState::Idle;
  bool hasHistoryEntries_ = false;
  int historyLineCount_ = 0;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TaskCenterPanel)
};

} // namespace automix::app
