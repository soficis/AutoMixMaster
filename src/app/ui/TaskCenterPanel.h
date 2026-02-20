#pragma once

#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"

namespace automix::app {

enum class TaskState { Idle, Running, Cancelled, Completed, Failed };

/// Task progress display with state badge, current task, progress bar, cancel button, and history.
class TaskCenterPanel final : public juce::Component {
public:
  TaskCenterPanel();

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setCurrentTask(const juce::String& name, const juce::String& status);
  void setProgress(double progress); // 0..1 determinate, <0 indeterminate
  void setCanCancel(bool canCancel);
  void setTaskState(TaskState state);
  void appendHistory(const juce::String& line);
  void clearHistory();

  // Callbacks
  std::function<void()> onCancel;

private:
  static juce::Colour stateColour(TaskState state);
  static const char* stateLabel(TaskState state);

  juce::Label taskLabel_;
  juce::Label stateBadge_;
  juce::ProgressBar progressBar_;
  juce::Label progressLabel_;
  juce::TextButton cancelButton_{"Cancel"};
  juce::TextEditor historyEditor_;

  double progressValue_ = 0.0;
  TaskState currentState_ = TaskState::Idle;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TaskCenterPanel)
};

} // namespace automix::app
