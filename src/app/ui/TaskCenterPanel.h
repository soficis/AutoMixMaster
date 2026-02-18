#pragma once

#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"

namespace automix::app {

/// Task progress display with current task, progress bar, cancel button, and history.
class TaskCenterPanel final : public juce::Component {
public:
  TaskCenterPanel();

  void paint(juce::Graphics& g) override;
  void resized() override;

  void setCurrentTask(const juce::String& name, const juce::String& status);
  void setProgress(double progress); // 0..1 determinate, <0 indeterminate
  void setCanCancel(bool canCancel);
  void appendHistory(const juce::String& line);
  void clearHistory();

  // Callbacks
  std::function<void()> onCancel;

private:
  juce::Label taskLabel_;
  juce::ProgressBar progressBar_;
  juce::TextButton cancelButton_{"Cancel"};
  juce::TextEditor historyEditor_;

  double progressValue_ = 0.0;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TaskCenterPanel)
};

} // namespace automix::app
