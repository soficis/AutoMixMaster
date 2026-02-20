#pragma once

#include <string>
#include <vector>

#include <juce_core/juce_core.h>

#include "app/ui/TaskCenterPanel.h"
#include "app/ui/TaskLifecycle.h"

namespace automix::app {

/// Centralizes task scheduling, progress routing, and completion handling.
/// Owns TaskLifecycle and drives TaskCenterPanel state updates.
class TaskOrchestrator {
 public:
  explicit TaskOrchestrator(TaskCenterPanel& taskCenter);

  bool beginTask(ActiveTask task,
                 const juce::String& title,
                 const juce::String& details,
                 const juce::String& historyLine);

  void finishTask(ActiveTask task);
  void finishTaskCancelled(ActiveTask task, const juce::String& label);
  void finishTaskFailed(ActiveTask task, const juce::String& errorMsg);
  void finishTaskCompleted(ActiveTask task, const juce::String& label);

  void cancelActiveTask();
  void cancelAll();

  void appendHistory(const juce::String& line);
  void setStatus(const juce::String& name, const juce::String& detail);
  void setProgress(double progress);

  bool isTaskRunning() const;
  ActiveTask activeTask() const;
  std::atomic_bool& cancelFlag(ActiveTask task);

  const std::vector<juce::String>& historyLines() const;

 private:
  TaskCenterPanel& taskCenter_;
  TaskLifecycle taskLifecycle_;
  std::vector<juce::String> historyLines_;
};

} // namespace automix::app
