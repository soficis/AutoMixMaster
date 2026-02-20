#include "app/ui/TaskOrchestrator.h"

namespace automix::app {

TaskOrchestrator::TaskOrchestrator(TaskCenterPanel& taskCenter)
    : taskCenter_(taskCenter) {}

bool TaskOrchestrator::beginTask(const ActiveTask task,
                                 const juce::String& title,
                                 const juce::String& details,
                                 const juce::String& historyLine) {
  if (!taskLifecycle_.beginTask(task)) {
    taskCenter_.setCurrentTask("Busy", "A task is already running");
    return false;
  }

  taskCenter_.setCanCancel(true);
  taskCenter_.setCurrentTask(title, details);
  taskCenter_.setProgress(-1.0);
  taskCenter_.setTaskState(TaskState::Running);
  appendHistory(historyLine);
  return true;
}

void TaskOrchestrator::finishTask(const ActiveTask task) {
  taskLifecycle_.finishTask(task);
  taskCenter_.setCanCancel(false);
  taskCenter_.setTaskState(TaskState::Idle);
}

void TaskOrchestrator::finishTaskCancelled(const ActiveTask task, const juce::String& label) {
  taskLifecycle_.finishTask(task);
  taskCenter_.setCanCancel(false);
  taskCenter_.setCurrentTask(label, "");
  taskCenter_.setTaskState(TaskState::Cancelled);
}

void TaskOrchestrator::finishTaskFailed(const ActiveTask task, const juce::String& errorMsg) {
  taskLifecycle_.finishTask(task);
  taskCenter_.setCanCancel(false);
  taskCenter_.setCurrentTask("Failed", "");
  taskCenter_.setTaskState(TaskState::Failed);
  appendHistory("Error: " + errorMsg);
}

void TaskOrchestrator::finishTaskCompleted(const ActiveTask task, const juce::String& label) {
  taskLifecycle_.finishTask(task);
  taskCenter_.setCanCancel(false);
  taskCenter_.setCurrentTask(label, "");
  taskCenter_.setProgress(1.0);
  taskCenter_.setTaskState(TaskState::Completed);
}

void TaskOrchestrator::cancelActiveTask() {
  const auto task = taskLifecycle_.activeTask();
  if (task == ActiveTask::None) {
    taskCenter_.setCurrentTask("No active task", "");
    return;
  }

  taskLifecycle_.cancelFlag(task).store(true);
  taskCenter_.setCurrentTask("Cancelling", "");
  appendHistory("Cancel requested: " + juce::String(TaskLifecycle::taskLabel(task)));
}

void TaskOrchestrator::cancelAll() {
  taskLifecycle_.cancelAll();
}

void TaskOrchestrator::appendHistory(const juce::String& line) {
  historyLines_.push_back(line);
  taskCenter_.appendHistory(line);
}

void TaskOrchestrator::setStatus(const juce::String& name, const juce::String& detail) {
  taskCenter_.setCurrentTask(name, detail);
}

void TaskOrchestrator::setProgress(double progress) {
  taskCenter_.setProgress(progress);
}

bool TaskOrchestrator::isTaskRunning() const {
  return taskLifecycle_.isTaskRunning();
}

ActiveTask TaskOrchestrator::activeTask() const {
  return taskLifecycle_.activeTask();
}

std::atomic_bool& TaskOrchestrator::cancelFlag(const ActiveTask task) {
  return taskLifecycle_.cancelFlag(task);
}

const std::vector<juce::String>& TaskOrchestrator::historyLines() const {
  return historyLines_;
}

} // namespace automix::app
