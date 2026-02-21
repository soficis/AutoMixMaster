#include "app/ui/TaskLifecycle.h"

namespace automix::app {

bool TaskLifecycle::beginTask(const ActiveTask task) {
  if (task == ActiveTask::None) {
    return false;
  }

  ActiveTask expected = ActiveTask::None;
  if (!activeTask_.compare_exchange_strong(expected, task)) {
    return false;
  }

  clearCancelFlag(task);
  taskRunning_.store(true);
  return true;
}

void TaskLifecycle::finishTask(const ActiveTask task) {
  if (task == ActiveTask::None) {
    return;
  }

  clearCancelFlag(task);
  ActiveTask expected = task;
  if (activeTask_.compare_exchange_strong(expected, ActiveTask::None)) {
    taskRunning_.store(false);
  }
}

void TaskLifecycle::clearCancelFlag(const ActiveTask task) {
  cancelFlagInternal(task).store(false);
}

ActiveTask TaskLifecycle::activeTask() const {
  return activeTask_.load();
}

bool TaskLifecycle::isTaskRunning() const {
  return taskRunning_.load();
}

std::atomic_bool& TaskLifecycle::cancelFlag(const ActiveTask task) {
  return cancelFlagInternal(task);
}

void TaskLifecycle::requestCancelForActiveTask() {
  const auto task = activeTask();
  if (task == ActiveTask::None) {
    return;
  }
  cancelFlagInternal(task).store(true);
}

void TaskLifecycle::cancelAll() {
  cancelImport_.store(true);
  cancelModel_.store(true);
  cancelSession_.store(true);
  cancelAutoMix_.store(true);
  cancelAutoMaster_.store(true);
  cancelBatch_.store(true);
  cancelExport_.store(true);
  activeTask_.store(ActiveTask::None);
  taskRunning_.store(false);
}

const char* TaskLifecycle::taskLabel(const ActiveTask task) {
  switch (task) {
    case ActiveTask::None:
      return "none";
    case ActiveTask::Import:
      return "import";
    case ActiveTask::Model:
      return "model";
    case ActiveTask::Session:
      return "session";
    case ActiveTask::AutoMix:
      return "auto-mix";
    case ActiveTask::AutoMaster:
      return "auto-master";
    case ActiveTask::Batch:
      return "batch";
    case ActiveTask::Export:
      return "export";
  }
  return "unknown";
}

std::atomic_bool& TaskLifecycle::cancelFlagInternal(const ActiveTask task) {
  switch (task) {
    case ActiveTask::Import:
      return cancelImport_;
    case ActiveTask::Model:
      return cancelModel_;
    case ActiveTask::Session:
      return cancelSession_;
    case ActiveTask::AutoMix:
      return cancelAutoMix_;
    case ActiveTask::AutoMaster:
      return cancelAutoMaster_;
    case ActiveTask::Batch:
      return cancelBatch_;
    case ActiveTask::Export:
      return cancelExport_;
    case ActiveTask::None:
      return cancelNone_;
  }
  return cancelNone_;
}

} // namespace automix::app
