#pragma once

#include <atomic>

namespace automix::app {

enum class ActiveTask {
  None = 0,
  Import,
  Model,
  Session,
  AutoMix,
  AutoMaster,
  Batch,
  Export,
};

class TaskLifecycle {
 public:
  bool beginTask(ActiveTask task);
  void finishTask(ActiveTask task);
  void clearCancelFlag(ActiveTask task);

  ActiveTask activeTask() const;
  bool isTaskRunning() const;

  std::atomic_bool& cancelFlag(ActiveTask task);
  void requestCancelForActiveTask();
  void cancelAll();

  static const char* taskLabel(ActiveTask task);

 private:
  std::atomic_bool& cancelFlagInternal(ActiveTask task);

  std::atomic<ActiveTask> activeTask_ {ActiveTask::None};
  std::atomic_bool taskRunning_ {false};
  std::atomic_bool cancelImport_ {false};
  std::atomic_bool cancelModel_ {false};
  std::atomic_bool cancelSession_ {false};
  std::atomic_bool cancelAutoMix_ {false};
  std::atomic_bool cancelAutoMaster_ {false};
  std::atomic_bool cancelBatch_ {false};
  std::atomic_bool cancelExport_ {false};
  std::atomic_bool cancelNone_ {false};
};

} // namespace automix::app
