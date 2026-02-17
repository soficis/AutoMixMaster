#pragma once

#include <functional>
#include <utility>

#include <juce_core/juce_core.h>

namespace automix::util {

class BackgroundJob final : public juce::ThreadPoolJob {
 public:
  explicit BackgroundJob(std::function<void()> task)
      : juce::ThreadPoolJob("BackgroundJob"), task_(std::move(task)) {}

  JobStatus runJob() override {
    task_();
    return jobHasFinished;
  }

 private:
  std::function<void()> task_;
};

} // namespace automix::util
