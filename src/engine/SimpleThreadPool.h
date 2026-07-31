#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace automix::engine {

class SimpleThreadPool final {
 public:
  explicit SimpleThreadPool(unsigned int numThreads = 0)
      : stop_(false) {
    const auto count = numThreads > 0
                           ? numThreads
                           : std::max(1u, std::thread::hardware_concurrency());
    workers_.reserve(static_cast<size_t>(count));
    for (unsigned int i = 0; i < count; ++i) {
      workers_.emplace_back([this] { workerLoop(); });
    }
  }

  ~SimpleThreadPool() { shutdown(); }

  SimpleThreadPool(const SimpleThreadPool&) = delete;
  SimpleThreadPool& operator=(const SimpleThreadPool&) = delete;
  SimpleThreadPool(SimpleThreadPool&&) = delete;
  SimpleThreadPool& operator=(SimpleThreadPool&&) = delete;

  template <typename F>
  std::future<void> enqueue(F&& task) {
    auto packaged = std::make_shared<std::packaged_task<void()>>(
        std::forward<F>(task));
    auto future = packaged->get_future();
    {
      std::scoped_lock lock(queueMutex_);
      if (stop_) {
        return {};
      }
      tasks_.emplace([packaged]() { (*packaged)(); });
    }
    condition_.notify_one();
    return future;
  }

  void waitForAll() {
    std::unique_lock lock(queueMutex_);
    completion_.wait(lock, [this] { return tasks_.empty(); });
  }

  [[nodiscard]] size_t size() const { return workers_.size(); }

 private:
  void workerLoop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock lock(queueMutex_);
        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
      {
        std::scoped_lock lock(queueMutex_);
        if (tasks_.empty()) {
          completion_.notify_all();
        }
      }
    }
  }

  void shutdown() {
    {
      std::scoped_lock lock(queueMutex_);
      stop_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queueMutex_;
  std::condition_variable condition_;
  std::condition_variable completion_;
  std::atomic<bool> stop_;
};

} // namespace automix::engine
