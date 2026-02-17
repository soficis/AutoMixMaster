#include "app/controllers/SessionController.h"

#include <exception>
#include <utility>

#include "engine/SessionRepository.h"
#include "util/CallbackDispatch.h"

namespace automix::app {

SessionController::SessionController(juce::ThreadPool& threadPool, Callbacks callbacks)
    : threadPool_(threadPool), callbacks_(std::move(callbacks)) {}

void SessionController::saveSession(std::string path,
                                    domain::Session session,
                                    std::atomic_bool& cancelFlag) {
  if (path.empty()) {
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Saving session...");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Session save started: " + path);
  }

  struct SaveSessionJob final : juce::ThreadPoolJob {
    std::string path;
    domain::Session session;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;

    SaveSessionJob(std::string outputPath,
                   domain::Session sourceSession,
                   std::atomic_bool* cancel,
                   Callbacks cb)
        : juce::ThreadPoolJob("SaveSessionJob"),
          path(std::move(outputPath)),
          session(std::move(sourceSession)),
          cancelFlag(cancel),
          callbacks(std::move(cb)) {}

    bool isCancellationRequested() const {
      return shouldExit() || (cancelFlag != nullptr && cancelFlag->load());
    }

    void requestCancellation() const {
      if (cancelFlag != nullptr) {
        cancelFlag->store(true);
      }
    }

    JobStatus runJob() override {
      SessionSaveResult result;
      result.path = path;
      if (isCancellationRequested()) {
        requestCancellation();
        result.cancelled = true;
      }

      try {
        if (!result.cancelled) {
          engine::SessionRepository repository;
          repository.save(path, session);
          result.success = true;
        }
        if (isCancellationRequested()) {
          requestCancellation();
          result.cancelled = true;
          result.success = false;
        }
      } catch (const std::exception& error) {
        result.errorText = error.what();
      } catch (...) {
        result.errorText = "Unknown session save error";
      }

      if (result.cancelled) {
        if (callbacks.onStatus) {
          callbacks.onStatus("Session save cancelled");
        }
        if (callbacks.onTaskHistory) {
          callbacks.onTaskHistory("Session save cancelled: " + path);
        }
      }

      auto capturedCallbacks = callbacks;
      util::dispatchCallback([capturedCallbacks, result = std::move(result)]() mutable {
        if (capturedCallbacks.onSaveComplete) {
          capturedCallbacks.onSaveComplete(std::move(result));
        }
      });

      return jobHasFinished;
    }
  };

  threadPool_.addJob(new SaveSessionJob(std::move(path), std::move(session), &cancelFlag, callbacks_), true);
}

void SessionController::loadSession(std::string path, std::atomic_bool& cancelFlag) {
  if (path.empty()) {
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Loading session...");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Session load started: " + path);
  }

  struct LoadSessionJob final : juce::ThreadPoolJob {
    std::string path;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;

    LoadSessionJob(std::string inputPath, std::atomic_bool* cancel, Callbacks cb)
        : juce::ThreadPoolJob("LoadSessionJob"),
          path(std::move(inputPath)),
          cancelFlag(cancel),
          callbacks(std::move(cb)) {}

    bool isCancellationRequested() const {
      return shouldExit() || (cancelFlag != nullptr && cancelFlag->load());
    }

    void requestCancellation() const {
      if (cancelFlag != nullptr) {
        cancelFlag->store(true);
      }
    }

    JobStatus runJob() override {
      SessionLoadResult result;
      result.path = path;
      if (isCancellationRequested()) {
        requestCancellation();
        result.cancelled = true;
      }

      try {
        if (!result.cancelled) {
          engine::SessionRepository repository;
          result.session = repository.load(path);
        }
        if (isCancellationRequested()) {
          requestCancellation();
          result.cancelled = true;
          result.session.reset();
        }
      } catch (const std::exception& error) {
        result.errorText = error.what();
      } catch (...) {
        result.errorText = "Unknown session load error";
      }

      if (result.cancelled) {
        if (callbacks.onStatus) {
          callbacks.onStatus("Session load cancelled");
        }
        if (callbacks.onTaskHistory) {
          callbacks.onTaskHistory("Session load cancelled: " + path);
        }
      }

      auto capturedCallbacks = callbacks;
      util::dispatchCallback([capturedCallbacks, result = std::move(result)]() mutable {
        if (capturedCallbacks.onLoadComplete) {
          capturedCallbacks.onLoadComplete(std::move(result));
        }
      });

      return jobHasFinished;
    }
  };

  threadPool_.addJob(new LoadSessionJob(std::move(path), &cancelFlag, callbacks_), true);
}

} // namespace automix::app
