#pragma once

#include <atomic>
#include <functional>
#include <optional>
#include <string>

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include "domain/Session.h"

namespace automix::app {

struct SessionSaveResult {
  bool success = false;
  bool cancelled = false;
  std::string path;
  juce::String errorText;
};

struct SessionLoadResult {
  bool cancelled = false;
  std::optional<domain::Session> session;
  std::string path;
  juce::String errorText;
};

class SessionController {
 public:
  struct Callbacks {
    std::function<void(const std::string&)> onStatus;
    std::function<void(const std::string&)> onTaskHistory;
    std::function<void(double)> onProgress;
    std::function<void(SessionSaveResult)> onSaveComplete;
    std::function<void(SessionLoadResult)> onLoadComplete;
  };

  SessionController(juce::ThreadPool& threadPool, Callbacks callbacks);

  void saveSession(std::string path, domain::Session session, std::atomic_bool& cancelFlag);
  void loadSession(std::string path, std::atomic_bool& cancelFlag);

 private:
  juce::ThreadPool& threadPool_;
  Callbacks callbacks_;
};

} // namespace automix::app
