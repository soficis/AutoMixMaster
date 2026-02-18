#pragma once

#include <utility>

#include <juce_events/juce_events.h>

namespace automix::util {

template <typename Fn>
inline void dispatchCallback(Fn&& callback) {
  if (juce::MessageManager::getInstanceWithoutCreating() != nullptr) {
    juce::MessageManager::callAsync(std::forward<Fn>(callback));
    return;
  }
  std::forward<Fn>(callback)();
}

} // namespace automix::util
