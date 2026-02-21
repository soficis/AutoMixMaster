#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/controllers/PreviewController.h"
#include "app/ui/StemPanel.h"
#include "domain/Session.h"
#include "engine/AudioBuffer.h"

namespace automix::app {

/// Manages preview audio buffer lifecycle: rebuild requests,
/// thread-safe buffer access for the audio callback, and generation tracking.
class AudioPreviewManager {
 public:
  AudioPreviewManager(juce::ThreadPool& pool, juce::Component* owner);

  void rebuildPreview(const domain::Session& session,
                      const std::vector<StemPanel::StemDisplay>& displays,
                      double currentProgress);

  /// Direct buffer update (e.g. from auto-master results).
  void setBuffer(const engine::AudioBuffer& buffer);

  /// Buffer access for audio callback. Caller must lock bufferMutex() first.
  std::mutex& bufferMutex();
  const engine::AudioBuffer& buffer() const;

  /// Called on message thread when a new preview buffer is ready.
  std::function<void(const engine::AudioBuffer& buffer, double previousProgress)> onPreviewReady;
  std::function<void(const juce::String& errorText)> onPreviewError;
  std::function<void(const juce::String& line)> onHistoryLine;

 private:
  juce::Component::SafePointer<juce::Component> owner_;
  std::unique_ptr<PreviewController> previewController_;
  std::mutex bufferMutex_;
  engine::AudioBuffer buffer_;
  std::atomic_uint64_t generation_{0};
};

} // namespace automix::app
