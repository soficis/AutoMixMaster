#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/controllers/PreviewController.h"
#include "app/ui/StemPanel.h"
#include "domain/Session.h"
#include "engine/AudioBuffer.h"

namespace automix::app {

/// Manages preview audio buffer lifecycle: rebuild requests,
/// lock-free buffer publication for the audio callback, and generation tracking.
class AudioPreviewManager {
 public:
  AudioPreviewManager(juce::ThreadPool& pool, juce::Component* owner);

  void rebuildPreview(const domain::Session& session,
                      const std::vector<StemPanel::StemDisplay>& displays,
                      double currentProgress);

  /// Publishes a new preview buffer (message thread). The buffer is moved in;
  /// no copy of the audio data is made.
  void setBuffer(engine::AudioBuffer buffer);

  /// Realtime-safe read for the audio callback: never blocks and never
  /// allocates. Returns the current preview buffer, or nullptr before the
  /// first publish. The returned shared_ptr keeps the buffer alive even if the
  /// message thread replaces it mid-block.
  std::shared_ptr<const engine::AudioBuffer> currentBuffer() const;

  /// Called on message thread when a new preview buffer is ready.
  std::function<void(const engine::AudioBuffer& buffer, double previousProgress)> onPreviewReady;
  std::function<void(const juce::String& errorText)> onPreviewError;
  std::function<void(const juce::String& line)> onHistoryLine;

 private:
  juce::Component::SafePointer<juce::Component> owner_;
  std::unique_ptr<PreviewController> previewController_;
  mutable juce::SpinLock bufferLock_;
  std::shared_ptr<const engine::AudioBuffer> buffer_{nullptr};
  std::atomic_uint64_t generation_{0};
};

} // namespace automix::app
