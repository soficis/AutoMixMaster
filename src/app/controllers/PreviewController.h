#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include "domain/Session.h"
#include "engine/AudioBuffer.h"
#include "engine/TransportController.h"

namespace automix::app {

struct PreviewBuildRequest {
  domain::Session session;
  std::string soloStemId;
  std::string muteStemId;
  uint64_t generation = 0;
  double previousProgress = 0.0;
};

struct PreviewBuildResult {
  uint64_t generation = 0;
  double previousProgress = 0.0;
  bool success = false;
  juce::String errorText;
  engine::AudioBuffer rawMix;
  engine::AudioBuffer mastered;
  engine::AudioBuffer preview;
};

class PreviewController {
 public:
  struct Callbacks {
    std::function<void(PreviewBuildResult)> onPreviewReady;
  };

  PreviewController(juce::ThreadPool& threadPool, Callbacks callbacks);

  void rebuildPreview(PreviewBuildRequest request);

  static void applyTransportBuffer(const engine::AudioBuffer& buffer,
                                   const domain::TimelineState& timeline,
                                   engine::TransportController& transportController,
                                   std::atomic<int64_t>& playbackCursorSamples);

 private:
  juce::ThreadPool& threadPool_;
  Callbacks callbacks_;
};

} // namespace automix::app
