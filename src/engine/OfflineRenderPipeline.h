#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "domain/Session.h"
#include "engine/AudioBuffer.h"

namespace automix::engine {

struct RenderProgress {
  double fraction = 0.0;
  std::string stage;
};

struct OfflineRenderResult {
  AudioBuffer mixBuffer;
  bool cancelled = false;
  std::vector<std::string> logs;
};

using ProgressCallback = std::function<void(const RenderProgress&)>;

class OfflineRenderPipeline {
 public:
  static void clearCaches();

  OfflineRenderResult renderRawMix(const domain::Session& session,
                                   const domain::RenderSettings& settings,
                                   const ProgressCallback& onProgress,
                                   const std::atomic_bool* cancelFlag) const;
};

} // namespace automix::engine
