#pragma once

#include <cstddef>

#include "engine/AudioBuffer.h"

namespace automix::engine {

struct LoudnessMetrics {
  double integratedLufs = -120.0;
  double shortTermLufs = -120.0;
  double loudnessRange = 0.0;
  double samplePeakDbfs = -120.0;
};

class LoudnessMeter {
 public:
  LoudnessMetrics analyze(const AudioBuffer& buffer, size_t chunkSize = 1024) const;
  double computeIntegratedLufs(const AudioBuffer& buffer, size_t chunkSize = 1024) const;
  double computeShortTermLufs(const AudioBuffer& buffer, size_t chunkSize = 1024) const;
  double computeLoudnessRange(const AudioBuffer& buffer, size_t chunkSize = 1024) const;
  double computeSamplePeakDbfs(const AudioBuffer& buffer) const;
};

} // namespace automix::engine
