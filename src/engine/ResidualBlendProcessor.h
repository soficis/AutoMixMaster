#pragma once

#include "engine/AudioBuffer.h"

namespace automix::engine {

struct AlignmentResult {
  int sampleOffset = 0;
  double normalizedCorrelation = 0.0;
};

struct ResidualComputation {
  AudioBuffer alignedOriginalMix;
  AudioBuffer residual;
  AlignmentResult alignment;
};

class ResidualBlendProcessor {
 public:
  AlignmentResult estimateAlignment(const AudioBuffer& stemSum,
                                    const AudioBuffer& originalMix,
                                    int maxOffsetSamples) const;

  ResidualComputation computeResidual(const AudioBuffer& stemSum,
                                      const AudioBuffer& originalMix,
                                      int maxOffsetSamples) const;

  AudioBuffer applyResidualBlend(const AudioBuffer& stemSum,
                                 const AudioBuffer& residual,
                                 double blendPercent,
                                 double ceilingDbtp) const;
};

} // namespace automix::engine
