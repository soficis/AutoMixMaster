#pragma once

#include "engine/AudioBuffer.h"

namespace automix::engine {

class AudioResampler {
 public:
  AudioBuffer resampleLinear(const AudioBuffer& input, double targetSampleRate) const;
};

} // namespace automix::engine
