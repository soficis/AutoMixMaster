#pragma once

#include "engine/AudioBuffer.h"

namespace automix::dsp {

class DeEsser {
 public:
  void process(engine::AudioBuffer& buffer, double strength) const;
};

} // namespace automix::dsp
