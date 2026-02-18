#pragma once

#include "engine/AudioBuffer.h"

namespace automix::dsp {

class SoftClipper {
 public:
  void process(engine::AudioBuffer& buffer, double drive) const;
};

} // namespace automix::dsp
