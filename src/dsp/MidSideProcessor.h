#pragma once

#include "engine/AudioBuffer.h"

namespace automix::dsp {

class MidSideProcessor {
 public:
  void process(engine::AudioBuffer& buffer, double monoBelowHz, double width) const;
};

} // namespace automix::dsp
