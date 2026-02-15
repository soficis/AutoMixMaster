#pragma once

#include "domain/MasterPlan.h"
#include "engine/AudioBuffer.h"

namespace automix::dsp {

class MultibandProcessor final {
 public:
  void process(engine::AudioBuffer& buffer, const domain::MultibandSettings& settings) const;
};

} // namespace automix::dsp
