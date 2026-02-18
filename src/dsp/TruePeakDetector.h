#pragma once

#include <vector>

#include "engine/AudioBuffer.h"

namespace automix::dsp {

class TruePeakDetector {
 public:
  explicit TruePeakDetector(int oversampleFactor = 4, int tapsPerPhase = 16);

  void configure(int oversampleFactor, int tapsPerPhase = 16);
  double computeTruePeakLinear(const engine::AudioBuffer& buffer) const;
  double computeTruePeakDbtp(const engine::AudioBuffer& buffer) const;

 private:
  void redesign();

  int oversampleFactor_ = 4;
  int tapsPerPhase_ = 16;
  std::vector<double> prototypeFilter_;
};

} // namespace automix::dsp
