#include "dsp/SignalMath.h"

#include <algorithm>
#include <cmath>

namespace automix::dsp {

double dbToLinear(const double db) { return std::pow(10.0, db / 20.0); }

double linearToDb(const double linear) {
  constexpr double minValue = 1.0e-12;
  return 20.0 * std::log10(std::max(linear, minValue));
}

} // namespace automix::dsp
