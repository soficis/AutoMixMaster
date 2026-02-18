#include "dsp/TruePeakDetector.h"

#include <algorithm>
#include <cmath>

namespace automix::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;

double linearToDb(const double linear) {
  constexpr double minValue = 1.0e-12;
  return 20.0 * std::log10(std::max(linear, minValue));
}

} // namespace

TruePeakDetector::TruePeakDetector(const int oversampleFactor, const int tapsPerPhase)
    : oversampleFactor_(std::max(2, oversampleFactor)),
      tapsPerPhase_(std::max(8, tapsPerPhase)) {
  redesign();
}

void TruePeakDetector::configure(const int oversampleFactor, const int tapsPerPhase) {
  oversampleFactor_ = std::max(2, oversampleFactor);
  tapsPerPhase_ = std::max(8, tapsPerPhase);
  redesign();
}

void TruePeakDetector::redesign() {
  const int totalTaps = oversampleFactor_ * tapsPerPhase_;
  prototypeFilter_.assign(static_cast<size_t>(totalTaps), 0.0);

  const double fc = 0.5 / static_cast<double>(oversampleFactor_) * 0.95;
  const double center = 0.5 * static_cast<double>(totalTaps - 1);

  double sum = 0.0;
  for (int i = 0; i < totalTaps; ++i) {
    const double n = static_cast<double>(i) - center;
    const double sinc = std::abs(n) < 1.0e-12 ? 2.0 * fc : std::sin(2.0 * kPi * fc * n) / (kPi * n);
    const double window = 0.54 - 0.46 * std::cos((2.0 * kPi * static_cast<double>(i)) / static_cast<double>(totalTaps - 1));
    const double value = sinc * window;
    prototypeFilter_[static_cast<size_t>(i)] = value;
    sum += value;
  }

  if (std::abs(sum) < 1.0e-12) {
    return;
  }
  for (auto& value : prototypeFilter_) {
    value /= sum;
  }
}

double TruePeakDetector::computeTruePeakLinear(const engine::AudioBuffer& buffer) const {
  if (buffer.getNumChannels() == 0 || buffer.getNumSamples() == 0) {
    return 0.0;
  }

  double peak = 0.0;

  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    const float* channel = buffer.getReadPointer(ch);
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
      peak = std::max(peak, static_cast<double>(std::abs(channel[sample])));

      for (int phase = 1; phase < oversampleFactor_; ++phase) {
        double upsampled = 0.0;
        for (int tap = 0; tap < tapsPerPhase_; ++tap) {
          const int inputIndex = sample - tap;
          if (inputIndex < 0) {
            continue;
          }
          const int coeffIndex = phase + tap * oversampleFactor_;
          upsampled += static_cast<double>(channel[inputIndex]) * prototypeFilter_[static_cast<size_t>(coeffIndex)];
        }
        peak = std::max(peak, std::abs(upsampled));
      }
    }
  }

  return peak;
}

double TruePeakDetector::computeTruePeakDbtp(const engine::AudioBuffer& buffer) const {
  return linearToDb(computeTruePeakLinear(buffer));
}

} // namespace automix::dsp
