#pragma once

#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "engine/AudioBuffer.h"

namespace automix::app {

class WaveformPreviewComponent final : public juce::Component {
 public:
  void setBuffer(const engine::AudioBuffer& buffer);
  void setPlayheadProgress(double progress);
  void setZoom(double zoomFactor, double centerProgress);
  void setLoopRange(bool enabled, double loopStartProgress, double loopEndProgress);

  void paint(juce::Graphics& g) override;

 private:
  std::vector<float> waveform_;
  double playheadProgress_ = 0.0;
  double zoomFactor_ = 1.0;
  double zoomCenterProgress_ = 0.5;
  bool loopEnabled_ = false;
  double loopStartProgress_ = 0.0;
  double loopEndProgress_ = 0.0;
};

} // namespace automix::app
