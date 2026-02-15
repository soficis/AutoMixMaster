#pragma once

#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "engine/AudioBuffer.h"

namespace automix::app {

class WaveformPreviewComponent final : public juce::Component {
 public:
  void setBuffer(const engine::AudioBuffer& buffer);
  void setPlayheadProgress(double progress);

  void paint(juce::Graphics& g) override;

 private:
  std::vector<float> waveform_;
  double playheadProgress_ = 0.0;
};

} // namespace automix::app
