#pragma once

#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "app/style/Theme.h"
#include "engine/AudioBuffer.h"

namespace automix::app {

/// Full-width stereo waveform display with playhead, zoom, seek, and loop regions.
class HeroWaveform final : public juce::Component {
public:
  HeroWaveform();

  void paint(juce::Graphics& g) override;
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  void resized() override;

  void setBuffer(const engine::AudioBuffer& buffer);
  void setPlayheadProgress(double progress);
  void setZoom(double zoomFactor, double centerProgress);
  void setLoopRange(bool enabled, double startProgress, double endProgress);

  // Callbacks
  std::function<void(double)> onSeek; // progress fraction 0..1

private:
  double progressFromX(int x) const;
  void buildWaveformCache();
  void buildMipLevels();

  // Mip-level cached peak data (built once on setBuffer)
  static constexpr int kMipLevels = 3;
  static constexpr int kMipFactors[kMipLevels] = {1, 4, 16};
  std::vector<float> mipPeaks_[kMipLevels];
  int mipSampleCounts_[kMipLevels] = {0, 0, 0};

  std::vector<float> waveformPeaks_;
  int cachedWidth_ = 0;
  double cachedZoomFactor_ = 0.0;
  double cachedZoomCenter_ = 0.0;
  double playheadProgress_ = 0.0;
  double zoomFactor_ = 1.0;
  double zoomCenter_ = 0.5;
  bool loopEnabled_ = false;
  double loopStart_ = 0.0;
  double loopEnd_ = 0.0;

  // Source buffer data (downsampled for display)
  std::vector<float> rawSamples_;
  int rawSampleCount_ = 0;
  int rawChannelCount_ = 0;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HeroWaveform)
};

} // namespace automix::app
