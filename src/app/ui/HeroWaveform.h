#pragma once

#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_opengl/juce_opengl.h>

#include "app/style/Theme.h"
#include "engine/AudioBuffer.h"

namespace automix::app {

/// Full-width stereo waveform display with playhead, zoom, seek, loop regions, file drop target,
/// stem colour overlays, zoom controls, and OpenGL-accelerated rendering.
class HeroWaveform final : public juce::Component,
                           public juce::FileDragAndDropTarget,
                           private juce::Timer {
public:
  HeroWaveform();
  ~HeroWaveform() override;

  void paint(juce::Graphics& g) override;
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  void mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) override;
  void resized() override;

  // FileDragAndDropTarget
  bool isInterestedInFileDrag(const juce::StringArray& files) override;
  void fileDragEnter(const juce::StringArray& files, int x, int y) override;
  void fileDragExit(const juce::StringArray& files) override;
  void filesDropped(const juce::StringArray& files, int x, int y) override;

  void setBuffer(const engine::AudioBuffer& buffer);
  void setPlayheadProgress(double progress);
  void setZoom(double zoomFactor, double centerProgress);
  void setLoopRange(bool enabled, double startProgress, double endProgress);
  struct StemGroup { std::vector<float> peaks; juce::Colour colour; juce::String name; };
  void setStemGroups(const std::vector<StemGroup>& groups);

  // Callbacks
  std::function<void(double)> onSeek;                        // progress fraction 0..1
  std::function<void(double)> onZoomChanged;                 // new zoom factor
  std::function<void(std::vector<juce::File>)> onFilesDropped; // audio/preset files dropped
  std::function<void(juce::File)> onPresetDropped;           // JSON preset file dropped

private:
  double progressFromX(int x) const;
  void buildWaveformCache();
  void buildMipLevels();
  void timerCallback() override;

  void drawZoomControls(juce::Graphics& g);
  void drawStemOverlay(juce::Graphics& g, const juce::Rectangle<float>& bounds, float midY, int w, float h);

  // OpenGL context for accelerated rendering
  juce::OpenGLContext openGLContext_;

  // Zoom control state
  bool zoomInHover_ = false;
  bool zoomOutHover_ = false;
  bool zoomResetHover_ = false;

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

  // Per-channel stem data for colour overlays
  std::vector<float> stemPeaks_[4]; // up to 4 stem groups
  juce::Colour stemColours_[4];
  int numStemGroups_ = 0;

  bool isDragOver_ = false;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HeroWaveform)
};

} // namespace automix::app
