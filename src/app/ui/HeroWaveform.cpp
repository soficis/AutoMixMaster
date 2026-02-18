#include "app/ui/HeroWaveform.h"

#include <algorithm>
#include <cmath>

namespace automix::app {

using namespace theme;

HeroWaveform::HeroWaveform() {
  setOpaque(true);
}

void HeroWaveform::setBuffer(const engine::AudioBuffer& buffer) {
  rawChannelCount_ = buffer.getNumChannels();
  rawSampleCount_ = buffer.getNumSamples();

  // Store mono-mixed absolute peaks for rendering
  rawSamples_.resize(static_cast<size_t>(rawSampleCount_));
  if (rawSampleCount_ > 0 && rawChannelCount_ > 0) {
    for (int i = 0; i < rawSampleCount_; ++i) {
      float sum = 0.0f;
      for (int ch = 0; ch < rawChannelCount_; ++ch) {
        sum += std::abs(buffer.getSample(ch, i));
      }
      rawSamples_[static_cast<size_t>(i)] = sum / static_cast<float>(rawChannelCount_);
    }
  }

  buildMipLevels();
  cachedWidth_ = 0; // invalidate display cache
  cachedZoomFactor_ = 0.0;
  cachedZoomCenter_ = 0.0;
  repaint();
}

void HeroWaveform::setPlayheadProgress(double progress) {
  playheadProgress_ = progress;
  repaint();
}

void HeroWaveform::setZoom(double zoomFactor, double centerProgress) {
  zoomFactor_ = std::max(1.0, zoomFactor);
  zoomCenter_ = centerProgress;
  cachedWidth_ = 0; // Force cache rebuild on next paint
  repaint();
}

void HeroWaveform::setLoopRange(bool enabled, double startProgress, double endProgress) {
  loopEnabled_ = enabled;
  loopStart_ = startProgress;
  loopEnd_ = endProgress;
  repaint();
}

double HeroWaveform::progressFromX(int x) const {
  if (getWidth() <= 0)
    return 0.0;

  double visibleWidth = 1.0 / zoomFactor_;
  double viewStart = std::clamp(zoomCenter_ - visibleWidth * 0.5, 0.0, 1.0 - visibleWidth);
  return viewStart + (static_cast<double>(x) / static_cast<double>(getWidth())) * visibleWidth;
}

void HeroWaveform::mouseDown(const juce::MouseEvent& event) {
  double progress = std::clamp(progressFromX(event.x), 0.0, 1.0);
  if (onSeek)
    onSeek(progress);
}

void HeroWaveform::mouseDrag(const juce::MouseEvent& event) {
  double progress = std::clamp(progressFromX(event.x), 0.0, 1.0);
  if (onSeek)
    onSeek(progress);
}

void HeroWaveform::resized() {
  cachedWidth_ = 0;
}

void HeroWaveform::buildMipLevels() {
  for (int level = 0; level < kMipLevels; ++level) {
    int factor = kMipFactors[level];
    int mipCount = (rawSampleCount_ + factor - 1) / factor;
    mipSampleCounts_[level] = mipCount;
    mipPeaks_[level].resize(static_cast<size_t>(mipCount));

    for (int i = 0; i < mipCount; ++i) {
      int start = i * factor;
      int end = std::min(start + factor, rawSampleCount_);
      float peak = 0.0f;
      for (int s = start; s < end; ++s) {
        peak = std::max(peak, rawSamples_[static_cast<size_t>(s)]);
      }
      mipPeaks_[level][static_cast<size_t>(i)] = peak;
    }
  }
}

void HeroWaveform::buildWaveformCache() {
  int w = getWidth();
  if (w <= 0 || rawSampleCount_ == 0) {
    waveformPeaks_.clear();
    cachedWidth_ = w;
    cachedZoomFactor_ = zoomFactor_;
    cachedZoomCenter_ = zoomCenter_;
    return;
  }

  double visibleWidth = 1.0 / zoomFactor_;
  double viewStart = std::clamp(zoomCenter_ - visibleWidth * 0.5, 0.0, 1.0 - visibleWidth);

  // Choose the coarsest mip level where each pixel spans at least 1 mip sample
  double samplesPerPixel = (visibleWidth * rawSampleCount_) / static_cast<double>(w);
  int bestLevel = 0;
  for (int level = kMipLevels - 1; level >= 0; --level) {
    double mipSamplesPerPixel = samplesPerPixel / static_cast<double>(kMipFactors[level]);
    if (mipSamplesPerPixel >= 1.0) {
      bestLevel = level;
      break;
    }
  }

  int factor = kMipFactors[bestLevel];
  int mipCount = mipSampleCounts_[bestLevel];
  const auto& mipData = mipPeaks_[bestLevel];

  waveformPeaks_.resize(static_cast<size_t>(w));

  for (int px = 0; px < w; ++px) {
    double fracStart = viewStart + (static_cast<double>(px) / static_cast<double>(w)) * visibleWidth;
    double fracEnd = viewStart + (static_cast<double>(px + 1) / static_cast<double>(w)) * visibleWidth;

    int mipStart = static_cast<int>(fracStart * rawSampleCount_) / factor;
    int mipEnd = static_cast<int>(fracEnd * rawSampleCount_) / factor;
    mipStart = std::clamp(mipStart, 0, mipCount - 1);
    mipEnd = std::clamp(mipEnd, mipStart + 1, mipCount);

    float peak = 0.0f;
    for (int s = mipStart; s < mipEnd; ++s) {
      peak = std::max(peak, mipData[static_cast<size_t>(s)]);
    }
    waveformPeaks_[static_cast<size_t>(px)] = peak;
  }

  cachedWidth_ = w;
  cachedZoomFactor_ = zoomFactor_;
  cachedZoomCenter_ = zoomCenter_;
}

void HeroWaveform::paint(juce::Graphics& g) {
  auto bounds = getLocalBounds().toFloat();
  g.fillAll(colour(colours::background));

  if (cachedWidth_ != getWidth() || cachedZoomFactor_ != zoomFactor_ || cachedZoomCenter_ != zoomCenter_) {
    buildWaveformCache();
  }

  int w = getWidth();
  float h = bounds.getHeight();
  float midY = h * 0.5f;

  if (waveformPeaks_.empty()) {
    // No waveform data — show placeholder
    g.setColour(colour(colours::textMuted));
    g.setFont(typography::body());
    g.drawText("Drop audio files here or import stems", bounds, juce::Justification::centred);
    return;
  }

  // Draw waveform fill
  juce::Path waveformPath;
  waveformPath.startNewSubPath(0.0f, midY);

  for (int px = 0; px < w; ++px) {
    float peak = waveformPeaks_[static_cast<size_t>(px)];
    float amplitude = peak * midY * 0.9f;
    waveformPath.lineTo(static_cast<float>(px), midY - amplitude);
  }
  for (int px = w - 1; px >= 0; --px) {
    float peak = waveformPeaks_[static_cast<size_t>(px)];
    float amplitude = peak * midY * 0.9f;
    waveformPath.lineTo(static_cast<float>(px), midY + amplitude);
  }
  waveformPath.closeSubPath();

  juce::ColourGradient gradient(colour(colours::waveformFill), 0.0f, 0.0f,
                                colour(colours::waveformFill).withAlpha(0.4f), 0.0f, h, false);
  g.setGradientFill(gradient);
  g.fillPath(waveformPath);

  g.setColour(colour(colours::waveformOutline));
  // Draw top outline
  juce::Path outlinePath;
  outlinePath.startNewSubPath(0.0f, midY);
  for (int px = 0; px < w; ++px) {
    float peak = waveformPeaks_[static_cast<size_t>(px)];
    float amplitude = peak * midY * 0.9f;
    outlinePath.lineTo(static_cast<float>(px), midY - amplitude);
  }
  g.strokePath(outlinePath, juce::PathStrokeType(1.0f));

  // Loop region overlay
  if (loopEnabled_ && loopEnd_ > loopStart_) {
    double visibleWidth = 1.0 / zoomFactor_;
    double viewStart = std::clamp(zoomCenter_ - visibleWidth * 0.5, 0.0, 1.0 - visibleWidth);

    float loopStartX = static_cast<float>((loopStart_ - viewStart) / visibleWidth * w);
    float loopEndX = static_cast<float>((loopEnd_ - viewStart) / visibleWidth * w);

    g.setColour(colour(colours::selectionFill));
    g.fillRect(loopStartX, 0.0f, loopEndX - loopStartX, h);

    g.setColour(colour(colours::primary).withAlpha(0.8f));
    g.fillRect(loopStartX, 0.0f, 2.0f, h);
    g.fillRect(loopEndX - 2.0f, 0.0f, 2.0f, h);
  }

  // Playhead
  if (playheadProgress_ >= 0.0) {
    double visibleWidth = 1.0 / zoomFactor_;
    double viewStart = std::clamp(zoomCenter_ - visibleWidth * 0.5, 0.0, 1.0 - visibleWidth);
    float playheadX = static_cast<float>((playheadProgress_ - viewStart) / visibleWidth * w);

    if (playheadX >= 0.0f && playheadX <= static_cast<float>(w)) {
      g.setColour(colour(colours::playhead));
      g.fillRect(playheadX - 1.0f, 0.0f, 2.0f, h);
    }
  }

  // Center line
  g.setColour(colour(colours::surfaceBorder).withAlpha(0.3f));
  g.fillRect(0.0f, midY - 0.5f, static_cast<float>(w), 1.0f);
}

} // namespace automix::app
