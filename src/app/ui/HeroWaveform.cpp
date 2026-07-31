#include "app/ui/HeroWaveform.h"

#include <algorithm>
#include <cmath>

#include <juce_opengl/juce_opengl.h>

namespace automix::app {

using namespace theme;

HeroWaveform::HeroWaveform() {
  setOpaque(true);
  openGLContext_.setComponentPaintingEnabled(true);
  openGLContext_.attachTo(*this);
  startTimerHz(30);

  // Default stem colours for up to 4 groups
  stemColours_[0] = juce::Colour(0xFF4361EE); // blue
  stemColours_[1] = juce::Colour(0xFF06D6A0); // green
  stemColours_[2] = juce::Colour(0xFFFFD166); // yellow
  stemColours_[3] = juce::Colour(0xFFE63946); // red
}

HeroWaveform::~HeroWaveform() {
  stopTimer();
  openGLContext_.detach();
}

void HeroWaveform::timerCallback() {
  // Periodic repaint for playhead animation during playback
  if (playheadProgress_ >= 0.0)
    repaint();
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
  cachedWidth_ = 0;
  cachedZoomFactor_ = 0.0;
  cachedZoomCenter_ = 0.0;
  numStemGroups_ = 0;
  repaint();
}

void HeroWaveform::setStemGroups(const std::vector<StemGroup>& groups) {
  numStemGroups_ = std::min(static_cast<int>(groups.size()), 4);
  for (int i = 0; i < numStemGroups_; ++i) {
    stemPeaks_[i] = groups[static_cast<size_t>(i)].peaks;
    stemColours_[i] = groups[static_cast<size_t>(i)].colour;
  }
  repaint();
}

void HeroWaveform::setPlayheadProgress(double progress) {
  playheadProgress_ = progress;
  repaint();
}

void HeroWaveform::setZoom(double zoomFactor, double centerProgress) {
  zoomFactor_ = std::max(1.0, zoomFactor);
  zoomCenter_ = centerProgress;
  cachedWidth_ = 0;
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
  // Check if click is on zoom controls (top-right corner)
  auto bounds = getLocalBounds();
  auto zoomArea = bounds.removeFromTop(28).removeFromRight(140);
  if (zoomArea.contains(event.getPosition())) {
    if (event.x >= bounds.getRight() - 130 && event.x < bounds.getRight() - 90) {
      setZoom(zoomFactor_ * 2.0, zoomCenter_);
      if (onZoomChanged) onZoomChanged(zoomFactor_);
    } else if (event.x >= bounds.getRight() - 90 && event.x < bounds.getRight() - 46) {
      setZoom(std::max(1.0, zoomFactor_ * 0.5), zoomCenter_);
      if (onZoomChanged) onZoomChanged(zoomFactor_);
    } else if (event.x >= bounds.getRight() - 46) {
      setZoom(1.0, 0.5);
      if (onZoomChanged) onZoomChanged(zoomFactor_);
    }
    return;
  }

  double progress = std::clamp(progressFromX(event.x), 0.0, 1.0);
  if (onSeek)
    onSeek(progress);
}

void HeroWaveform::mouseDrag(const juce::MouseEvent& event) {
  double progress = std::clamp(progressFromX(event.x), 0.0, 1.0);
  if (onSeek)
    onSeek(progress);
}

void HeroWaveform::mouseWheelMove(const juce::MouseEvent& event, const juce::MouseWheelDetails& wheel) {
  const double zoomSensitivity = 0.3;
  double zoomDelta = -wheel.deltaY * zoomSensitivity;
  double newZoom = std::max(1.0, zoomFactor_ * (1.0 + zoomDelta));

  double clickProgress = progressFromX(event.x);
  setZoom(newZoom, clickProgress);
  if (onZoomChanged) onZoomChanged(zoomFactor_);
}

void HeroWaveform::resized() {
  cachedWidth_ = 0;
}

// ── FileDragAndDropTarget ─────────────────────────────────────────

static bool isAudioExtension(const juce::String& path) {
  auto ext = juce::File(path).getFileExtension().toLowerCase();
  return ext == ".wav" || ext == ".aiff" || ext == ".aif" ||
         ext == ".flac"|| ext == ".mp3"  || ext == ".ogg";
}

bool HeroWaveform::isInterestedInFileDrag(const juce::StringArray& files) {
  for (const auto& path : files) {
    const auto ext = juce::File(path).getFileExtension().toLowerCase();
    if (isAudioExtension(path) || ext == ".json")
      return true;
  }
  return false;
}

void HeroWaveform::fileDragEnter(const juce::StringArray& /*files*/, int /*x*/, int /*y*/) {
  isDragOver_ = true;
  repaint();
}

void HeroWaveform::fileDragExit(const juce::StringArray& /*files*/) {
  isDragOver_ = false;
  repaint();
}

void HeroWaveform::filesDropped(const juce::StringArray& files, int /*x*/, int /*y*/) {
  isDragOver_ = false;
  repaint();

  std::vector<juce::File> audioFiles;
  for (const auto& path : files) {
    const auto f = juce::File(path);
    const auto ext = f.getFileExtension().toLowerCase();
    if (isAudioExtension(path)) {
      audioFiles.push_back(f);
    } else if (ext == ".json" && onPresetDropped) {
      onPresetDropped(f);
    }
  }

  if (!audioFiles.empty() && onFilesDropped)
    onFilesDropped(std::move(audioFiles));
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

void HeroWaveform::drawZoomControls(juce::Graphics& g) {
  const auto bounds = getLocalBounds().toFloat();
  const float btnSize = 28.0f;
  const float gap = 4.0f;
  const float xStart = bounds.getRight() - 140.0f;

  // Background pill for zoom controls
  g.setColour(juce::Colours::black.withAlpha(0.35f));
  g.fillRoundedRectangle(xStart, 4.0f, 136.0f, btnSize, btnSize * 0.5f);

  g.setFont(typography::caption());
  auto drawBtn = [&](float x, const juce::String& label, bool hovered) {
    auto r = juce::Rectangle<float>(x, 4.0f, btnSize, btnSize);
    if (hovered) {
      g.setColour(juce::Colours::white.withAlpha(0.25f));
      g.fillRoundedRectangle(r, 4.0f);
    }
    g.setColour(juce::Colours::white);
    g.drawText(label, r, juce::Justification::centred);
  };

  drawBtn(xStart + gap, "+", zoomInHover_);
  drawBtn(xStart + btnSize + gap * 2, "\u2212", zoomOutHover_);
  drawBtn(xStart + (btnSize + gap) * 2 + gap, "R", zoomResetHover_);

  // Zoom level text
  auto zoomLabel = juce::Rectangle<float>(xStart + (btnSize + gap) * 3 + gap * 2, 4.0f, 40.0f, btnSize);
  g.setColour(juce::Colours::white.withAlpha(0.7f));
  g.drawText(juce::String(static_cast<int>(zoomFactor_)) + "x", zoomLabel, juce::Justification::centredLeft);
}

void HeroWaveform::drawStemOverlay(juce::Graphics& g, const juce::Rectangle<float>& bounds, float midY, int w, float h) {
  if (numStemGroups_ == 0 || waveformPeaks_.empty())
    return;

  double visibleWidth = 1.0 / zoomFactor_;
  double viewStart = std::clamp(zoomCenter_ - visibleWidth * 0.5, 0.0, 1.0 - visibleWidth);

  for (int group = 0; group < numStemGroups_; ++group) {
    const auto& peaks = stemPeaks_[group];
    if (peaks.empty())
      continue;

    int peakCount = static_cast<int>(peaks.size());
    juce::Path stemPath;
    stemPath.startNewSubPath(0.0f, midY);

    for (int px = 0; px < w; ++px) {
      double frac = viewStart + (static_cast<double>(px) / static_cast<double>(w)) * visibleWidth;
      int idx = std::clamp(static_cast<int>(frac * peakCount), 0, peakCount - 1);
      float amplitude = peaks[static_cast<size_t>(idx)] * midY * 0.9f;
      stemPath.lineTo(static_cast<float>(px), midY - amplitude);
    }
    for (int px = w - 1; px >= 0; --px) {
      double frac = viewStart + (static_cast<double>(px) / static_cast<double>(w)) * visibleWidth;
      int idx = std::clamp(static_cast<int>(frac * peakCount), 0, peakCount - 1);
      float amplitude = peaks[static_cast<size_t>(idx)] * midY * 0.9f;
      stemPath.lineTo(static_cast<float>(px), midY + amplitude);
    }
    stemPath.closeSubPath();

    g.setColour(stemColours_[group].withAlpha(0.15f));
    g.fillPath(stemPath);

    // Outline for the stem group
    g.setColour(stemColours_[group].withAlpha(0.5f));
    juce::Path outlinePath;
    outlinePath.startNewSubPath(0.0f, midY);
    for (int px = 0; px < w; ++px) {
      double frac = viewStart + (static_cast<double>(px) / static_cast<double>(w)) * visibleWidth;
      int idx = std::clamp(static_cast<int>(frac * peakCount), 0, peakCount - 1);
      float amplitude = peaks[static_cast<size_t>(idx)] * midY * 0.9f;
      outlinePath.lineTo(static_cast<float>(px), midY - amplitude);
    }
    g.strokePath(outlinePath, juce::PathStrokeType(1.0f));
  }
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
    auto upperHalf = bounds.withHeight(h * 0.5f);
    g.setFont(typography::subhead());
    g.setColour(colour(colours::primary).withAlpha(0.85f));
    g.drawText("Drop stems here  or  click Import (Ctrl+I)", upperHalf, juce::Justification::centredBottom);

    auto lowerHalf = bounds.withY(h * 0.5f).withHeight(h * 0.5f);
    g.setFont(typography::caption());
    g.setColour(colour(colours::textMuted));
    g.drawText("1  Import Stems    ->    2  Mix + Master    ->    3  Export\n"
               "Tip: enable 'AI Stem Separation' in the control deck to split one full mix into stems.",
               lowerHalf.reduced(0.0f, 8.0f), juce::Justification::centredTop);
  } else {
    // Draw stem overlay if we have per-stem data
    if (numStemGroups_ > 0) {
      drawStemOverlay(g, bounds, midY, w, h);
    }

    // Draw main waveform fill
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

    juce::ColourGradient gradient(
      colour(colours::waveformFill), 0.0f, 0.0f,
      colour(colours::waveformFill).withAlpha(0.4f), 0.0f, h, false);
    g.setGradientFill(gradient);
    g.fillPath(waveformPath);

    g.setColour(colour(colours::waveformOutline));
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

    // Draw zoom controls overlay
    drawZoomControls(g);
  }

  if (isDragOver_) {
    g.setColour(colour(colours::primary).withAlpha(0.12f));
    g.fillAll();
    g.setColour(colour(colours::primary));
    g.drawRect(bounds.reduced(3.0f), 2.0f);
    g.setFont(typography::subhead());
    g.setColour(colour(colours::primary));
    g.drawText("Release to import audio files", bounds, juce::Justification::centred);
  }
}

} // namespace automix::app