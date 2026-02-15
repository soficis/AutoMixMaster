#include "app/WaveformPreviewComponent.h"

#include <algorithm>
#include <cmath>

namespace automix::app {

void WaveformPreviewComponent::setBuffer(const engine::AudioBuffer& buffer) {
  waveform_.clear();

  const int samples = buffer.getNumSamples();
  if (buffer.getNumChannels() <= 0 || samples <= 0) {
    repaint();
    return;
  }

  const int columns = std::max(512, getWidth() > 0 ? getWidth() * 4 : 2048);
  waveform_.assign(static_cast<size_t>(columns), 0.0f);

  const int blockSamples = std::max(1, samples / columns);
  for (int x = 0; x < columns; ++x) {
    const int start = x * blockSamples;
    const int end = std::min(samples, start + blockSamples);

    float peak = 0.0f;
    for (int i = start; i < end; ++i) {
      float mono = 0.0f;
      for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        mono += buffer.getSample(ch, i);
      }
      mono /= static_cast<float>(buffer.getNumChannels());
      peak = std::max(peak, std::abs(mono));
    }
    waveform_[static_cast<size_t>(x)] = peak;
  }

  repaint();
}

void WaveformPreviewComponent::setPlayheadProgress(const double progress) {
  playheadProgress_ = std::clamp(progress, 0.0, 1.0);
  repaint();
}

void WaveformPreviewComponent::setZoom(const double zoomFactor, const double centerProgress) {
  zoomFactor_ = std::clamp(zoomFactor, 1.0, 64.0);
  zoomCenterProgress_ = std::clamp(centerProgress, 0.0, 1.0);
  repaint();
}

void WaveformPreviewComponent::setLoopRange(const bool enabled,
                                            const double loopStartProgress,
                                            const double loopEndProgress) {
  loopEnabled_ = enabled;
  loopStartProgress_ = std::clamp(loopStartProgress, 0.0, 1.0);
  loopEndProgress_ = std::clamp(loopEndProgress, 0.0, 1.0);
  if (loopEndProgress_ <= loopStartProgress_) {
    loopEnabled_ = false;
  }
  repaint();
}

void WaveformPreviewComponent::paint(juce::Graphics& g) {
  auto area = getLocalBounds();
  g.fillAll(juce::Colour::fromRGB(16, 18, 24));

  if (waveform_.empty()) {
    g.setColour(juce::Colours::white.withAlpha(0.35f));
    g.drawFittedText("Waveform preview", area, juce::Justification::centred, 1);
    return;
  }

  const float centerY = static_cast<float>(area.getCentreY());
  const float halfHeight = static_cast<float>(area.getHeight()) * 0.42f;
  const int width = std::max(1, area.getWidth());

  const double windowFraction = 1.0 / std::max(1.0, zoomFactor_);
  const double maxStart = std::max(0.0, 1.0 - windowFraction);
  const double visibleStart = std::clamp(zoomCenterProgress_ - windowFraction * 0.5, 0.0, maxStart);
  const double visibleEnd = std::min(1.0, visibleStart + windowFraction);

  const int totalColumns = static_cast<int>(waveform_.size());
  const int startIndex = std::clamp(static_cast<int>(std::floor(visibleStart * static_cast<double>(totalColumns - 1))), 0, totalColumns - 1);
  const int endIndex = std::clamp(static_cast<int>(std::ceil(visibleEnd * static_cast<double>(totalColumns - 1))),
                                  startIndex + 1,
                                  totalColumns - 1);
  const int visibleColumns = std::max(1, endIndex - startIndex);

  if (loopEnabled_) {
    const auto toVisibleX = [&](const double progress) {
      if (progress < visibleStart || progress > visibleEnd) {
        return -1;
      }
      const double normalized = (progress - visibleStart) / std::max(1.0e-9, (visibleEnd - visibleStart));
      return static_cast<int>(std::round(normalized * static_cast<double>(width - 1)));
    };

    const int loopStartX = toVisibleX(loopStartProgress_);
    const int loopEndX = toVisibleX(loopEndProgress_);
    if (loopStartX >= 0 && loopEndX >= 0 && loopEndX > loopStartX) {
      g.setColour(juce::Colour::fromRGB(90, 130, 210).withAlpha(0.18f));
      g.fillRect(loopStartX, area.getY(), loopEndX - loopStartX, area.getHeight());
      g.setColour(juce::Colour::fromRGB(140, 180, 255).withAlpha(0.65f));
      g.drawVerticalLine(loopStartX, static_cast<float>(area.getY()), static_cast<float>(area.getBottom()));
      g.drawVerticalLine(loopEndX, static_cast<float>(area.getY()), static_cast<float>(area.getBottom()));
    }
  }

  g.setColour(juce::Colour::fromRGB(65, 180, 255).withAlpha(0.8f));
  for (int x = 0; x < width; ++x) {
    const double interp = static_cast<double>(x) / static_cast<double>(std::max(1, width - 1));
    const int sampleIndex = startIndex + static_cast<int>(std::round(interp * static_cast<double>(visibleColumns)));
    const int clampedIndex = std::clamp(sampleIndex, 0, totalColumns - 1);
    const float value = std::clamp(waveform_[static_cast<size_t>(clampedIndex)], 0.0f, 1.0f);
    const float h = value * halfHeight;
    g.drawVerticalLine(area.getX() + x, centerY - h, centerY + h);
  }

  int playheadX = 0;
  if (playheadProgress_ <= visibleStart) {
    playheadX = area.getX();
  } else if (playheadProgress_ >= visibleEnd) {
    playheadX = area.getRight() - 1;
  } else {
    const double normalized = (playheadProgress_ - visibleStart) / std::max(1.0e-9, (visibleEnd - visibleStart));
    playheadX = area.getX() + static_cast<int>(std::round(normalized * static_cast<double>(width - 1)));
  }

  g.setColour(juce::Colour::fromRGB(255, 190, 64));
  g.drawLine(static_cast<float>(playheadX), static_cast<float>(area.getY()), static_cast<float>(playheadX),
             static_cast<float>(area.getBottom()), 2.0f);

  g.setColour(juce::Colours::white.withAlpha(0.2f));
  g.drawRect(area);
}

} // namespace automix::app
