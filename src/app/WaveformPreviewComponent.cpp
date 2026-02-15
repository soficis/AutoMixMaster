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

  const int columns = std::max(128, getWidth() > 0 ? getWidth() : 512);
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

  g.setColour(juce::Colour::fromRGB(65, 180, 255).withAlpha(0.8f));
  const int columns = static_cast<int>(waveform_.size());
  for (int x = 0; x < columns; ++x) {
    const float value = std::clamp(waveform_[static_cast<size_t>(x)], 0.0f, 1.0f);
    const float h = value * halfHeight;
    g.drawVerticalLine(x, centerY - h, centerY + h);
  }

  const int playheadX = static_cast<int>(std::round(playheadProgress_ * static_cast<double>(columns - 1)));
  g.setColour(juce::Colour::fromRGB(255, 190, 64));
  g.drawLine(static_cast<float>(playheadX), static_cast<float>(area.getY()), static_cast<float>(playheadX),
             static_cast<float>(area.getBottom()), 2.0f);

  g.setColour(juce::Colours::white.withAlpha(0.2f));
  g.drawRect(area);
}

} // namespace automix::app
