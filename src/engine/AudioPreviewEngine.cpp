#include "engine/AudioPreviewEngine.h"

#include <algorithm>

namespace automix::engine {

void AudioPreviewEngine::setBuffers(const AudioBuffer& originalMix, const AudioBuffer& renderedMix) {
  originalMix_ = originalMix;
  renderedMix_ = renderedMix;
}

void AudioPreviewEngine::setSource(const PreviewSource source) {
  if (source_ == source) {
    return;
  }
  previousSource_ = source_;
  source_ = source;
}

PreviewSource AudioPreviewEngine::source() const { return source_; }

void AudioPreviewEngine::play() { playing_ = true; }

void AudioPreviewEngine::stop() { playing_ = false; }

bool AudioPreviewEngine::isPlaying() const { return playing_; }

AudioBuffer AudioPreviewEngine::buildCrossfadedPreview(const int crossfadeSamples) const {
  const AudioBuffer* target = source_ == PreviewSource::RenderedMix ? &renderedMix_ : &originalMix_;
  const AudioBuffer* previous = previousSource_ == PreviewSource::RenderedMix ? &renderedMix_ : &originalMix_;

  if (target->getNumChannels() == 0 || target->getNumSamples() == 0) {
    return *target;
  }

  AudioBuffer output = *target;
  const int fadeSamples = std::clamp(crossfadeSamples, 0, output.getNumSamples());
  if (fadeSamples <= 0 || previous->getNumSamples() == 0) {
    return output;
  }

  const int commonChannels = std::min(output.getNumChannels(), previous->getNumChannels());
  const int commonSamples = std::min(output.getNumSamples(), previous->getNumSamples());

  for (int ch = 0; ch < commonChannels; ++ch) {
    for (int i = 0; i < std::min(fadeSamples, commonSamples); ++i) {
      const float a = previous->getSample(ch, i);
      const float b = output.getSample(ch, i);
      const float t = static_cast<float>(i) / static_cast<float>(std::max(1, fadeSamples - 1));
      output.setSample(ch, i, a * (1.0f - t) + b * t);
    }
  }

  return output;
}

} // namespace automix::engine
