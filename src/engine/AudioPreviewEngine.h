#pragma once

#include "engine/AudioBuffer.h"

namespace automix::engine {

enum class PreviewSource {
  OriginalMix = 0,
  RenderedMix = 1,
};

class AudioPreviewEngine {
 public:
  void setBuffers(const AudioBuffer& originalMix, const AudioBuffer& renderedMix);
  void setSource(PreviewSource source);
  PreviewSource source() const;

  void play();
  void stop();
  bool isPlaying() const;

  AudioBuffer buildCrossfadedPreview(int crossfadeSamples) const;

 private:
  AudioBuffer originalMix_;
  AudioBuffer renderedMix_;
  PreviewSource source_ = PreviewSource::OriginalMix;
  PreviewSource previousSource_ = PreviewSource::OriginalMix;
  bool playing_ = false;
};

} // namespace automix::engine
