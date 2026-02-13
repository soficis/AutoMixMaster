#include <catch2/catch_test_macros.hpp>

#include "engine/AudioBuffer.h"
#include "engine/AudioPreviewEngine.h"

namespace {

automix::engine::AudioBuffer makeConstantBuffer(const float value) {
  automix::engine::AudioBuffer buffer(2, 512, 48000.0);
  for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      buffer.setSample(ch, i, value);
    }
  }
  return buffer;
}

} // namespace

TEST_CASE("AudioPreviewEngine crossfades between A/B sources", "[preview]") {
  automix::engine::AudioPreviewEngine engine;
  const auto original = makeConstantBuffer(0.1f);
  const auto rendered = makeConstantBuffer(0.9f);
  engine.setBuffers(original, rendered);

  engine.setSource(automix::engine::PreviewSource::OriginalMix);
  auto a = engine.buildCrossfadedPreview(64);
  REQUIRE(a.getSample(0, 0) == 0.1f);

  engine.setSource(automix::engine::PreviewSource::RenderedMix);
  auto b = engine.buildCrossfadedPreview(64);
  REQUIRE(b.getSample(0, 0) == 0.1f);
  REQUIRE(b.getSample(0, 63) > 0.5f);
}
