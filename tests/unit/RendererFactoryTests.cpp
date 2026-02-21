#include <catch2/catch_test_macros.hpp>

#include "renderers/RendererFactory.h"

TEST_CASE("Renderer factory routes bundled CLI renderer ids", "[renderer][factory]") {
  {
    auto renderer = automix::renderers::createRenderer("FFmpeg");
    REQUIRE(renderer != nullptr);
  }
  {
    auto renderer = automix::renderers::createRenderer("SoX");
    REQUIRE(renderer != nullptr);
  }
  {
    auto renderer = automix::renderers::createRenderer("rsgain");
    REQUIRE(renderer != nullptr);
  }
}
