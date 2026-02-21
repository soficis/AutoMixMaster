#include <catch2/catch_test_macros.hpp>

#include "renderers/PostRendererFactory.h"

TEST_CASE("Post renderer factory only creates supported post-renderers", "[renderer][post-renderer][factory]") {
  {
    auto renderer = automix::renderers::createPostRenderer("rsgain");
    REQUIRE(renderer != nullptr);
  }
  {
    auto renderer = automix::renderers::createPostRenderer("BuiltIn");
    REQUIRE(renderer == nullptr);
  }
}
