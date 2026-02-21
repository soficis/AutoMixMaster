#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "domain/RenderSettings.h"
#include "renderers/RendererPipeline.h"

TEST_CASE("Renderer pipeline resolves single renderer when chain is disabled", "[renderer][pipeline]") {
  automix::domain::RenderSettings settings;
  settings.rendererName = "PhaseLimiter";
  settings.rendererChainEnabled = false;

  const auto chain = automix::renderers::resolveRendererChain(settings);
  REQUIRE(chain.size() == 1);
  REQUIRE(chain.front() == "PhaseLimiter");
}

TEST_CASE("Renderer pipeline normalizes master-then-rsgain chain primary", "[renderer][pipeline]") {
  automix::domain::RenderSettings settings;
  settings.rendererName = "rsgain";
  settings.rendererChainEnabled = true;
  settings.rendererChainMode = "master_then_rsgain";

  const auto chain = automix::renderers::resolveRendererChain(settings);
  REQUIRE_FALSE(chain.empty());
  REQUIRE(chain.front() == "BuiltIn");
}

TEST_CASE("Renderer pipeline logical-all chain always includes BuiltIn", "[renderer][pipeline]") {
  automix::domain::RenderSettings settings;
  settings.rendererName = "SoX";
  settings.rendererChainEnabled = true;
  settings.rendererChainMode = "logical_all";

  const auto chain = automix::renderers::resolveRendererChain(settings);
  REQUIRE_FALSE(chain.empty());
  REQUIRE(std::find(chain.begin(), chain.end(), "BuiltIn") != chain.end());
}

TEST_CASE("Renderer pipeline uses explicit chain list when provided", "[renderer][pipeline]") {
  automix::domain::RenderSettings settings;
  settings.rendererName = "BuiltIn";
  settings.rendererChainEnabled = true;
  settings.rendererChain = {"BuiltIn", "BuiltIn", "rsgain"};

  const auto chain = automix::renderers::resolveRendererChain(settings);
  REQUIRE(chain.size() == 2);
  REQUIRE(chain[0] == "BuiltIn");
  REQUIRE(chain[1] == "rsgain");
}
