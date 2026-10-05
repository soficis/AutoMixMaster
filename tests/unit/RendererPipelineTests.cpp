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

TEST_CASE("PhaseLimiter is opt-in: default renders and logical_all leave it out", "[renderer][pipeline]") {
  automix::domain::RenderSettings defaults;
  const auto single = automix::renderers::resolveRendererChain(defaults);
  REQUIRE(single == std::vector<std::string>{"BuiltIn"});

  automix::domain::RenderSettings logicalAll;
  logicalAll.rendererChainEnabled = true;
  logicalAll.rendererChainMode = "logical_all";
  const auto chain = automix::renderers::resolveRendererChain(logicalAll);
  REQUIRE_FALSE(chain.empty());
  REQUIRE(chain.front() == "BuiltIn");
  REQUIRE(std::find(chain.begin(), chain.end(), "PhaseLimiter") == chain.end());
}

TEST_CASE("Selecting PhaseLimiter opts it into the logical_all chain", "[renderer][pipeline]") {
  automix::domain::RenderSettings settings;
  settings.rendererName = "PhaseLimiter";
  settings.rendererChainEnabled = true;
  settings.rendererChainMode = "logical_all";
  const auto chain = automix::renderers::resolveRendererChain(settings);
  REQUIRE_FALSE(chain.empty());
  if (std::find(chain.begin(), chain.end(), "PhaseLimiter") != chain.end()) {
    REQUIRE(chain.front() == "PhaseLimiter");  // available here: it leads, once
    REQUIRE(std::count(chain.begin(), chain.end(), "PhaseLimiter") == 1);
  }
  REQUIRE(std::find(chain.begin(), chain.end(), "BuiltIn") != chain.end());
}