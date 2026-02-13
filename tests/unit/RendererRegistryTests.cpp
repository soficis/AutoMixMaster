#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "renderers/RendererRegistry.h"

TEST_CASE("Renderer registry always includes built-in renderer", "[renderer][registry]") {
  automix::renderers::RendererRegistry registry;
  const auto infos = registry.list();

  bool foundBuiltIn = false;
  for (const auto& info : infos) {
    if (info.id == "BuiltIn") {
      foundBuiltIn = true;
      REQUIRE(info.available);
      REQUIRE(info.linkMode == automix::renderers::RendererLinkMode::InProcess);
      break;
    }
  }
  REQUIRE(foundBuiltIn);
}

TEST_CASE("Renderer registry includes configured user external renderer metadata", "[renderer][registry]") {
  const std::filesystem::path tempBinary = std::filesystem::temp_directory_path() / "automix_external_renderer_dummy.bin";
  {
    std::ofstream out(tempBinary);
    out << "dummy";
  }

  automix::renderers::ExternalRendererConfig config;
  config.id = "ExternalUser1";
  config.name = "Dummy Tool";
  config.licenseId = "GPL-3.0-only";
  config.binaryPath = tempBinary;

  automix::renderers::RendererRegistry registry;
  const auto infos = registry.list({config});

  bool foundExternal = false;
  for (const auto& info : infos) {
    if (info.id == "ExternalUser1") {
      foundExternal = true;
      REQUIRE(info.linkMode == automix::renderers::RendererLinkMode::External);
      REQUIRE(info.available);
      REQUIRE(info.binaryPath == tempBinary);
      break;
    }
  }

  REQUIRE(foundExternal);
  std::filesystem::remove(tempBinary);
}

TEST_CASE("Renderer registry scans asset descriptors", "[renderer][registry]") {
  automix::renderers::RendererRegistry registry;
  const auto infos = registry.list();

  bool foundTemplate = false;
  for (const auto& info : infos) {
    if (info.id == "ExternalTemplate") {
      foundTemplate = true;
      REQUIRE(info.linkMode == automix::renderers::RendererLinkMode::External);
      break;
    }
  }
  REQUIRE(foundTemplate);
}
