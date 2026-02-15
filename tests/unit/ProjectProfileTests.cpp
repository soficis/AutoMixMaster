#include <chrono>
#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "domain/ProjectProfile.h"

TEST_CASE("Project profile defaults are available", "[profile]") {
  const auto defaults = automix::domain::defaultProjectProfiles();
  REQUIRE_FALSE(defaults.empty());

  const auto foundDefault = automix::domain::findProjectProfile(defaults, "default");
  REQUIRE(foundDefault.has_value());
  REQUIRE(foundDefault->rendererName == "BuiltIn");
}

TEST_CASE("Project profile loader merges asset profiles with defaults", "[profile]") {
  const auto nonce = std::to_string(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count());
  const auto root = std::filesystem::temp_directory_path() / ("automix_profile_loader_" + nonce);
  const auto profileDir = root / "assets" / "profiles";
  std::filesystem::create_directories(profileDir);

  nlohmann::json fileProfiles = nlohmann::json::array();
  fileProfiles.push_back({
      {"id", "aggressive_test"},
      {"name", "Aggressive Test"},
      {"platformPreset", "youtube"},
      {"rendererName", "PhaseLimiter"},
      {"outputFormat", "mp3"},
      {"lossyBitrateKbps", 999},
      {"mp3UseVbr", true},
      {"mp3VbrQuality", 99},
      {"gpuProvider", "cuda"},
      {"roleModelPackId", "demo-role-v1"},
      {"mixModelPackId", "demo-mix-v1"},
      {"masterModelPackId", "demo-master-v1"},
      {"safetyPolicyId", "strict"},
      {"preferredStemCount", 12},
      {"pinnedRendererIds", nlohmann::json::array({"BuiltIn", "PhaseLimiter"})},
  });

  {
    std::ofstream out(profileDir / "project_profiles.json");
    out << fileProfiles.dump(2);
  }

  const auto profiles = automix::domain::loadProjectProfiles(root);
  const auto custom = automix::domain::findProjectProfile(profiles, "aggressive_test");
  REQUIRE(custom.has_value());
  REQUIRE(custom->rendererName == "PhaseLimiter");
  REQUIRE(custom->lossyBitrateKbps == 320);
  REQUIRE(custom->mp3UseVbr == true);
  REQUIRE(custom->mp3VbrQuality == 9);
  REQUIRE(custom->preferredStemCount == 6);
  REQUIRE(custom->pinnedRendererIds.size() == 2);

  const auto builtIn = automix::domain::findProjectProfile(profiles, "default");
  REQUIRE(builtIn.has_value());

  std::filesystem::remove_all(root);
}
