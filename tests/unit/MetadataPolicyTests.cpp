#include <map>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "util/MetadataPolicy.h"

TEST_CASE("Metadata policy strip removes all tags", "[metadata][policy]") {
  const std::map<std::string, std::string> source = {
      {"title", "Song"},
      {"custom_tag", "value"},
  };

  const auto filtered = automix::util::applyMetadataPolicy(source, "strip");
  REQUIRE(filtered.empty());
}

TEST_CASE("Metadata policy copy_common keeps only common distribution tags", "[metadata][policy]") {
  const std::map<std::string, std::string> source = {
      {"title", "Song"},
      {"artist", "Artist"},
      {"custom_tag", "value"},
  };

  const auto filtered = automix::util::applyMetadataPolicy(source, "copy_common");
  REQUIRE(filtered.count("title") == 1);
  REQUIRE(filtered.count("artist") == 1);
  REQUIRE(filtered.count("custom_tag") == 0);
}

TEST_CASE("Metadata policy override_template merges template values", "[metadata][policy]") {
  const std::map<std::string, std::string> source = {
      {"title", "Old Title"},
      {"artist", "Artist"},
  };
  const std::map<std::string, std::string> templ = {
      {"title", "New Title"},
      {"album", "Compilation"},
  };

  const auto filtered = automix::util::applyMetadataPolicy(source, "override_template", templ);
  REQUIRE(filtered.at("title") == "New Title");
  REQUIRE(filtered.at("album") == "Compilation");
  REQUIRE(filtered.at("artist") == "Artist");
}
