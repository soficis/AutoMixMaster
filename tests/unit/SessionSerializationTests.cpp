#include <filesystem>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "domain/JsonSerialization.h"
#include "engine/SessionRepository.h"

TEST_CASE("Session serialization round trip preserves required fields", "[session]") {
  automix::domain::Session session;
  session.schemaVersion = 2;
  session.sessionName = "round_trip";
  session.originalMixPath = "C:/audio/original_mix.wav";
  session.residualBlend = 7.5;
  session.renderSettings.outputFormat = "mp3";
  session.renderSettings.exportSpeedMode = "quick";
  session.renderSettings.lossyBitrateKbps = 256;
  session.renderSettings.lossyQuality = 8;
  session.renderSettings.mp3UseVbr = true;
  session.renderSettings.mp3VbrQuality = 2;
  session.renderSettings.processingThreads = 4;
  session.renderSettings.preferHardwareAcceleration = true;
  session.renderSettings.metadataPolicy = "override_template";
  session.renderSettings.metadataTemplate = {{"artist", "AutoMixMaster"}, {"comment", "test"}};
  session.timeline.loopEnabled = true;
  session.timeline.loopInSeconds = 12.5;
  session.timeline.loopOutSeconds = 28.0;
  session.timeline.zoom = 4.0;
  session.timeline.fineScrub = true;
  session.projectProfileId = "streaming_spotify";
  session.safetyPolicyId = "strict";
  session.preferredStemCount = 6;

  automix::domain::Stem stem;
  stem.id = "s1";
  stem.name = "Vocal";
  stem.filePath = "C:/audio/vocal.wav";
  stem.origin = automix::domain::StemOrigin::Separated;
  stem.enabled = true;
  session.stems.push_back(stem);

  automix::domain::MixPlan mixPlan;
  mixPlan.dryWet = 0.8;
  mixPlan.decisionLog.push_back("test decision");
  session.mixPlan = mixPlan;

  const automix::domain::Json json = session;
  const auto decoded = json.get<automix::domain::Session>();

  REQUIRE(decoded.schemaVersion == 2);
  REQUIRE(decoded.sessionName == "round_trip");
  REQUIRE(decoded.originalMixPath.has_value());
  REQUIRE(decoded.originalMixPath.value() == "C:/audio/original_mix.wav");
  REQUIRE(decoded.residualBlend == Catch::Approx(7.5));
  REQUIRE(decoded.stems.size() == 1);
  REQUIRE(decoded.stems.front().origin == automix::domain::StemOrigin::Separated);
  REQUIRE(decoded.mixPlan.has_value());
  REQUIRE(decoded.mixPlan->dryWet == Catch::Approx(0.8));
  REQUIRE(decoded.renderSettings.outputFormat == "mp3");
  REQUIRE(decoded.renderSettings.exportSpeedMode == "quick");
  REQUIRE(decoded.renderSettings.lossyBitrateKbps == 256);
  REQUIRE(decoded.renderSettings.lossyQuality == 8);
  REQUIRE(decoded.renderSettings.mp3UseVbr == true);
  REQUIRE(decoded.renderSettings.mp3VbrQuality == 2);
  REQUIRE(decoded.renderSettings.processingThreads == 4);
  REQUIRE(decoded.renderSettings.metadataPolicy == "override_template");
  REQUIRE(decoded.renderSettings.metadataTemplate.at("artist") == "AutoMixMaster");
  REQUIRE(decoded.timeline.loopEnabled == true);
  REQUIRE(decoded.timeline.loopInSeconds == Catch::Approx(12.5));
  REQUIRE(decoded.timeline.loopOutSeconds == Catch::Approx(28.0));
  REQUIRE(decoded.timeline.zoom == Catch::Approx(4.0));
  REQUIRE(decoded.timeline.fineScrub == true);
  REQUIRE(decoded.projectProfileId == "streaming_spotify");
  REQUIRE(decoded.safetyPolicyId == "strict");
  REQUIRE(decoded.preferredStemCount == 6);
}

TEST_CASE("Session deserialization handles missing optional fields", "[session]") {
  automix::domain::Json json = {
      {"schemaVersion", 1},
      {"sessionName", "missing_optionals"},
      {"stems", automix::domain::Json::array({{{"id", "s1"}, {"name", "stem"}, {"filePath", "stem.wav"}}})},
  };

  const auto decoded = json.get<automix::domain::Session>();

  REQUIRE(decoded.mixPlan.has_value() == false);
  REQUIRE(decoded.masterPlan.has_value() == false);
  REQUIRE(decoded.originalMixPath.has_value() == false);
  REQUIRE(decoded.residualBlend == Catch::Approx(0.0));
  REQUIRE(decoded.renderSettings.blockSize == 1024);
  REQUIRE(decoded.renderSettings.outputFormat == "auto");
  REQUIRE(decoded.renderSettings.exportSpeedMode == "final");
  REQUIRE(decoded.renderSettings.lossyBitrateKbps == 320);
  REQUIRE(decoded.renderSettings.lossyQuality == 7);
  REQUIRE(decoded.renderSettings.mp3UseVbr == false);
  REQUIRE(decoded.renderSettings.mp3VbrQuality == 4);
  REQUIRE(decoded.renderSettings.metadataPolicy == "copy_all");
  REQUIRE(decoded.renderSettings.metadataTemplate.empty());
  REQUIRE(decoded.renderSettings.preferHardwareAcceleration == true);
  REQUIRE(decoded.timeline.loopEnabled == false);
  REQUIRE(decoded.timeline.zoom == Catch::Approx(1.0));
  REQUIRE(decoded.projectProfileId == "default");
  REQUIRE(decoded.safetyPolicyId == "balanced");
  REQUIRE(decoded.preferredStemCount == 4);
  REQUIRE(decoded.stems.front().enabled == true);
  REQUIRE(decoded.stems.front().origin == automix::domain::StemOrigin::Recorded);
}

TEST_CASE("Session repository save load", "[session]") {
  automix::domain::Session session;
  session.sessionName = "repo_test";

  automix::engine::SessionRepository repository;

  const std::filesystem::path path = std::filesystem::temp_directory_path() / "automix_session_test.json";
  repository.save(path, session);
  auto loaded = repository.load(path);

  REQUIRE(loaded.sessionName == "repo_test");

  std::filesystem::remove(path);
}
