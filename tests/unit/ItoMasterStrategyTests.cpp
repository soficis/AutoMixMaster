#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ai/ItoMasterAdapter.h"
#include "automaster/ItoMasterStrategy.h"
#include "domain/MasterPlan.h"
#include "engine/AudioBuffer.h"
#include "ito_master_fixture.h"

namespace {

automix::engine::AudioBuffer makeReferenceBuffer(const int seconds) {
  constexpr double sampleRate = 44100.0;
  automix::engine::AudioBuffer buffer(2, seconds * 44100, sampleRate);
  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    const double t = static_cast<double>(i) / sampleRate;
    const float value = static_cast<float>(0.2 * std::sin(2.0 * 3.14159265358979323846 * 220.0 * t) +
                                           0.1 * std::sin(2.0 * 3.14159265358979323846 * 880.0 * t));
    buffer.setSample(0, i, value);
    buffer.setSample(1, i, value);
  }
  return buffer;
}

struct ToggleGuard {
  ToggleGuard() { automix::automaster::ItoMasterStrategy::setExperimentalEnabled(true); }
  ~ToggleGuard() { automix::automaster::ItoMasterStrategy::setExperimentalEnabled(false); }
};

std::filesystem::path makePackDir(const std::string& name) {
  const auto packDir = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(packDir);
  std::filesystem::create_directories(packDir);
  {
    std::ofstream encoder(packDir / automix::ai::kItoMasterEncoderFile, std::ios::binary);
    encoder << "dummy";
    std::ofstream predictor(packDir / automix::ai::kItoMasterPredictorFile, std::ios::binary);
    predictor << "dummy";
    std::ofstream config(packDir / automix::ai::kItoMasterConfigFile);
    config << ito_test::kConfigFixture;
  }
  return packDir;
}

std::optional<std::filesystem::path> findInstalledItoPack() {
  constexpr const char* installKey = "huggingface_kramp__ito-master-onnx";
  std::vector<std::filesystem::path> roots;
#ifdef AUTOMIX_SOURCE_DIR
  roots.push_back(std::filesystem::path(AUTOMIX_SOURCE_DIR) / "assets" / "modelhub");
#endif
  roots.push_back(std::filesystem::current_path() / "assets" / "modelhub");
  roots.push_back(std::filesystem::current_path().parent_path() / "assets" / "modelhub");

  for (const auto& root : roots) {
    const auto packDir = root / installKey;
    std::error_code error;
    if (std::filesystem::is_regular_file(packDir / automix::ai::kItoMasterConfigFile, error) && !error &&
        std::filesystem::is_regular_file(packDir / automix::ai::kItoMasterEncoderFile, error) && !error &&
        std::filesystem::is_regular_file(packDir / automix::ai::kItoMasterPredictorFile, error) && !error) {
      return packDir;
    }
  }
  return std::nullopt;
}

bool hasLogEntry(const automix::domain::MasterPlan& plan, const std::string& needle) {
  return std::any_of(plan.decisionLog.begin(), plan.decisionLog.end(),
                     [&](const std::string& entry) { return entry.find(needle) != std::string::npos; });
}

} // namespace

TEST_CASE("ITO-Master strategy is inactive when the experimental toggle is off", "[ai][ito]") {
  automix::automaster::ItoMasterStrategy::setExperimentalEnabled(false);
  const auto packDir = makePackDir("automix_ito_toggle_off");
  automix::automaster::ItoMasterStrategy strategy({.packDirectory = packDir, .licenseConsented = true});

  REQUIRE_FALSE(strategy.isAvailable());
  const auto buffer = makeReferenceBuffer(1);
  const auto plan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, buffer);
  REQUIRE(hasLogEntry(plan, "experimental toggle is OFF"));
  REQUIRE(plan.targetLufs == Catch::Approx(-14.0));

  automix::automaster::MasteringReport report;
  const auto mastered = strategy.applyPlan(buffer, plan, &report);
  REQUIRE(mastered.getNumChannels() == 2);
  REQUIRE(mastered.getNumSamples() == buffer.getNumSamples());

  std::filesystem::remove_all(packDir);
}

TEST_CASE("ITO-Master strategy requires license consent and a complete pack", "[ai][ito]") {
  ToggleGuard toggleGuard;
  const auto packDir = makePackDir("automix_ito_gating");
  const auto buffer = makeReferenceBuffer(1);

  // Consent missing -> inactive even though the pack is complete.
  automix::automaster::ItoMasterStrategy noConsent({.packDirectory = packDir, .licenseConsented = false});
  REQUIRE_FALSE(noConsent.isAvailable());
  const auto noConsentPlan = noConsent.buildPlan(automix::domain::MasterPreset::DefaultStreaming, buffer);
  REQUIRE(hasLogEntry(noConsentPlan, "license consent not acknowledged"));

  // Pack incomplete -> inactive even with consent.
  const auto brokenDir = std::filesystem::temp_directory_path() / "automix_ito_broken_pack";
  std::filesystem::remove_all(brokenDir);
  std::filesystem::create_directories(brokenDir);
  automix::automaster::ItoMasterStrategy missingPack({.packDirectory = brokenDir, .licenseConsented = true});
  REQUIRE_FALSE(missingPack.isAvailable());
  const auto missingPlan = missingPack.buildPlan(automix::domain::MasterPreset::DefaultStreaming, buffer);
  REQUIRE(hasLogEntry(missingPlan, "pack artifacts missing"));

  std::filesystem::remove_all(packDir);
}

TEST_CASE("ITO-Master strategy validates the static contract and falls back without native params", "[ai][ito]") {
  ToggleGuard toggleGuard;
  const auto packDir = makePackDir("automix_ito_fallback");
  automix::automaster::ItoMasterStrategy strategy({.packDirectory = packDir, .licenseConsented = true});

  REQUIRE(strategy.isAvailable());
  const auto buffer = makeReferenceBuffer(1);

  const auto plan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, buffer);
  // The static tensor-shape contract is validated against config.json.
  REQUIRE(hasLogEntry(plan, "tensor contract valid"));
  // Without a native ORT session the model yields no 46-param tensor, so the
  // route must defer to the heuristic chain with a logged reason.
  REQUIRE(hasLogEntry(plan, "fell back to heuristic"));

  automix::automaster::MasteringReport report;
  const auto mastered = strategy.applyPlan(buffer, plan, &report);
  REQUIRE(mastered.getNumChannels() == 2);
  REQUIRE(mastered.getNumSamples() == buffer.getNumSamples());
  for (int ch = 0; ch < mastered.getNumChannels(); ++ch) {
    for (int i = 0; i < mastered.getNumSamples(); ++i) {
      REQUIRE(std::isfinite(mastered.getSample(ch, i)));
    }
  }
  REQUIRE(report.integratedLufs > -120.0);

  std::filesystem::remove_all(packDir);
}

TEST_CASE("ITO-Master strategy metadata surface exposes license, attribution and badge", "[ai][ito]") {
  REQUIRE(std::string(automix::automaster::ItoMasterStrategy::licenseLabel()) == "CC BY-NC 4.0");
  REQUIRE(std::string(automix::automaster::ItoMasterStrategy::experimentalBadge()) == "experimental");
  const std::string attribution = automix::automaster::ItoMasterStrategy::attributionText();
  REQUIRE(attribution.find("SonyResearch/ITO-Master") != std::string::npos);
  REQUIRE(attribution.find("kramp") != std::string::npos);
}

TEST_CASE("ITO-Master integration quality: skips cleanly without the installed pack", "[ai][ito][integration]") {
  const auto installedPack = findInstalledItoPack();
  if (!installedPack.has_value()) {
    WARN("ITO-Master pack not installed at assets/modelhub/huggingface_kramp__ito-master-onnx; "
         "integration quality test skipped.");
    return;
  }

  ToggleGuard toggleGuard;
  automix::automaster::ItoMasterStrategy strategy({.packDirectory = *installedPack, .licenseConsented = true});
  REQUIRE(strategy.isAvailable());

  const auto buffer = makeReferenceBuffer(3);
  const auto plan = strategy.buildPlan(automix::domain::MasterPreset::DefaultStreaming, buffer);
  REQUIRE_FALSE(plan.decisionLog.empty());

  automix::automaster::MasteringReport report;
  const auto mastered = strategy.applyPlan(buffer, plan, &report);
  REQUIRE(mastered.getNumChannels() == buffer.getNumChannels());
  REQUIRE(mastered.getNumSamples() == buffer.getNumSamples());
  for (int ch = 0; ch < mastered.getNumChannels(); ++ch) {
    for (int i = 0; i < mastered.getNumSamples(); ++i) {
      REQUIRE(std::isfinite(mastered.getSample(ch, i)));
    }
  }
  REQUIRE(report.integratedLufs > -120.0);
  REQUIRE(report.samplePeakDbfs > -120.0);
}
