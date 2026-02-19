#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <numbers>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <juce_events/juce_events.h>

#include "app/controllers/ExportController.h"
#include "app/controllers/ImportController.h"
#include "app/controllers/OriginalMixController.h"
#include "app/controllers/ProcessingController.h"
#include "app/controllers/ProfileController.h"
#include "app/controllers/PreviewController.h"
#include "app/controllers/SessionController.h"
#include "domain/ProjectProfile.h"
#include "engine/AudioBuffer.h"
#include "util/WavWriter.h"

namespace {

automix::engine::AudioBuffer makeTone(const double sampleRate, const int samples, const double frequency) {
  automix::engine::AudioBuffer buffer(2, samples, sampleRate);
  for (int i = 0; i < samples; ++i) {
    const double t = static_cast<double>(i) / sampleRate;
    const float sample = static_cast<float>(0.2 * std::sin(2.0 * std::numbers::pi * frequency * t));
    buffer.setSample(0, i, sample);
    buffer.setSample(1, i, sample);
  }
  return buffer;
}

std::filesystem::path uniqueTempPath(const std::string& stem) {
  const auto nonce = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  return std::filesystem::temp_directory_path() / (stem + "_" + nonce);
}

bool waitFor(const std::function<bool()>& predicate, const int timeoutMs = 6000) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }

    if (auto* messageManager = juce::MessageManager::getInstanceWithoutCreating(); messageManager != nullptr) {
      messageManager->runDispatchLoopUntil(10);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  return predicate();
}

automix::domain::Session makeBasicSession(const std::filesystem::path& stemPath = {}) {
  automix::domain::Session session;
  session.sessionName = "controller-test-session";
  session.projectProfileId = "default";
  session.safetyPolicyId = "balanced";
  session.renderSettings.rendererName = "BuiltIn";
  session.renderSettings.outputFormat = "wav";
  session.renderSettings.outputSampleRate = 44100;
  session.renderSettings.blockSize = 1024;
  session.renderSettings.outputBitDepth = 24;
  session.renderSettings.exportSpeedMode = "quick";

  if (!stemPath.empty()) {
    automix::domain::Stem stem;
    stem.id = "stem_1";
    stem.name = "tone";
    stem.filePath = stemPath.string();
    stem.enabled = true;
    session.stems.push_back(stem);
  }
  return session;
}

} // namespace

TEST_CASE("ImportController imports selected files", "[controllers][import]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  const auto testDir = uniqueTempPath("automix_import_ok");
  std::filesystem::create_directories(testDir);
  const auto wavPath = testDir / "tone.wav";
  automix::util::WavWriter().write(wavPath, makeTone(44100.0, 2048, 220.0), 24);

  juce::ThreadPool pool(1);
  std::optional<automix::app::ImportResult> result;

  automix::app::ImportController::Callbacks callbacks;
  callbacks.onImportComplete = [&](automix::app::ImportResult value) {
    result = std::move(value);
  };
  automix::app::ImportController controller(pool, std::move(callbacks));

  controller.importFiles({juce::File(wavPath.string())}, false, 4);

  REQUIRE(waitFor([&]() { return result.has_value(); }));
  REQUIRE(result->stems.size() == 1);
  REQUIRE(result->stems.front().filePath == wavPath.string());

  std::filesystem::remove_all(testDir);
}

TEST_CASE("ImportController ignores empty file list", "[controllers][import]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  juce::ThreadPool pool(1);
  bool invoked = false;

  automix::app::ImportController::Callbacks callbacks;
  callbacks.onImportComplete = [&](automix::app::ImportResult) {
    invoked = true;
  };
  automix::app::ImportController controller(pool, std::move(callbacks));

  controller.importFiles({}, false, 4);
  if (auto* messageManager = juce::MessageManager::getInstanceWithoutCreating(); messageManager != nullptr) {
    messageManager->runDispatchLoopUntil(100);
  }
  REQUIRE_FALSE(invoked);
}

TEST_CASE("ExportController returns cancelled result when cancel flag is pre-set", "[controllers][export][cancel]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  const auto testDir = uniqueTempPath("automix_export_cancel");
  std::filesystem::create_directories(testDir);
  const auto wavPath = testDir / "tone.wav";
  automix::util::WavWriter().write(wavPath, makeTone(44100.0, 4096, 110.0), 24);

  juce::ThreadPool pool(1);
  std::atomic_bool cancelFlag {true};
  std::optional<automix::app::ExportResult> result;

  automix::app::ExportController::Callbacks callbacks;
  callbacks.onExportComplete = [&](automix::app::ExportResult value) {
    result = std::move(value);
  };
  automix::app::ExportController controller(pool, std::move(callbacks));

  auto session = makeBasicSession(wavPath);
  auto settings = session.renderSettings;

  controller.runExport(session, settings, {}, cancelFlag);

  REQUIRE(waitFor([&]() { return result.has_value(); }));
  REQUIRE(result->cancelled);

  std::filesystem::remove_all(testDir);
}

TEST_CASE("ProcessingController auto mix cancellation terminates early", "[controllers][processing][cancel]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  const auto testDir = uniqueTempPath("automix_processing_cancel");
  std::filesystem::create_directories(testDir);
  const auto wavPath = testDir / "tone.wav";
  automix::util::WavWriter().write(wavPath, makeTone(44100.0, 4096, 110.0), 24);

  juce::ThreadPool pool(1);
  std::atomic_bool cancelFlag {true};
  std::optional<automix::app::AutoMixResult> result;

  automix::app::ProcessingController::Callbacks callbacks;
  callbacks.onAutoMixComplete = [&](automix::app::AutoMixResult value) {
    result = std::move(value);
  };
  automix::app::ProcessingController controller(pool, std::move(callbacks));

  controller.runAutoMix(makeBasicSession(wavPath), std::nullopt, cancelFlag);

  REQUIRE(waitFor([&]() { return result.has_value(); }));
  REQUIRE(result->cancelled);

  std::filesystem::remove_all(testDir);
}

TEST_CASE("ProcessingController batch callback fires for empty folder", "[controllers][processing][batch]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  const auto inputDir = uniqueTempPath("automix_batch_empty");
  std::filesystem::create_directories(inputDir);

  juce::ThreadPool pool(1);
  std::atomic_bool cancelFlag {false};
  std::optional<automix::app::BatchResult> result;

  automix::app::ProcessingController::Callbacks callbacks;
  callbacks.onBatchComplete = [&](automix::app::BatchResult value) {
    result = std::move(value);
  };
  automix::app::ProcessingController controller(pool, std::move(callbacks));

  automix::domain::RenderSettings settings;
  settings.rendererName = "BuiltIn";
  settings.outputFormat = "wav";

  controller.runBatch(inputDir, settings, cancelFlag);

  REQUIRE(waitFor([&]() { return result.has_value(); }));
  REQUIRE_FALSE(result->errorText.isEmpty());

  std::filesystem::remove_all(inputDir);
}

TEST_CASE("SessionController save and load roundtrip", "[controllers][session]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  const auto sessionPath = uniqueTempPath("automix_session_roundtrip").replace_extension(".json");

  juce::ThreadPool pool(1);
  std::atomic_bool cancelFlag {false};
  std::optional<automix::app::SessionSaveResult> saveResult;
  std::optional<automix::app::SessionLoadResult> loadResult;

  automix::app::SessionController::Callbacks callbacks;
  callbacks.onSaveComplete = [&](automix::app::SessionSaveResult value) {
    saveResult = std::move(value);
  };
  callbacks.onLoadComplete = [&](automix::app::SessionLoadResult value) {
    loadResult = std::move(value);
  };
  automix::app::SessionController controller(pool, std::move(callbacks));

  auto session = makeBasicSession();
  session.sessionName = "roundtrip";

  controller.saveSession(sessionPath.string(), session, cancelFlag);
  REQUIRE(waitFor([&]() { return saveResult.has_value(); }));
  REQUIRE(saveResult->success);
  REQUIRE_FALSE(saveResult->cancelled);

  controller.loadSession(sessionPath.string(), cancelFlag);
  REQUIRE(waitFor([&]() { return loadResult.has_value(); }));
  REQUIRE_FALSE(loadResult->cancelled);
  REQUIRE(loadResult->session.has_value());
  REQUIRE(loadResult->session->sessionName == "roundtrip");

  std::filesystem::remove(sessionPath);
}

TEST_CASE("SessionController save respects cancellation flag", "[controllers][session][cancel]") {
  juce::ScopedJuceInitialiser_GUI juceInit;

  const auto sessionPath = uniqueTempPath("automix_session_cancel").replace_extension(".json");

  juce::ThreadPool pool(1);
  std::atomic_bool cancelFlag {true};
  std::optional<automix::app::SessionSaveResult> saveResult;

  automix::app::SessionController::Callbacks callbacks;
  callbacks.onSaveComplete = [&](automix::app::SessionSaveResult value) {
    saveResult = std::move(value);
  };
  automix::app::SessionController controller(pool, std::move(callbacks));

  controller.saveSession(sessionPath.string(), makeBasicSession(), cancelFlag);

  REQUIRE(waitFor([&]() { return saveResult.has_value(); }));
  REQUIRE(saveResult->cancelled);
  REQUIRE_FALSE(saveResult->success);
}

TEST_CASE("ProfileController applies project profile to session", "[controllers][profile]") {
  automix::app::ProfileController controller;
  automix::domain::Session session;
  automix::domain::ProjectProfile profile;
  profile.id = "podcast";
  profile.safetyPolicyId = "strict";
  profile.preferredStemCount = 6;
  profile.gpuProvider = "cpu";
  profile.outputFormat = "mp3";
  profile.lossyBitrateKbps = 192;
  profile.mp3UseVbr = true;
  profile.mp3VbrQuality = 2;
  profile.metadataPolicy = "copy_common";
  profile.rendererName = "BuiltIn";
  profile.roleModelPackId = "role-pack";
  profile.mixModelPackId = "mix-pack";
  profile.masterModelPackId = "master-pack";
  profile.platformPreset = "spotify";

  const auto applied = controller.applyProfile(session, profile);

  REQUIRE(session.projectProfileId == "podcast");
  REQUIRE(session.safetyPolicyId == "strict");
  REQUIRE(session.preferredStemCount == 6);
  REQUIRE(session.renderSettings.outputFormat == "mp3");
  REQUIRE(session.renderSettings.lossyBitrateKbps == 192);
  REQUIRE(session.renderSettings.mp3UseVbr);
  REQUIRE(applied.roleModelPackId == "role-pack");
  REQUIRE(applied.mixModelPackId == "mix-pack");
  REQUIRE(applied.masterModelPackId == "master-pack");
}

TEST_CASE("PreviewController transport application updates timeline and loop", "[controllers][preview]") {
  automix::engine::TransportController transport;
  std::atomic<int64_t> cursor {123};
  automix::domain::TimelineState timeline;
  timeline.loopEnabled = true;
  timeline.loopInSeconds = 0.2;
  timeline.loopOutSeconds = 0.8;

  automix::engine::AudioBuffer buffer(2, 48000, 48000.0);
  automix::app::PreviewController::applyTransportBuffer(buffer, timeline, transport, cursor);

  REQUIRE(cursor.load() == 0);
  REQUIRE(transport.totalSamples() == 48000);
  REQUIRE(transport.loopEnabled());
  REQUIRE(transport.loopInSeconds() == 0.2);
  REQUIRE(transport.loopOutSeconds() == 0.8);
}

TEST_CASE("OriginalMixController applySelectedPath fills payload", "[controllers][originalmix]") {
  automix::app::OriginalMixController controller;
  const auto result = controller.applySelectedPath("C:/music/original.wav", "original.wav");

  REQUIRE(result.applied);
  REQUIRE(result.path == "C:/music/original.wav");
  REQUIRE(result.statusText == "Original mix loaded");
}

TEST_CASE("OriginalMixController clear reports cleared state", "[controllers][originalmix]") {
  automix::app::OriginalMixController controller;
  const auto result = controller.clear(std::optional<std::string>("C:/music/original.wav"));

  REQUIRE(result.cleared);
  REQUIRE(result.statusText == "Original mix cleared");
}
