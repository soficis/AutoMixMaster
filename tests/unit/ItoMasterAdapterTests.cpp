#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ai/ItoMasterAdapter.h"
#include "engine/AudioBuffer.h"
#include "ito_master_fixture.h"

TEST_CASE("ITO-Master config parses the 46-param static contract", "[ai][ito]") {
  const auto configPath = ito_test::writeConfigFixture("automix_ito_adapter_tests");
  const auto config = automix::ai::ItoMasterAdapter::loadConfig(configPath);
  REQUIRE(config.has_value());
  REQUIRE(config->numParams == automix::ai::kItoMasterParamCount);
  REQUIRE(config->params.size() == 46);
  REQUIRE(config->encoder.file == automix::ai::kItoMasterEncoderFile);
  REQUIRE(config->predictor.file == automix::ai::kItoMasterPredictorFile);
  REQUIRE(config->encoder.outputShapes.size() == 1);
  REQUIRE(config->predictor.inputShapes.size() == 2);
  REQUIRE(automix::ai::ItoMasterAdapter::validateConfig(*config));
  std::filesystem::remove_all(configPath.parent_path());
}

TEST_CASE("ITO-Master tensor-shape smoke: [1,2,N]->[1,2048]->[1,46] static contract", "[ai][ito]") {
  const auto configPath = ito_test::writeConfigFixture("automix_ito_smoke_tests");
  const auto config = automix::ai::ItoMasterAdapter::loadConfig(configPath);
  REQUIRE(config.has_value());

  // The shape contract is validated against config.json regardless of whether
  // native ONNX Runtime is linked into this build.
  REQUIRE(automix::ai::ItoMasterAdapter::validateTensorContract(*config, 2, 441000));
  REQUIRE(automix::ai::ItoMasterAdapter::validateTensorContract(*config, 2, 1));

  // A non-stereo reference or an empty buffer breaks the [1,2,N] contract.
  REQUIRE_FALSE(automix::ai::ItoMasterAdapter::validateTensorContract(*config, 1, 441000));
  REQUIRE_FALSE(automix::ai::ItoMasterAdapter::validateTensorContract(*config, 2, 0));

  // extractEmbedding/extractParams enforce the static output dims [1,2048]/[1,46].
  std::vector<double> embedding(automix::ai::kItoMasterEmbeddingSize, 0.25);
  const auto embeddingOut = automix::ai::ItoMasterAdapter::extractEmbedding(embedding);
  REQUIRE(embeddingOut.has_value());
  REQUIRE(embeddingOut->size() == automix::ai::kItoMasterEmbeddingSize);
  REQUIRE_FALSE(
      automix::ai::ItoMasterAdapter::extractEmbedding(
          std::vector<double>(automix::ai::kItoMasterEmbeddingSize - 1, 0.25))
          .has_value());

  std::vector<double> params(automix::ai::kItoMasterParamCount, 0.5);
  const auto paramsOut = automix::ai::ItoMasterAdapter::extractParams(params);
  REQUIRE(paramsOut.has_value());
  REQUIRE(paramsOut->size() == automix::ai::kItoMasterParamCount);
  REQUIRE_FALSE(
      automix::ai::ItoMasterAdapter::extractParams(
          std::vector<double>(automix::ai::kItoMasterParamCount - 1, 0.5))
          .has_value());

  // A deterministic-fallback output ({confidence} only) must never pass as a
  // valid 46-param prediction (no false pass on the fallback path).
  REQUIRE_FALSE(automix::ai::ItoMasterAdapter::extractParams({0.61}).has_value());

#ifdef AUTOMIX_HAS_NATIVE_ORT
  SUCCEED("AUTOMIX_HAS_NATIVE_ORT is on; live model sessions are exercised by the model-gated integration test.");
#else
  SUCCEED("AUTOMIX_HAS_NATIVE_ORT is off; asserting the static tensor-shape contract parsed from config.json.");
#endif

  std::filesystem::remove_all(configPath.parent_path());
}

TEST_CASE("ITO-Master denormalization round-trips and clamps to bounds", "[ai][ito]") {
  const auto configPath = ito_test::writeConfigFixture("automix_ito_denorm_tests");
  const auto config = automix::ai::ItoMasterAdapter::loadConfig(configPath);
  REQUIRE(config.has_value());
  const auto& params = config->params;
  REQUIRE(params.size() == 46);

  // norm 0 -> min and norm 1 -> max for every parameter.
  const auto mins = automix::ai::ItoMasterAdapter::denormalize(*config, std::vector<double>(46, 0.0));
  const auto maxs = automix::ai::ItoMasterAdapter::denormalize(*config, std::vector<double>(46, 1.0));
  for (size_t i = 0; i < params.size(); ++i) {
    REQUIRE(mins[i] == Catch::Approx(params[i].min));
    REQUIRE(maxs[i] == Catch::Approx(params[i].max));
  }

  // norm == normalize(denormalize(norm)) at sampled points.
  for (const double norm : {0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0}) {
    const auto physical = automix::ai::ItoMasterAdapter::denormalize(*config, std::vector<double>(46, norm));
    const auto roundTrip = automix::ai::ItoMasterAdapter::normalize(*config, physical);
    for (size_t i = 0; i < roundTrip.size(); ++i) {
      REQUIRE(roundTrip[i] == Catch::Approx(norm).margin(1.0e-9));
    }
  }

  // Out-of-range norms clamp to the [min, max] bounds.
  const auto clampedHigh = automix::ai::ItoMasterAdapter::denormalize(*config, std::vector<double>(46, 2.5));
  const auto clampedLow = automix::ai::ItoMasterAdapter::denormalize(*config, std::vector<double>(46, -0.5));
  for (size_t i = 0; i < params.size(); ++i) {
    REQUIRE(clampedHigh[i] == Catch::Approx(params[i].max));
    REQUIRE(clampedLow[i] == Catch::Approx(params[i].min));
  }

  std::filesystem::remove_all(configPath.parent_path());
}

TEST_CASE("ITO-Master adapter maps 46 params onto chain settings within bounds", "[ai][ito]") {
  const auto configPath = ito_test::writeConfigFixture("automix_ito_mapping_tests");
  const auto config = automix::ai::ItoMasterAdapter::loadConfig(configPath);
  REQUIRE(config.has_value());

  std::vector<double> norms(46, 0.5);
  const auto settings = automix::ai::ItoMasterAdapter::apply(*config, norms);
  const auto physical = automix::ai::ItoMasterAdapter::denormalize(*config, norms);

  // Spot-check fx-scoped mapping (duplicate "parallel_weight_factor" names must
  // land on the correct stage, and the band prefixes must map by index).
  REQUIRE(settings.eq.lowShelf.gainDb == Catch::Approx(physical[0]));
  REQUIRE(settings.eq.lowShelf.freqHz == Catch::Approx(physical[1]));
  REQUIRE(settings.eq.bands[0].gainDb == Catch::Approx(physical[3]));
  REQUIRE(settings.eq.bands[1].freqHz == Catch::Approx(physical[7]));
  REQUIRE(settings.eq.bands[3].qFactor == Catch::Approx(physical[14]));
  REQUIRE(settings.eq.highShelf.gainDb == Catch::Approx(physical[15]));
  REQUIRE(settings.distortion.driveDb == Catch::Approx(physical[18]));
  REQUIRE(settings.distortion.parallelWeight == Catch::Approx(physical[19]));
  REQUIRE(settings.multiband.lowCrossoverHz == Catch::Approx(physical[20]));
  REQUIRE(settings.multiband.highCrossoverHz == Catch::Approx(physical[21]));
  REQUIRE(settings.multiband.parallelWeight == Catch::Approx(physical[22]));
  REQUIRE(settings.multiband.bands[0].compThresholdDb == Catch::Approx(physical[23]));
  REQUIRE(settings.multiband.bands[0].compRatio == Catch::Approx(physical[24]));
  REQUIRE(settings.multiband.bands[1].expThresholdDb == Catch::Approx(physical[31]));
  REQUIRE(settings.multiband.bands[1].attackMs == Catch::Approx(physical[33]));
  REQUIRE(settings.multiband.bands[2].compRatio == Catch::Approx(physical[36]));
  REQUIRE(settings.multiband.bands[2].releaseMs == Catch::Approx(physical[40]));
  REQUIRE(settings.gain.gainDb == Catch::Approx(physical[41]));
  REQUIRE(settings.imager.width == Catch::Approx(physical[42]));
  REQUIRE(settings.limiter.thresholdDb == Catch::Approx(physical[43]));
  REQUIRE(settings.limiter.attackMs == Catch::Approx(physical[44]));
  REQUIRE(settings.limiter.releaseMs == Catch::Approx(physical[45]));

  // Every denormalized value sits within its config bounds after clamping.
  for (size_t i = 0; i < config->params.size(); ++i) {
    REQUIRE(physical[i] >= config->params[i].min);
    REQUIRE(physical[i] <= config->params[i].max);
  }

  // apply() == toChainSettings(denormalize()) by contract.
  const auto mapped = automix::ai::ItoMasterAdapter::toChainSettings(*config, physical);
  REQUIRE(mapped.eq.bands[2].gainDb == settings.eq.bands[2].gainDb);
  REQUIRE(mapped.multiband.highCrossoverHz == settings.multiband.highCrossoverHz);

  std::filesystem::remove_all(configPath.parent_path());
}

TEST_CASE("ITO-Master model runner never fabricates params on the deterministic fallback", "[ai][ito]") {
  const auto packDir = std::filesystem::temp_directory_path() / "automix_ito_runner_tests";
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

  const auto config = automix::ai::ItoMasterAdapter::loadConfig(packDir / automix::ai::kItoMasterConfigFile);
  REQUIRE(config.has_value());

  automix::ai::ItoMasterModelRunner runner;
  REQUIRE(runner.load(packDir, *config));
  REQUIRE_FALSE(runner.usesNativeSession());

  automix::engine::AudioBuffer reference(2, 44100, 44100.0);
  for (int i = 0; i < reference.getNumSamples(); ++i) {
    const float value = static_cast<float>(0.25 * std::sin(2.0 * 3.14159265358979323846 * 440.0 *
                                                           static_cast<double>(i) / 44100.0));
    reference.setSample(0, i, value);
    reference.setSample(1, i, value);
  }

  const auto params = runner.predict(reference);
  REQUIRE_FALSE(params.has_value());

  std::filesystem::remove_all(packDir);
}
