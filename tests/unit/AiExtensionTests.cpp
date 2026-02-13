#include <filesystem>
#include <fstream>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ai/IModelInference.h"
#include "ai/FeatureSchema.h"
#include "ai/ModelManager.h"
#include "ai/ModelPackLoader.h"
#include "ai/ModelStrategy.h"

namespace {

class DummyModelInference final : public automix::ai::IModelInference {
 public:
  bool isAvailable() const override { return loaded_; }

  bool loadModel(const std::filesystem::path&) override {
    loaded_ = true;
    return true;
  }

  automix::ai::InferenceResult run(const automix::ai::InferenceRequest&) const override {
    automix::ai::InferenceResult result;
    result.usedModel = loaded_;
    result.logMessage = "dummy inference";
    result.outputs = {
        {"dryWet", 0.77},
        {"targetLufs", -12.5},
    };
    return result;
  }

 private:
  bool loaded_ = false;
};

} // namespace

TEST_CASE("Model pack loader parses schema and defaults", "[ai]") {
  automix::ai::ModelPackLoader loader;
  const auto none = loader.load("missing_model_pack_dir");
  REQUIRE(none.has_value() == false);

  const auto tempDir = std::filesystem::temp_directory_path() / "automix_model_pack";
  std::filesystem::create_directories(tempDir);

  {
    std::ofstream model(tempDir / "model.onnx", std::ios::binary);
    model << "dummy";
  }

  {
    std::ofstream meta(tempDir / "model.json");
    meta << R"({"schema_version":1,"id":"mix-v1","name":"Mix V1","type":"mix_parameters","engine":"onnxruntime","version":"1.0.0","model_file":"model.onnx","license":"MIT","source":"unit-test","feature_schema_version":"1.0.0","output_schema":{"target_lufs":"float"}})";
  }

  const auto pack = loader.load(tempDir);
  REQUIRE(pack.has_value());
  REQUIRE(pack->id == "mix-v1");
  REQUIRE(pack->type == "mix_parameters");
  REQUIRE(pack->engine == "onnxruntime");

  std::filesystem::remove_all(tempDir);
}

TEST_CASE("Model manager scans packs and stores active selections", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_model_manager";
  const auto roleDir = root / "role-classifier-v1";
  const auto mixDir = root / "mix-params-v1";
  std::filesystem::create_directories(roleDir);
  std::filesystem::create_directories(mixDir);

  {
    std::ofstream model(roleDir / "model.onnx", std::ios::binary);
    model << "role";
    std::ofstream meta(roleDir / "model.json");
    meta << R"({"id":"role-classifier-v1","type":"role_classifier","model_file":"model.onnx","license":"MIT","source":"unit-test","feature_schema_version":"1.0.0","output_schema":{"prob_vocals":"float"}})";
  }
  {
    std::ofstream model(mixDir / "model.onnx", std::ios::binary);
    model << "mix";
    std::ofstream meta(mixDir / "model.json");
    meta << R"({"id":"mix-params-v1","type":"mix_parameters","model_file":"model.onnx","license":"MIT","source":"unit-test","feature_schema_version":"1.0.0","output_schema":{"target_lufs":"float"}})";
  }

  automix::ai::ModelManager manager(root);
  const auto packs = manager.scan();
  REQUIRE(packs.size() >= 2);
  bool foundRole = false;
  bool foundMix = false;
  for (const auto& pack : packs) {
    foundRole = foundRole || pack.id == "role-classifier-v1";
    foundMix = foundMix || pack.id == "mix-params-v1";
  }
  REQUIRE(foundRole);
  REQUIRE(foundMix);
  const auto rolePacks = manager.packsForType("role_classifier");
  REQUIRE(rolePacks.empty() == false);

  manager.setActivePackId("role", "role-classifier-v1");
  REQUIRE(manager.activePackId("role") == "role-classifier-v1");

  std::filesystem::remove_all(root);
}

TEST_CASE("Model pack loader rejects packs missing licensing metadata", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_model_pack_invalid_meta";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  {
    std::ofstream model(root / "model.onnx", std::ios::binary);
    model << "dummy";
    std::ofstream meta(root / "model.json");
    meta << R"({"id":"invalid-meta-pack","type":"mix_parameters","engine":"onnxruntime","model_file":"model.onnx","feature_schema_version":"1.0.0"})";
  }

  automix::ai::ModelPackLoader loader;
  const auto pack = loader.load(root);
  REQUIRE_FALSE(pack.has_value());

  std::filesystem::remove_all(root);
}

TEST_CASE("Null model inference returns clear run log", "[ai]") {
  automix::ai::NullModelInference inference;
  const auto loaded = inference.loadModel("unused.onnx");
  REQUIRE(loaded == false);

  const automix::ai::InferenceRequest request{
      .task = "mix_parameters",
      .features = {1.0, 2.0},
  };
  const auto result = inference.run(request);
  REQUIRE(result.usedModel == false);
  REQUIRE(result.outputs.empty());
  REQUIRE(result.logMessage.find("no model loaded") != std::string::npos);
}

TEST_CASE("Model strategy applies overrides when model inference is available", "[ai]") {
  DummyModelInference inference;
  REQUIRE(inference.loadModel("dummy.onnx"));

  std::vector<automix::analysis::StemAnalysisEntry> entries;
  entries.push_back(
      {.stemId = "s1", .stemName = "stem", .metrics = {.rmsDb = -20.0, .lowEnergy = 0.4, .midEnergy = 0.4, .highEnergy = 0.2}});

  automix::domain::MixPlan baseMix;
  automix::domain::MasterPlan baseMaster;

  automix::ai::ModelStrategy strategy;
  auto [mixOut, masterOut] = strategy.applyOverrides(&inference, entries, baseMix, baseMaster);

  REQUIRE(mixOut.dryWet == Catch::Approx(0.77));
  REQUIRE(masterOut.targetLufs == Catch::Approx(-12.5));
}

TEST_CASE("Feature schema exposes rich feature vector for AI plans", "[ai]") {
  REQUIRE(automix::ai::FeatureSchemaV1::featureCount() >= 20);
}

TEST_CASE("Model manager scans demo packs from assets roots", "[ai]") {
  automix::ai::ModelManager manager("missing_root_for_test");
  const auto packs = manager.scan();

  bool foundDemoRole = false;
  bool foundDemoMix = false;
  bool foundDemoMaster = false;
  for (const auto& pack : packs) {
    foundDemoRole = foundDemoRole || pack.id == "demo-role-v1";
    foundDemoMix = foundDemoMix || pack.id == "demo-mix-v1";
    foundDemoMaster = foundDemoMaster || pack.id == "demo-master-v1";
  }

  REQUIRE(foundDemoRole);
  REQUIRE(foundDemoMix);
  REQUIRE(foundDemoMaster);
}
