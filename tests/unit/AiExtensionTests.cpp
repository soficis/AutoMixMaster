#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <unordered_map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <juce_events/juce_events.h>
#include <nlohmann/json.hpp>

#include "ai/AutoMixStrategyAI.h"
#include "ai/IModelInference.h"
#include "util/Sha256.h"
#include "ai/FeatureSchema.h"
#include "ai/GpuProvider.h"
#include "ai/HuggingFaceModelHub.h"
#include "ai/ModelManager.h"
#include "ai/ModelPackLoader.h"
#include "ai/ModelStrategy.h"
#include "app/controllers/ModelController.h"
#include "ai/ModelLicensePolicy.h"
#include "app/ui/HeroWaveform.h"

namespace {

std::optional<std::filesystem::path> findRepoRelativePath(const std::filesystem::path& relative) {
  std::vector<std::filesystem::path> candidates;
#ifdef AUTOMIX_SOURCE_DIR
  candidates.push_back(std::filesystem::path(AUTOMIX_SOURCE_DIR) / relative);
#endif
  candidates.push_back(std::filesystem::current_path() / relative);
  candidates.push_back(std::filesystem::current_path().parent_path() / relative);
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::vector<std::string> readCuratedModelIdsFromSource(const std::filesystem::path& sourcePath) {
  std::ifstream stream(sourcePath);
  if (!stream.good()) {
    return {};
  }
  const std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  const auto markerPos = content.find("curatedModelIds()");
  if (markerPos == std::string::npos) {
    return {};
  }
  const auto openBrace = content.find('{', markerPos);
  if (openBrace == std::string::npos) {
    return {};
  }
  std::vector<std::string> ids;
  std::string current;
  bool inString = false;
  for (size_t i = openBrace + 1; i < content.size(); ++i) {
    const char c = content[i];
    if (inString) {
      if (c == '\\') {
        ++i;
      } else if (c == '"') {
        ids.push_back(current);
        current.clear();
        inString = false;
      } else {
        current.push_back(c);
      }
      continue;
    }
    if (c == '"') {
      inString = true;
    } else if (c == '}') {
      break;
    }
  }
  return ids;
}

} // namespace

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

class ScriptedMixInference final : public automix::ai::IModelInference {
 public:
  bool isAvailable() const override { return true; }

  bool loadModel(const std::filesystem::path&) override { return true; }

  automix::ai::InferenceResult run(const automix::ai::InferenceRequest&) const override {
    automix::ai::InferenceResult result;
    result.usedModel = true;
    result.logMessage = "scripted inference";
    result.outputs = outputs_;
    return result;
  }

  void setOutputs(std::unordered_map<std::string, double> outputs) { outputs_ = std::move(outputs); }

 private:
  std::unordered_map<std::string, double> outputs_;
};

automix::domain::MixPlan heuristicMixPlanWithStems(const std::vector<std::string>& stemIds) {
  automix::domain::MixPlan plan;
  for (const auto& stemId : stemIds) {
    automix::domain::StemMixDecision decision;
    decision.stemId = stemId;
    decision.gainDb = 0.0;
    decision.pan = 0.0;
    plan.stemDecisions.push_back(decision);
  }
  return plan;
}

std::vector<automix::analysis::StemAnalysisEntry> analysisEntriesForStems(
    const std::vector<std::string>& stemIds) {
  std::vector<automix::analysis::StemAnalysisEntry> entries;
  for (const auto& stemId : stemIds) {
    entries.push_back({.stemId = stemId,
                       .stemName = stemId,
                       .metrics = {.rmsDb = -20.0, .lowEnergy = 0.4, .midEnergy = 0.4, .highEnergy = 0.2}});
  }
  return entries;
}

automix::domain::Session sessionWithStems(const std::vector<std::string>& stemIds) {
  automix::domain::Session session;
  for (const auto& stemId : stemIds) {
    automix::domain::Stem stem;
    stem.id = stemId;
    stem.name = stemId;
    session.stems.push_back(stem);
  }
  return session;
}

bool decisionLogContains(const automix::domain::MixPlan& plan, const std::string& needle) {
  for (const auto& line : plan.decisionLog) {
    if (line.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

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
    meta << R"({
  "schema_version": 1,
  "id": "mix-v1",
  "name": "Mix V1",
  "type": "mix_parameters",
  "engine": "onnxruntime",
  "version": "1.0.0",
  "model_file": "model.onnx",
  "license": "MIT",
  "source": "unit-test",
  "feature_schema_version": "1.0.0",
  "output_schema": {
    "confidence": "float",
    "global_gain_db": "float",
    "global_pan_bias": "float"
  }
})";
  }

  const auto pack = loader.load(tempDir);
  REQUIRE(pack.has_value());
  REQUIRE(pack->id == "mix-v1");
  REQUIRE(pack->type == "mix_parameters");
  REQUIRE(pack->taskScope == "mix");
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
    meta << R"({
  "id": "role-classifier-v1",
  "type": "role_classifier",
  "model_file": "model.onnx",
  "license": "MIT",
  "source": "unit-test",
  "feature_schema_version": "1.0.0",
  "output_schema": {
    "prob_vocals": "float"
  }
})";
  }
  {
    std::ofstream model(mixDir / "model.onnx", std::ios::binary);
    model << "mix";
    std::ofstream meta(mixDir / "model.json");
    meta << R"({
  "id": "mix-params-v1",
  "type": "mix_parameters",
  "model_file": "model.onnx",
  "license": "MIT",
  "source": "unit-test",
  "feature_schema_version": "1.0.0",
  "output_schema": {
    "confidence": "float",
    "global_gain_db": "float",
    "global_pan_bias": "float"
  }
})";
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

TEST_CASE("Model manager remaps legacy demucs analysis packs to separation scope", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_model_manager_legacy_demucs_scope";
  const auto packDir = root / "github_smartdaze_otowake-oto_htdemucs_6s.onnx";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(packDir);

  {
    std::ofstream model(packDir / "htdemucs_6s.onnx", std::ios::binary);
    model << "demucs";
    std::ofstream meta(packDir / "model.json");
    meta << R"({
  "id": "github-smartdaze-otowake-oto-htdemucs_6s-onnx",
  "name": "smartdaze/otowake-oto/htdemucs_6s.onnx",
  "type": "analysis_model",
  "task_scope": "analysis",
  "engine": "onnxruntime",
  "model_file": "htdemucs_6s.onnx",
  "license": "unknown",
  "source": "https://github.com/smartdaze/otowake-oto/releases",
  "feature_schema_version": "1.0.0",
  "output_schema": {
    "confidence": "float"
  }
})";
  }

  automix::ai::ModelManager manager(root);
  const auto packs = manager.scan();
  const auto selected = std::find_if(packs.begin(), packs.end(), [](const automix::ai::ModelPack& pack) {
    return pack.id == "github-smartdaze-otowake-oto-htdemucs_6s-onnx";
  });
  REQUIRE(selected != packs.end());
  REQUIRE(selected->taskScope == "separation");

  std::filesystem::remove_all(root);
}

TEST_CASE("Model pack loader rejects non-ONNX analysis packs", "[ai]") {
  const auto rejectedRoot = std::filesystem::temp_directory_path() / "automix_model_pack_pt_analysis";
  std::filesystem::remove_all(rejectedRoot);
  std::filesystem::create_directories(rejectedRoot);

  {
    std::ofstream model(rejectedRoot / "model.pt", std::ios::binary);
    model << "dummy";
    std::ofstream meta(rejectedRoot / "model.json");
    meta << R"({
  "id": "pt-analysis-pack",
  "type": "analysis_model",
  "task_scope": "analysis",
  "engine": "unknown",
  "model_file": "model.pt",
  "license": "cc-by-nc-4.0",
  "source": "unit-test",
  "feature_schema_version": "1.0.0",
  "output_schema": {
    "confidence": "float"
  }
})";
  }

  automix::ai::ModelPackLoader loader;
  REQUIRE_FALSE(loader.load(rejectedRoot).has_value());

  std::filesystem::remove_all(rejectedRoot);
}

TEST_CASE("Model pack loader accepts ONNX analysis packs", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_model_pack_onnx_analysis";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  {
    std::ofstream model(root / "model.onnx", std::ios::binary);
    model << "dummy";
    std::ofstream meta(root / "model.json");
    meta << R"({
  "id": "onnx-analysis-pack",
  "type": "analysis_model",
  "task_scope": "analysis",
  "engine": "onnxruntime",
  "model_file": "model.onnx",
  "license": "MIT",
  "source": "unit-test",
  "feature_schema_version": "1.0.0",
  "output_schema": {
    "confidence": "float"
  }
})";
  }

  automix::ai::ModelPackLoader loader;
  const auto pack = loader.load(root);
  REQUIRE(pack.has_value());
  REQUIRE(pack->taskScope == "analysis");

  std::filesystem::remove_all(root);
}

TEST_CASE("Model pack loader verifies the SHA-256 pack checksum", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_model_pack_sha256";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto modelPath = root / "model.onnx";
  {
    std::ofstream model(modelPath, std::ios::binary);
    model << "dummy";
  }
  const auto digest = automix::util::fileSha256(modelPath);
  REQUIRE(automix::util::isSha256Hex(digest));

  const auto writeManifest = [&root](const std::string& checksum) {
    std::ofstream meta(root / "model.json");
    meta << R"({
  "id": "sha256-analysis-pack",
  "type": "analysis_model",
  "task_scope": "analysis",
  "engine": "onnxruntime",
  "model_file": "model.onnx",
  "license": "MIT",
  "source": "unit-test",
  "feature_schema_version": "1.0.0",
  "checksum": ")";
    meta << checksum;
    meta << R"(",
  "output_schema": {
    "confidence": "float"
  }
})";
  };

  writeManifest(digest);
  automix::ai::ModelPackLoader loader;
  REQUIRE(loader.load(root).has_value());

  writeManifest("0000000000000000000000000000000000000000000000000000000000000000");
  REQUIRE_FALSE(loader.load(root).has_value());

  writeManifest("not-a-digest");
  REQUIRE_FALSE(loader.load(root).has_value());

  std::filesystem::remove_all(root);
}

TEST_CASE("Model pack loader still accepts a legacy FNV-1a checksum", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_model_pack_legacy_fnv";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  {
    std::ofstream model(root / "model.onnx", std::ios::binary);
    model << "dummy";
    std::ofstream meta(root / "model.json");
    meta << R"({
  "id": "legacy-fnv-analysis-pack",
  "type": "analysis_model",
  "task_scope": "analysis",
  "engine": "onnxruntime",
  "model_file": "model.onnx",
  "license": "MIT",
  "source": "unit-test",
  "feature_schema_version": "1.0.0",
  "checksum": "ae7a92f300881b9b",
  "output_schema": {
    "confidence": "float"
  }
})";
  }

  automix::ai::ModelPackLoader loader;
  REQUIRE(loader.load(root).has_value());

  std::filesystem::remove_all(root);
}

TEST_CASE("Only one SHA-256 implementation remains in the tree", "[ai]") {
  const auto shaSource = findRepoRelativePath("src/util/Sha256.h");
  REQUIRE(shaSource.has_value());

  for (const auto* relative : {"src/ai/HuggingFaceModelHub.cpp", "src/ai/GitHubReleaseModelHub.cpp"}) {
    const auto hubPath = findRepoRelativePath(relative);
    REQUIRE(hubPath.has_value());

    std::ifstream stream(*hubPath);
    REQUIRE(stream.good());
    const std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    REQUIRE(content.find("namespace sha256_impl") == std::string::npos);
    REQUIRE(content.find("computeFileSha256") == std::string::npos);
    REQUIRE(content.find("#include \"util/Sha256.h\"") != std::string::npos);
  }
}

TEST_CASE("Model pack loader rejects packs missing licensing metadata", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_model_pack_invalid_meta";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  {
    std::ofstream model(root / "model.onnx", std::ios::binary);
    model << "dummy";
    std::ofstream meta(root / "model.json");
    meta << R"({
  "id": "invalid-meta-pack",
  "type": "mix_parameters",
  "engine": "onnxruntime",
  "model_file": "model.onnx",
  "feature_schema_version": "1.0.0"
})";
  }

  automix::ai::ModelPackLoader loader;
  const auto pack = loader.load(root);
  REQUIRE_FALSE(pack.has_value());

  std::filesystem::remove_all(root);
}

TEST_CASE("Model pack loader rejects mismatched task scope metadata", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_model_pack_scope_mismatch";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  {
    std::ofstream model(root / "model.onnx", std::ios::binary);
    model << "dummy";
    std::ofstream meta(root / "model.json");
    meta << R"({
  "id": "scope-mismatch-pack",
  "type": "mix_parameters",
  "task_scope": "master",
  "engine": "onnxruntime",
  "model_file": "model.onnx",
  "license": "MIT",
  "source": "unit-test",
  "feature_schema_version": "1.0.0",
  "output_schema": {
    "confidence": "float",
    "global_gain_db": "float",
    "global_pan_bias": "float"
  }
})";
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

TEST_CASE("AI mix strategy applies the full documented global pan contract", "[ai]") {
  ScriptedMixInference inference;
  inference.setOutputs({{"confidence", 1.0}, {"global_gain_db", 0.0}, {"global_pan_bias", 1.0}});

  const std::vector<std::string> stemIds = {"s1"};
  const auto plan = automix::ai::AutoMixStrategyAI().buildPlan(
      sessionWithStems(stemIds), analysisEntriesForStems(stemIds), heuristicMixPlanWithStems(stemIds), &inference);

  REQUIRE(plan.stemDecisions.size() == 1);
  // A 1.0 request exceeds the 0.8 pan-deviation tolerance, so calibration scales it by 0.6.
  REQUIRE(plan.stemDecisions[0].pan == Catch::Approx(0.6));
}

TEST_CASE("AI mix strategy clamps out-of-contract output and records it", "[ai]") {
  const std::vector<std::string> stemIds = {"s1"};
  const auto base = heuristicMixPlanWithStems(stemIds);

  ScriptedMixInference atContract;
  atContract.setOutputs({{"confidence", 1.0}, {"global_gain_db", 12.0}, {"global_pan_bias", 1.0}});
  const auto contractPlan = automix::ai::AutoMixStrategyAI().buildPlan(
      sessionWithStems(stemIds), analysisEntriesForStems(stemIds), base, &atContract);

  ScriptedMixInference beyondContract;
  beyondContract.setOutputs({{"confidence", 1.0}, {"global_gain_db", 500.0}, {"global_pan_bias", -500.0}});
  const auto clampedPlan = automix::ai::AutoMixStrategyAI().buildPlan(
      sessionWithStems(stemIds), analysisEntriesForStems(stemIds), base, &beyondContract);

  REQUIRE(clampedPlan.stemDecisions[0].gainDb == Catch::Approx(contractPlan.stemDecisions[0].gainDb));
  REQUIRE(clampedPlan.stemDecisions[0].pan == Catch::Approx(-contractPlan.stemDecisions[0].pan));
  REQUIRE(decisionLogContains(clampedPlan, "clamped model output to the pack contract"));
  REQUIRE(decisionLogContains(contractPlan, "clamped model output to the pack contract") == false);
}

TEST_CASE("AI mix strategy honours per-stem gain and pan output keys", "[ai]") {
  ScriptedMixInference inference;
  inference.setOutputs({{"confidence", 1.0},
                        {"global_gain_db", 0.0},
                        {"global_pan_bias", 0.0},
                        {"stem0_gain_db", 4.0},
                        {"stem0_pan", -0.5}});

  const std::vector<std::string> stemIds = {"s1"};
  const auto plan = automix::ai::AutoMixStrategyAI().buildPlan(
      sessionWithStems(stemIds), analysisEntriesForStems(stemIds), heuristicMixPlanWithStems(stemIds), &inference);

  REQUIRE(plan.stemDecisions.size() == 1);
  // 4 dB and 0.5 are both inside the 9 dB / 0.8 deviation tolerances, so no calibration applies.
  REQUIRE(plan.stemDecisions[0].gainDb == Catch::Approx(4.0));
  REQUIRE(plan.stemDecisions[0].pan == Catch::Approx(-0.5));
}

TEST_CASE("Feature schema exposes rich feature vector for AI plans", "[ai]") {
  REQUIRE(automix::ai::FeatureSchemaV1::featureCount() >= 20);
}

TEST_CASE("Training feature schema file stays in step with the runtime schema", "[ai]") {
  const auto schemaPath = findRepoRelativePath("tools/training/feature_schema_v1.json");
  REQUIRE(schemaPath.has_value());

  std::ifstream schemaStream(*schemaPath);
  REQUIRE(schemaStream.good());
  const std::string schemaText((std::istreambuf_iterator<char>(schemaStream)),
                               std::istreambuf_iterator<char>());

  const auto schema = nlohmann::json::parse(schemaText);
  REQUIRE(schema.contains("features"));
  REQUIRE(schema.at("features").is_array());
  REQUIRE(schema.at("features").size() == automix::ai::FeatureSchemaV1::featureCount());
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible(schema.value("version", std::string{})));

  size_t mfccCount = 0;
  size_t cqtCount = 0;
  std::set<std::string> seen;
  for (const auto& entry : schema.at("features")) {
    const auto name = entry.get<std::string>();
    REQUIRE(seen.insert(name).second);
    mfccCount += name.rfind("mfcc_", 0) == 0 ? 1 : 0;
    cqtCount += name.rfind("cqt_", 0) == 0 ? 1 : 0;
  }
  REQUIRE(mfccCount == 13);
  REQUIRE(cqtCount == 24);
}

TEST_CASE("Feature vector exporter reads the canonical schema file", "[ai]") {
  const auto exporterPath = findRepoRelativePath("tools/training/export_feature_vectors.py");
  REQUIRE(exporterPath.has_value());

  std::ifstream exporterStream(*exporterPath);
  REQUIRE(exporterStream.good());
  const std::string content((std::istreambuf_iterator<char>(exporterStream)),
                            std::istreambuf_iterator<char>());

  REQUIRE(content.find("feature_schema_v1.json") != std::string::npos);
  REQUIRE(content.find("load_schema") != std::string::npos);
}

TEST_CASE("Feature schema version compatibility uses semantic versioning", "[ai]") {
  // Exact version match should be compatible
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible("1.0.0"));
  
  // Patch version updates should be compatible (backward compatible)
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible("1.0.1"));
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible("1.0.2"));
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible("1.0.99"));
  
  // Minor version updates should be compatible (backward compatible)
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible("1.1.0"));
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible("1.2.0"));
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible("1.99.0"));
  REQUIRE(automix::ai::FeatureSchemaV1::isCompatible("1.1.5"));
  
  // Different major version should be incompatible (breaking changes)
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("0.9.0"));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("2.0.0"));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("2.1.0"));
  
  // Invalid or malformed versions should be incompatible
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible(""));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("invalid"));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("1.x.0"));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("a.b.c"));
  
  // Partial versions (missing components) should be rejected per strict semver
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("1"));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("1.0"));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("1.1"));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("0"));
  REQUIRE_FALSE(automix::ai::FeatureSchemaV1::isCompatible("2"));
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

TEST_CASE("Curated hub models all have license manifest rows", "[ai][licensing]") {
  const auto manifestPath = findRepoRelativePath("docs/model-licensing-audit.json");
  REQUIRE(manifestPath.has_value());

  std::ifstream manifestStream(*manifestPath);
  REQUIRE(manifestStream.good());

  nlohmann::json manifest;
  manifestStream >> manifest;
  REQUIRE(manifest.at("schemaVersion") == 1);
  REQUIRE(manifest.at("models").is_array());

  const auto& models = manifest.at("models");
  const auto rowFor = [&models](const std::string& id) -> const nlohmann::json* {
    for (const auto& row : models) {
      if (row.value("id", "") == id) {
        return &row;
      }
    }
    return nullptr;
  };

  const auto sourcePath = findRepoRelativePath("src/ai/HuggingFaceModelHub.cpp");
  REQUIRE(sourcePath.has_value());
  const auto curated = readCuratedModelIdsFromSource(*sourcePath);
  REQUIRE(curated.size() >= 12);
  for (const auto& id : curated) {
    INFO("curated model missing license manifest row: " << id);
    REQUIRE(rowFor(id) != nullptr);
  }

  const std::vector<std::string> knownNonCommercial = {
      "SonyCSLParis/music2latent",
      "kramp/ito-master-onnx",
  };
  for (const auto& id : knownNonCommercial) {
    INFO("known non-commercial model: " << id);
    const auto* row = rowFor(id);
    REQUIRE(row != nullptr);
    REQUIRE(row->value("flagged", false) == true);
    REQUIRE(row->value("commercialUsable", true) == false);
  }
}

TEST_CASE("Discovery filters exclude incompatible models identically in curated and search paths", "[ai]") {
  automix::ai::HubModelInfo synthetic;
  synthetic.repoId = "test-org/incompatible-model";
  synthetic.modelId = "huggingface:test-org/incompatible-model";
  synthetic.displayName = "Incompatible Model";
  synthetic.primaryFile = "model.onnx";
  synthetic.license = "MIT";
  synthetic.hasOnnx = true;
  synthetic.compatible = false;
  synthetic.compatibilityReport = "unsupported architecture";

  const automix::ai::HubModelInfo compatible = [&synthetic] {
    auto info = synthetic;
    info.compatible = true;
    info.compatibilityReport.clear();
    return info;
  }();

  const automix::ai::HubModelQueryOptions curated{
      .maxResultsPerQuery = 8,
      .includeGated = false,
      .curatedOnly = true,
  };
  const automix::ai::HubModelQueryOptions search{
      .maxResultsPerQuery = 8,
      .includeGated = false,
      .curatedOnly = false,
  };

  // The compatible entry passes the shared filter under both discovery modes.
  REQUIRE(automix::ai::HuggingFaceModelHub::passesDiscoveryFilters(compatible, curated));
  REQUIRE(automix::ai::HuggingFaceModelHub::passesDiscoveryFilters(compatible, search));

  // The incompatible entry is excluded by the shared filter in both modes,
  // so curated and search discovery filter incompatibility identically.
  REQUIRE_FALSE(automix::ai::HuggingFaceModelHub::passesDiscoveryFilters(synthetic, curated));
  REQUIRE_FALSE(automix::ai::HuggingFaceModelHub::passesDiscoveryFilters(synthetic, search));
  const bool curatedExcluded = automix::ai::HuggingFaceModelHub::passesDiscoveryFilters(synthetic, curated);
  const bool searchExcluded = automix::ai::HuggingFaceModelHub::passesDiscoveryFilters(synthetic, search);
  REQUIRE(curatedExcluded == searchExcluded);
}

// ────────────────────────────────────────────────────────────────
// HeroWaveform (T2.2): zoom hit-test geometry + playhead dirty-check
// ────────────────────────────────────────────────────────────────

TEST_CASE("HeroWaveform zoom control rects match draw geometry and stay in bounds", "[ui][waveform]") {
  using automix::app::HeroWaveform;

  for (const int width : {960, 640}) {
    const juce::Rectangle<int> bounds(0, 0, width, 200);

    const auto rectIn = HeroWaveform::zoomControlRectFor(0, bounds);
    const auto rectOut = HeroWaveform::zoomControlRectFor(1, bounds);
    const auto rectReset = HeroWaveform::zoomControlRectFor(2, bounds);

    // Draw rects == hit rects: [right-136, right-108), [right-104, right-76), [right-72, right-44)
    REQUIRE(rectIn == juce::Rectangle<int>(width - 136, 4, 28, 28));
    REQUIRE(rectOut == juce::Rectangle<int>(width - 104, 4, 28, 28));
    REQUIRE(rectReset == juce::Rectangle<int>(width - 72, 4, 28, 28));

    // Correct left-to-right order, no overlap, all inside bounds
    REQUIRE(rectIn.getX() < rectOut.getX());
    REQUIRE(rectOut.getX() < rectReset.getX());
    REQUIRE(rectIn.getRight() <= rectOut.getX());
    REQUIRE(rectOut.getRight() <= rectReset.getX());
    REQUIRE(rectReset.getRight() <= bounds.getRight());
    REQUIRE(rectIn.getBottom() <= bounds.getBottom());
  }
}

TEST_CASE("HeroWaveform playhead dirty-check skips repaint on unchanged pixel", "[ui][waveform]") {
  using automix::app::HeroWaveform;

  int lastPixel = -1; // matches the member initialiser; first update always repaints

  REQUIRE(HeroWaveform::playheadPixelChanged(100, lastPixel));   // first pixel -> repaint
  REQUIRE(lastPixel == 100);

  REQUIRE_FALSE(HeroWaveform::playheadPixelChanged(100, lastPixel)); // same pixel -> no repaint

  REQUIRE(HeroWaveform::playheadPixelChanged(101, lastPixel));   // moved -> repaint
  REQUIRE(lastPixel == 101);

  REQUIRE_FALSE(HeroWaveform::playheadPixelChanged(101, lastPixel)); // same pixel -> no repaint

  // Off-view sentinel (-1) toggling in and out of view still triggers a repaint.
  REQUIRE(HeroWaveform::playheadPixelChanged(-1, lastPixel));
  REQUIRE_FALSE(HeroWaveform::playheadPixelChanged(-1, lastPixel));
}

// ────────────────────────────────────────────────────────────────
// T3.5: ITO-Master curated hub entry + CC BY-NC consent gating
// ────────────────────────────────────────────────────────────────

namespace {

bool waitForAsync(const std::function<bool()>& predicate, const int timeoutMs = 6000) {
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

struct ModelInstallProbe {
  std::atomic<int> installCalls{0};
};

automix::app::ModelController::ModelHubOps makeProbeModelHubOps(ModelInstallProbe& probe) {
  automix::app::ModelController::ModelHubOps ops;
  ops.discoverRecommended = [](const automix::ai::HubModelQueryOptions&) {
    return std::vector<automix::ai::HubModelInfo>{};
  };
  ops.installModel = [&probe](const std::string& repoId, const automix::ai::HubInstallOptions&) {
    ++probe.installCalls;
    automix::ai::HubInstallResult result;
    result.success = true;
    result.repoId = repoId;
    result.message = "installed";
    return result;
  };
  ops.modelInfo = [](const std::string& repoId) -> std::optional<automix::ai::HubModelInfo> {
    automix::ai::HubModelInfo info;
    info.repoId = repoId;
    return info;
  };
  return ops;
}

} // namespace

TEST_CASE("Model license policy requires consent for non-commercial and undeclared terms", "[ai][licensing]") {
  namespace ModelLicensePolicy = automix::ai::ModelLicensePolicy;

  REQUIRE_FALSE(ModelLicensePolicy::requiresUserConsent("MIT"));
  REQUIRE_FALSE(ModelLicensePolicy::requiresUserConsent("Apache-2.0"));
  REQUIRE_FALSE(ModelLicensePolicy::requiresUserConsent("BSD-3-Clause"));
  REQUIRE_FALSE(ModelLicensePolicy::requiresUserConsent("CC BY 4.0"));
  REQUIRE_FALSE(ModelLicensePolicy::requiresUserConsent("CC BY-SA 4.0"));
  REQUIRE_FALSE(ModelLicensePolicy::requiresUserConsent("GPL-3.0"));

  REQUIRE(ModelLicensePolicy::requiresUserConsent("CC BY-NC 4.0"));
  REQUIRE(ModelLicensePolicy::requiresUserConsent("cc-by-nc-4.0"));
  REQUIRE(ModelLicensePolicy::requiresUserConsent("CC-BY-NC-SA 4.0"));
  REQUIRE(ModelLicensePolicy::requiresUserConsent("cc_by_nc_4_0"));
  REQUIRE(ModelLicensePolicy::isNonCommercial("CC BY-NC 4.0"));
  REQUIRE_FALSE(ModelLicensePolicy::isNonCommercial("CC BY 4.0"));

  // The strict direction is deliberate: a card that declares nothing has not
  // granted commercial use, so it must not install silently.
  REQUIRE(ModelLicensePolicy::requiresUserConsent(""));
  REQUIRE(ModelLicensePolicy::requiresUserConsent("unknown"));
  REQUIRE(ModelLicensePolicy::requiresUserConsent("unverified"));
  REQUIRE(ModelLicensePolicy::requiresUserConsent("other"));
  REQUIRE(ModelLicensePolicy::isUndeclared("unknown"));
  REQUIRE_FALSE(ModelLicensePolicy::isUndeclared("MIT"));

  // An identifier that merely contains "by" must not be handed a CC deed.
  REQUIRE(ModelLicensePolicy::licenseUrl("CC BY-NC 4.0") == "https://creativecommons.org/licenses/by-nc/4.0/");
  REQUIRE(ModelLicensePolicy::licenseUrl("CC BY-NC-SA 4.0") == "https://creativecommons.org/licenses/by-nc-sa/4.0/");
  REQUIRE(ModelLicensePolicy::licenseUrl("CC BY 4.0") == "https://creativecommons.org/licenses/by/4.0/");
  REQUIRE(ModelLicensePolicy::licenseUrl("MIT") == "https://opensource.org/license/mit");
  REQUIRE(ModelLicensePolicy::licenseUrl("Apache-2.0") == "https://www.apache.org/licenses/LICENSE-2.0");
  REQUIRE(ModelLicensePolicy::licenseUrl("custom-by-hand").empty());
  REQUIRE(ModelLicensePolicy::licenseUrl("unknown").empty());

  REQUIRE_FALSE(ModelLicensePolicy::consentReason("CC BY-NC 4.0").empty());
  REQUIRE_FALSE(ModelLicensePolicy::consentReason("unknown").empty());
  REQUIRE(ModelLicensePolicy::consentReason("MIT").empty());
}

TEST_CASE("NOTICE attributes every curated model id", "[ai][licensing]") {
  const auto noticePath = findRepoRelativePath("NOTICE");
  REQUIRE(noticePath.has_value());
  std::ifstream noticeStream(*noticePath);
  REQUIRE(noticeStream.good());
  const std::string notice((std::istreambuf_iterator<char>(noticeStream)), std::istreambuf_iterator<char>());

  REQUIRE_FALSE(notice.empty());
  REQUIRE(notice.find("GPL") != std::string::npos);
  REQUIRE(notice.find("No model weights are distributed with this software") != std::string::npos);

  const auto hubPath = findRepoRelativePath("src/ai/HuggingFaceModelHub.cpp");
  REQUIRE(hubPath.has_value());
  const auto curated = readCuratedModelIdsFromSource(*hubPath);
  REQUIRE_FALSE(curated.empty());

  // A curated model absent from NOTICE is a model shipped with no attribution,
  // which is exactly the defect this inventory exists to prevent.
  for (const auto& id : curated) {
    CAPTURE(id);
    REQUIRE(notice.find(id) != std::string::npos);
  }
}

TEST_CASE("Consent decision is licence-driven rather than a fixed repo list", "[ai][licensing][controllers]") {
  const auto controllerPath = findRepoRelativePath("src/app/controllers/ModelController.cpp");
  REQUIRE(controllerPath.has_value());
  std::ifstream stream(*controllerPath);
  REQUIRE(stream.good());
  const std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

  // Locks that both enforcement gates consult the licence overload, so adding a
  // non-commercial model to the catalog cannot bypass consent silently. The
  // trailing comma is what distinguishes the two gates from the one-argument
  // lookup the install report uses to pick a display label, which is not a gate.
  const auto gateCount = [&content] {
    size_t count = 0;
    size_t pos = content.find("modelRequiresLicenseConsent(repoIdFromModelId(modelId),");
    while (pos != std::string::npos) {
      ++count;
      pos = content.find("modelRequiresLicenseConsent(repoIdFromModelId(modelId),", pos + 1);
    }
    return count;
  }();
  REQUIRE(gateCount == 2);
  REQUIRE(content.find("ModelLicensePolicy::requiresUserConsent(licenseId)") != std::string::npos);
  REQUIRE(content.find("#include \"ai/ModelLicensePolicy.h\"") != std::string::npos);
}

TEST_CASE("Stem separator only claims model-backed separation when a model weight was applied", "[ai][separator]") {
  const auto separatorPath = findRepoRelativePath("src/ai/StemSeparator.cpp");
  REQUIRE(separatorPath.has_value());
  std::ifstream stream(*separatorPath);
  REQUIRE(stream.good());
  const std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

  // A positive end-to-end case needs a model that emits stem<N>_weight, which a
  // real ONNX graph is required for, so the invariant is locked on the source
  // instead: result.usedModel must be set inside the appliedModelWeights gate.
  // Collapse those two lines back together and the false claim returns.
  const auto gate = content.find("weights = weightsFromInference(inferenceResult, fallbackWeights, result.stemRoles, &appliedModelWeights);");
  REQUIRE(gate != std::string::npos);
  const auto gateEnd = gate + std::string("weights = weightsFromInference(inferenceResult, fallbackWeights, result.stemRoles, &appliedModelWeights);").size();
  const auto claim = content.find("result.usedModel = true;", gateEnd);
  REQUIRE(claim != std::string::npos);
  REQUIRE(claim - gateEnd < 400);
}

TEST_CASE("Curated hub includes ITO-Master mapped to mastering-assistant with three-asset pack metadata", "[ai][licensing]") {
  const auto curated = automix::ai::curatedModelIds();
  REQUIRE(std::find(curated.begin(), curated.end(), "kramp/ito-master-onnx") != curated.end());

  REQUIRE(automix::ai::HuggingFaceModelHub::inferUseCase("kramp/ito-master-onnx", {}, "") == "mastering-assistant");
  REQUIRE(automix::ai::HuggingFaceModelHub::inferUseCase("kramp/ito-master-onnx",
                                                         {"audio-to-audio", "mastering"}, "") == "mastering-assistant");

  // The license-audit manifest row carries the three-asset pack reference that
  // the T3.8 mastering route consumes.
  const auto manifestPath = findRepoRelativePath("docs/model-licensing-audit.json");
  REQUIRE(manifestPath.has_value());
  std::ifstream manifestStream(*manifestPath);
  REQUIRE(manifestStream.good());
  nlohmann::json manifest;
  manifestStream >> manifest;
  const nlohmann::json* row = nullptr;
  for (const auto& candidate : manifest.at("models")) {
    if (candidate.value("id", "") == "kramp/ito-master-onnx") {
      row = &candidate;
      break;
    }
  }
  REQUIRE(row != nullptr);
  REQUIRE(row->value("license", "") == "CC BY-NC 4.0");
  REQUIRE(row->value("commercialUsable", true) == false);
  REQUIRE(row->contains("assets"));
  const auto assets = row->at("assets").get<std::vector<std::string>>();
  REQUIRE(assets.size() >= 3);
  REQUIRE(std::find(assets.begin(), assets.end(), "fxencoder.onnx") != assets.end());
  REQUIRE(std::find(assets.begin(), assets.end(), "mastering_tcn.onnx") != assets.end());
  REQUIRE(std::find(assets.begin(), assets.end(), "config.json") != assets.end());
}

TEST_CASE("ITO-Master download and activation are gated on CC BY-NC consent", "[ai][licensing][controllers]") {
  juce::ScopedJuceInitialiser_GUI juceInit;
  juce::ThreadPool pool(1);
  automix::ai::ModelManager modelManager;
  ModelInstallProbe probe;

  const auto root = std::filesystem::temp_directory_path() / "automix_ito_consent";
  std::filesystem::remove_all(root);

  std::atomic<int> completions{0};
  std::string lastStatus;
  std::string lastReport;

  automix::app::ModelController::Callbacks callbacks;
  callbacks.onInstallComplete = [&](const bool) { ++completions; };
  callbacks.onStatus = [&](const std::string& value) { lastStatus = value; };
  callbacks.onReport = [&](const std::string& value) { lastReport = value; };

  automix::app::ModelController controller(modelManager, pool, std::move(callbacks), makeProbeModelHubOps(probe));
  controller.setModelHubRoot(root);

  const std::string itoModelId = "huggingface:kramp/ito-master-onnx";

  REQUIRE(automix::app::ModelController::modelRequiresLicenseConsent("kramp/ito-master-onnx"));
  REQUIRE_FALSE(automix::app::ModelController::modelRequiresLicenseConsent("onnx-community/whisper-tiny.en"));

  // Without consent the download is blocked synchronously before any hub call.
  std::atomic_bool cancelFlag{false};
  controller.installModel(itoModelId, cancelFlag);
  REQUIRE(probe.installCalls == 0);
  REQUIRE(completions.load() == 1);
  REQUIRE(lastStatus.find("consent") != std::string::npos);
  REQUIRE(lastReport.find("CC BY-NC") != std::string::npos);
  REQUIRE_FALSE(controller.hasModelLicenseConsent(itoModelId));

  // Activation is gated the same way: a registry entry exists but the model
  // cannot be activated until consent is on record.
  std::filesystem::create_directories(root);
  {
    std::ofstream registry(root / "install_registry.json");
    registry << nlohmann::json::array(
                    {{{"modelId", itoModelId},
                      {"repoId", "kramp/ito-master-onnx"},
                      {"taskScope", "master"},
                      {"installPath", (root / "ito-master-install").string()}}})
                    .dump(2);
  }
  REQUIRE_FALSE(controller.activateInstalledModelForTask(itoModelId, "master"));
  REQUIRE(lastStatus.find("consent") != std::string::npos);

  // Acknowledging the CC BY-NC license persists the opt-in per model.
  REQUIRE(controller.acknowledgeModelLicenseConsent(itoModelId));
  REQUIRE(controller.hasModelLicenseConsent(itoModelId));

  // With consent recorded activation proceeds past the license gate.
  REQUIRE_FALSE(controller.activateInstalledModelForTask(itoModelId, "master"));
  REQUIRE(lastStatus.find("consent") == std::string::npos);

  // With consent recorded the install proceeds to the hub download path.
  controller.installModel(itoModelId, cancelFlag);
  REQUIRE(waitForAsync([&]() { return probe.installCalls >= 1 && completions.load() >= 2; }));
  REQUIRE(probe.installCalls == 1);

  std::filesystem::remove_all(root);
}

TEST_CASE("Experimental mastering routes default to disabled in RenderSettings", "[automaster][routing]") {
  const auto settingsPath = findRepoRelativePath("src/domain/RenderSettings.h");
  REQUIRE(settingsPath.has_value());
  std::ifstream stream(*settingsPath);
  REQUIRE(stream.good());
  const std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

  REQUIRE(content.find("bool itoMasteringEnabled = false;") != std::string::npos);
  REQUIRE(content.find("bool referenceMasteringEnabled = false;") != std::string::npos);
}

TEST_CASE("ITO-Master route engages only behind the session flag, consent and pack identity", "[automaster][routing]") {
  const auto controllerPath = findRepoRelativePath("src/app/controllers/ProcessingController.cpp");
  REQUIRE(controllerPath.has_value());
  std::ifstream stream(*controllerPath);
  REQUIRE(stream.good());
  const std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

  REQUIRE(content.find("if (settings.itoMasteringEnabled)") != std::string::npos);
  REQUIRE(content.find("packIdentity.find(\"ito-master\") != std::string::npos") != std::string::npos);
  REQUIRE(content.find("licenseConsentQuery(ai::kItoMasterModelId)") != std::string::npos);
  REQUIRE(content.find("itoEngaged = itoMaster->isAvailable();") != std::string::npos);

  // The active ITO route drives its own FX chain and ignores MasterPlan, so the
  // AI parameter blend must not also run - it would be silently discarded.
  REQUIRE(content.find("if (masterInference != nullptr && !itoEngaged)") != std::string::npos);
}

// ────────────────────────────────────────────────────────────────────────────────
TEST_CASE("Optional ORT provider plugin and compiled-model cache policy needs no SDK to verify",
          "[ai][gpu]") {
  namespace gpu = automix::ai::gpu;

  const auto v130 = gpu::parseOrtVersion("1.30.0");
  REQUIRE(v130.known);
  REQUIRE(v130.major == 1);
  REQUIRE(v130.minor == 30);
  REQUIRE(v130.patch == 0);
  REQUIRE(gpu::toString(v130) == "1.30.0");

  const auto v122 = gpu::parseOrtVersion("1.22");
  REQUIRE(v122.known);
  REQUIRE(v122.minor == 22);
  REQUIRE(v122.patch == 0);

  const auto v2 = gpu::parseOrtVersion("2");
  REQUIRE(v2.known);
  REQUIRE(v2.major == 2);
  REQUIRE(v2.minor == 0);

  REQUIRE_FALSE(gpu::parseOrtVersion("").known);
  REQUIRE_FALSE(gpu::parseOrtVersion("not-a-version").known);
  REQUIRE(gpu::toString(gpu::parseOrtVersion("")) == "unknown");

  REQUIRE_FALSE(gpu::supportsEpPlugin(gpu::parseOrtVersion("1.22.0")));
  REQUIRE(gpu::supportsEpPlugin(gpu::parseOrtVersion("1.23.0")));
  REQUIRE(gpu::supportsEpPlugin(v130));
  REQUIRE(gpu::supportsEpPlugin(v2));
  REQUIRE_FALSE(gpu::supportsEpPlugin(gpu::parseOrtVersion("")));
  REQUIRE_FALSE(gpu::supportsEpContext(gpu::parseOrtVersion("1.21.0")));
  REQUIRE(gpu::supportsEpContext(gpu::parseOrtVersion("1.22.0")));

  const auto notCompiled = gpu::decidePluginEpAttempt(false, v130, "cuda", "cuda_provider.dll");
  REQUIRE_FALSE(notCompiled.attempt);
  REQUIRE(notCompiled.reason.find("not compiled in") != std::string::npos);

  const auto cpuNeedsNoPlugin = gpu::decidePluginEpAttempt(true, v130, "cpu", "cuda_provider.dll");
  REQUIRE_FALSE(cpuNeedsNoPlugin.attempt);
  REQUIRE(cpuNeedsNoPlugin.reason.find("does not need a plugin") != std::string::npos);

  const auto tooOld = gpu::decidePluginEpAttempt(true, gpu::parseOrtVersion("1.22.0"), "cuda",
                                                 "cuda_provider.dll");
  REQUIRE_FALSE(tooOld.attempt);
  REQUIRE(tooOld.reason.find("predates") != std::string::npos);
  REQUIRE(tooOld.reason.find("1.22.0") != std::string::npos);

  const auto noLibrary = gpu::decidePluginEpAttempt(true, v130, "cuda", "");
  REQUIRE_FALSE(noLibrary.attempt);
  REQUIRE(noLibrary.reason.find("no provider plugin library") != std::string::npos);

  const auto ready = gpu::decidePluginEpAttempt(true, v130, "cuda", "cuda_provider.dll");
  REQUIRE(ready.attempt);
  REQUIRE(ready.reason.empty());

  const std::string digest = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
  REQUIRE(gpu::isSha256Hex64(digest));
  REQUIRE(gpu::isSha256Hex64("9F86D081884C7D659A2FEAA0C55AD015A3BF4F1B2B0B822CD15D6C15B0F00A08"));
  REQUIRE_FALSE(gpu::isSha256Hex64(digest.substr(1)));
  REQUIRE_FALSE(gpu::isSha256Hex64(std::string(64, 'z')));

  const auto key = gpu::compiledModelCacheKey(digest, "CUDAExecutionProvider", "sm_90",
                                              "32.0.15.5222 / CUDA 12.8", "1.30.0");
  REQUIRE_FALSE(key.empty());
  REQUIRE(key.find(digest) == 0);
  REQUIRE(key.find("cuda") != std::string::npos);
  REQUIRE(key.find("sm_90") != std::string::npos);
  REQUIRE(key.find("1.30.0") != std::string::npos);
  REQUIRE(key.find("/") == std::string::npos);
  REQUIRE(key.find(" ") == std::string::npos);

  REQUIRE(gpu::compiledModelCacheKey(digest, "cuda", "sm_90", "32.0.15.5222 / CUDA 12.8",
                                     "1.30.0") == key);
  REQUIRE(gpu::compiledModelCacheKey(digest, "cuda", "sm_120", "32.0.15.5222 / CUDA 12.8",
                                     "1.30.0") != key);
  REQUIRE(gpu::compiledModelCacheKey(digest, "coreml", "sm_90", "32.0.15.5222 / CUDA 12.8",
                                     "1.30.0") != key);
  REQUIRE(gpu::compiledModelCacheKey(digest, "cuda", "sm_90", "32.0.15.5222 / CUDA 12.8",
                                     "1.31.0") != key);
  REQUIRE(gpu::compiledModelCacheKey("not-a-digest", "cuda", "sm_90", "driver", "1.30.0").empty());

  const auto& chain = gpu::providerPriorityChain();
  REQUIRE(chain.size() == 7);
  REQUIRE(chain.front() == gpu::kProviderAne);
  REQUIRE(chain.back() == gpu::kProviderCpu);
}

// T3.8: ITO-Master mastering stage — pack schema auxiliary artifacts
// ────────────────────────────────────────────────────────────────────────────────

TEST_CASE("Model pack loader carries auxiliary artifacts and rejects incomplete packs", "[ai]") {
  const auto root = std::filesystem::temp_directory_path() / "automix_ito_pack_aux";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  {
    std::ofstream model(root / "fxencoder.onnx", std::ios::binary);
    model << "encoder";
    std::ofstream predictor(root / "mastering_tcn.onnx", std::ios::binary);
    predictor << "predictor";
    std::ofstream config(root / "config.json");
    config << "{}";
    std::ofstream meta(root / "model.json");
    meta << R"({
  "id": "ito-master-pack",
  "name": "ITO-Master",
  "type": "master_parameters",
  "task_scope": "master",
  "engine": "onnxruntime",
  "model_file": "fxencoder.onnx",
  "auxiliary_files": ["mastering_tcn.onnx", "config.json"],
  "license": "CC BY-NC 4.0",
  "source": "https://huggingface.co/kramp/ito-master-onnx",
  "feature_schema_version": "1.0.0",
  "output_schema": {
    "confidence": "float",
    "target_lufs": "float",
    "pre_gain_db": "float",
    "limiter_ceiling_db": "float",
    "glue_ratio": "float"
  }
})";
  }

  automix::ai::ModelPackLoader loader;
  const auto complete = loader.load(root);
  REQUIRE(complete.has_value());
  REQUIRE(complete->auxiliaryFiles.size() == 2);
  REQUIRE(complete->auxiliaryFiles[0] == "mastering_tcn.onnx");
  REQUIRE(complete->auxiliaryFiles[1] == "config.json");

  // A pack whose manifest lists an auxiliary artifact that is not on disk must
  // be rejected: the ITO route can never complete with a partial pack.
  std::filesystem::remove(root / "mastering_tcn.onnx");
  REQUIRE_FALSE(loader.load(root).has_value());

  std::filesystem::remove_all(root);
}

TEST_CASE("Model strategy returns base plans unchanged when no model inference is available", "[ai]") {
  std::vector<automix::analysis::StemAnalysisEntry> entries;
  entries.push_back({.stemId = "s1",
                     .stemName = "stem",
                     .metrics = {.rmsDb = -20.0, .lowEnergy = 0.4, .midEnergy = 0.4, .highEnergy = 0.2}});

  automix::domain::MixPlan baseMix;
  baseMix.dryWet = 0.42;
  baseMix.mixBusHeadroomDb = 5.0;
  baseMix.decisionLog.push_back("heuristic mix");

  automix::domain::MasterPlan baseMaster;
  baseMaster.targetLufs = -16.0;
  baseMaster.preGainDb = 2.0;
  baseMaster.decisionLog.push_back("heuristic master");

  // With no inference the strategy must pass the base plans through unchanged
  // (strategies fall to heuristics) until the T3.8 mastering route ships.
  automix::ai::ModelStrategy strategy;
  const auto [mixOut, masterOut] = strategy.applyOverrides(nullptr, entries, baseMix, baseMaster);

  REQUIRE(mixOut.dryWet == Catch::Approx(0.42));
  REQUIRE(mixOut.mixBusHeadroomDb == Catch::Approx(5.0));
  REQUIRE(mixOut.decisionLog.size() == 1);
  REQUIRE(mixOut.decisionLog[0] == "heuristic mix");
  REQUIRE(masterOut.targetLufs == Catch::Approx(-16.0));
  REQUIRE(masterOut.preGainDb == Catch::Approx(2.0));
  REQUIRE(masterOut.decisionLog.size() == 1);
  REQUIRE(masterOut.decisionLog[0] == "heuristic master");

  // An unloaded inference object follows the same pass-through contract.
  DummyModelInference unloaded;
  const auto [mixPass, masterPass] = strategy.applyOverrides(&unloaded, entries, baseMix, baseMaster);
  REQUIRE(mixPass.dryWet == Catch::Approx(0.42));
  REQUIRE(masterPass.targetLufs == Catch::Approx(-16.0));
  REQUIRE(masterPass.decisionLog.size() == 1);
}
