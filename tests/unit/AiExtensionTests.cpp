#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "ai/IModelInference.h"
#include "ai/FeatureSchema.h"
#include "ai/HuggingFaceModelHub.h"
#include "ai/ModelManager.h"
#include "ai/ModelPackLoader.h"
#include "ai/ModelStrategy.h"
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

TEST_CASE("Feature schema exposes rich feature vector for AI plans", "[ai]") {
  REQUIRE(automix::ai::FeatureSchemaV1::featureCount() >= 20);
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
