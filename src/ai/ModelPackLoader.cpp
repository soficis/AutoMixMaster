#include "ai/ModelPackLoader.h"

#include <algorithm>
#include <cstdint>
#include <fstream>

#include <nlohmann/json.hpp>

#include "ai/FeatureSchema.h"
#include "util/HashUtils.h"
#include "util/StringUtils.h"

namespace automix::ai {
namespace {

std::vector<std::string> readStringArray(const nlohmann::json& json, const char* keyA, const char* keyB = nullptr) {
  if (json.contains(keyA) && json.at(keyA).is_array()) {
    return json.at(keyA).get<std::vector<std::string>>();
  }
  if (keyB != nullptr && json.contains(keyB) && json.at(keyB).is_array()) {
    return json.at(keyB).get<std::vector<std::string>>();
  }
  return {};
}

bool hasRequiredMetadata(const ModelPack& pack) {
  return !pack.licenseId.empty() && !pack.source.empty() && !pack.featureSchemaVersion.empty();
}

std::string normalizeTaskScope(std::string scope) {
  scope = util::toLower(util::trim(std::move(scope)));
  if (scope == "mix" || scope == "master" || scope == "analysis" || scope == "separation") {
    return scope;
  }
  if (scope == "stem-separation" || scope == "source-separation") {
    return "separation";
  }
  if (scope == "role" || scope == "classifier" || scope == "metadata") {
    return "analysis";
  }
  return "";
}

std::string inferTaskScopeFromType(const std::string& type) {
  const auto normalized = util::toLower(type);
  if (normalized == "mix_parameters" || normalized == "mix_model") {
    return "mix";
  }
  if (normalized == "master_parameters" || normalized == "master_model") {
    return "master";
  }
  if (normalized == "separation_model" || normalized == "source_separation" || normalized == "stem_separation") {
    return "separation";
  }
  if (normalized == "analysis_model" ||
      normalized == "role_classifier" ||
      normalized == "tag_classifier" ||
      normalized == "embedding_model" ||
      normalized == "analysis") {
    return "analysis";
  }
  return "";
}

bool requiresOnnxModelForScope(const std::string& scope) {
  return scope == "mix" || scope == "master";
}

bool hasRequiredOutputKeysForScope(const std::string& scope, const std::vector<std::string>& keys) {
  if (scope != "mix" && scope != "master") {
    return !keys.empty();
  }

  const auto hasKey = [&](const char* key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
  };

  if (scope == "mix") {
    return hasKey("confidence") && hasKey("global_gain_db") && hasKey("global_pan_bias");
  }

  return hasKey("confidence") &&
         hasKey("target_lufs") &&
         hasKey("pre_gain_db") &&
         hasKey("limiter_ceiling_db") &&
         hasKey("glue_ratio");
}

} // namespace

std::optional<ModelPack> ModelPackLoader::load(const std::filesystem::path& directory) const {
  const auto metadataPath = directory / "model.json";
  if (!std::filesystem::exists(metadataPath)) {
    return std::nullopt;
  }

  std::ifstream in(metadataPath);
  nlohmann::json json;
  in >> json;

  ModelPack pack;
  pack.schemaVersion = json.value("schemaVersion", json.value("schema_version", 1));
  pack.id = json.value("id", directory.filename().string());
  pack.name = json.value("name", pack.id);
  pack.type = json.value("type", "unknown");
  pack.taskScope = normalizeTaskScope(json.value("task_scope", json.value("taskScope", "")));
  if (pack.taskScope.empty()) {
    pack.taskScope = inferTaskScopeFromType(pack.type);
  }
  pack.engine = json.value("engine", "unknown");
  pack.minAppVersion = json.value("min_app_version", json.value("minAppVersion", "0.0.0"));
  pack.version = json.value("version", "0.0.0");
  pack.licenseId = json.value("license", json.value("licenseId", ""));
  pack.source = json.value("source", "");
  pack.intendedUse = json.value("intended_use", json.value("intendedUse", ""));
  pack.featureSchemaVersion = json.value("feature_schema_version", json.value("featureSchemaVersion", ""));
  pack.modelFile = json.value("modelFile", json.value("model_file", "model.onnx"));
  pack.auxiliaryFiles = readStringArray(json, "auxiliaryFiles", "auxiliary_files");
  pack.checksum = json.value("checksum", "");
  if (json.contains("inputFeatureCount")) {
    pack.inputFeatureCount = json.at("inputFeatureCount").get<size_t>();
  } else if (json.contains("input_feature_count")) {
    pack.inputFeatureCount = json.at("input_feature_count").get<size_t>();
  } else {
    pack.inputFeatureCount.reset();
  }

  pack.expectedOutputKeys = readStringArray(json, "expectedOutputKeys", "output_keys");
  pack.inputNames = readStringArray(json, "inputNames", "input_names");
  pack.outputNames = readStringArray(json, "outputNames", "output_names");

  if (pack.expectedOutputKeys.empty() && json.contains("output_schema") && json.at("output_schema").is_object()) {
    for (const auto& entry : json.at("output_schema").items()) {
      pack.expectedOutputKeys.push_back(entry.key());
    }
  }

  pack.preferredPrecision = json.value("preferredPrecision", json.value("preferred_precision", "auto"));
  pack.providerAffinity = readStringArray(json, "providerAffinity", "provider_affinity");
  if (json.contains("defaultIntraOpThreads")) {
    pack.defaultIntraOpThreads = json.at("defaultIntraOpThreads").get<int>();
  } else if (json.contains("default_intra_op_threads")) {
    pack.defaultIntraOpThreads = json.at("default_intra_op_threads").get<int>();
  } else {
    pack.defaultIntraOpThreads.reset();
  }
  if (json.contains("defaultInterOpThreads")) {
    pack.defaultInterOpThreads = json.at("defaultInterOpThreads").get<int>();
  } else if (json.contains("default_inter_op_threads")) {
    pack.defaultInterOpThreads = json.at("default_inter_op_threads").get<int>();
  } else {
    pack.defaultInterOpThreads.reset();
  }
  pack.enableProfiling = json.value("enableProfiling", json.value("enable_profiling", false));

  if (pack.featureSchemaVersion.empty() && json.contains("feature_schema") && json.at("feature_schema").is_object()) {
    pack.featureSchemaVersion = json.at("feature_schema").value("version", "");
  }

  pack.rootPath = directory;

  if (!hasRequiredMetadata(pack)) {
    return std::nullopt;
  }
  if (pack.taskScope.empty()) {
    return std::nullopt;
  }
  const auto inferredScope = inferTaskScopeFromType(pack.type);
  if (!inferredScope.empty() && inferredScope != pack.taskScope) {
    return std::nullopt;
  }
  if (!FeatureSchemaV1::isCompatible(pack.featureSchemaVersion)) {
    return std::nullopt;
  }
  if (pack.inputFeatureCount.has_value() && pack.inputFeatureCount.value() == 0) {
    return std::nullopt;
  }

  const auto modelPath = directory / pack.modelFile;
  if (!std::filesystem::exists(modelPath)) {
    return std::nullopt;
  }
  if (requiresOnnxModelForScope(pack.taskScope) &&
      util::toLower(modelPath.extension().string()) != ".onnx") {
    return std::nullopt;
  }
  for (const auto& auxiliaryFile : pack.auxiliaryFiles) {
    if (auxiliaryFile.empty() || !std::filesystem::exists(directory / auxiliaryFile)) {
      return std::nullopt;
    }
  }

  const std::string computedChecksum = computeChecksum(modelPath);
  if (!pack.checksum.empty() && pack.checksum != computedChecksum) {
    return std::nullopt;
  }

  if (pack.checksum.empty()) {
    pack.checksum = computedChecksum;
  }
  if (!hasRequiredOutputKeysForScope(pack.taskScope, pack.expectedOutputKeys)) {
    return std::nullopt;
  }

  return pack;
}

std::string ModelPackLoader::computeChecksum(const std::filesystem::path& filePath) const {
  std::ifstream in(filePath, std::ios::binary);
  if (!in.is_open()) {
    return "";
  }

  uint64_t hash = util::kFnv1a64OffsetBasis;
  char buffer[4096];
  while (in.good()) {
    in.read(buffer, static_cast<std::streamsize>(sizeof(buffer)));
    const auto readCount = static_cast<size_t>(in.gcount());
    if (readCount > 0) {
      hash = util::fnv1a64Update(hash, buffer, readCount);
    }
  }

  return util::toHex(hash);
}

} // namespace automix::ai
