#include "ai/ModelPackLoader.h"

#include <cstdint>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "ai/FeatureSchema.h"

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
  pack.engine = json.value("engine", "unknown");
  pack.minAppVersion = json.value("min_app_version", json.value("minAppVersion", "0.0.0"));
  pack.version = json.value("version", "0.0.0");
  pack.licenseId = json.value("license", json.value("licenseId", ""));
  pack.source = json.value("source", "");
  pack.intendedUse = json.value("intended_use", json.value("intendedUse", ""));
  pack.featureSchemaVersion = json.value("feature_schema_version", json.value("featureSchemaVersion", ""));
  pack.modelFile = json.value("modelFile", json.value("model_file", "model.onnx"));
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

  if (pack.featureSchemaVersion.empty() && json.contains("feature_schema") && json.at("feature_schema").is_object()) {
    pack.featureSchemaVersion = json.at("feature_schema").value("version", "");
  }

  pack.rootPath = directory;

  if (!hasRequiredMetadata(pack)) {
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

  const std::string computedChecksum = computeChecksum(modelPath);
  if (!pack.checksum.empty() && pack.checksum != computedChecksum) {
    return std::nullopt;
  }

  if (pack.checksum.empty()) {
    pack.checksum = computedChecksum;
  }

  return pack;
}

std::string ModelPackLoader::computeChecksum(const std::filesystem::path& filePath) const {
  std::ifstream in(filePath, std::ios::binary);
  if (!in.is_open()) {
    return "";
  }

  uint64_t hash = 14695981039346656037ull;
  constexpr uint64_t prime = 1099511628211ull;

  char byte = 0;
  while (in.get(byte)) {
    hash ^= static_cast<uint8_t>(byte);
    hash *= prime;
  }

  std::ostringstream output;
  output << std::hex << hash;
  return output.str();
}

} // namespace automix::ai
