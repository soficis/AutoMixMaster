#include "ai/ModelPackLoader.h"

#include <cstdint>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace automix::ai {

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
  pack.modelFile = json.value("modelFile", json.value("model_file", "model.onnx"));
  pack.checksum = json.value("checksum", "");
  if (json.contains("inputFeatureCount")) {
    pack.inputFeatureCount = json.at("inputFeatureCount").get<size_t>();
  } else if (json.contains("input_feature_count")) {
    pack.inputFeatureCount = json.at("input_feature_count").get<size_t>();
  } else {
    pack.inputFeatureCount.reset();
  }

  if (json.contains("expectedOutputKeys") && json.at("expectedOutputKeys").is_array()) {
    pack.expectedOutputKeys = json.at("expectedOutputKeys").get<std::vector<std::string>>();
  } else if (json.contains("output_keys") && json.at("output_keys").is_array()) {
    pack.expectedOutputKeys = json.at("output_keys").get<std::vector<std::string>>();
  } else {
    pack.expectedOutputKeys.clear();
  }
  pack.rootPath = directory;

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
