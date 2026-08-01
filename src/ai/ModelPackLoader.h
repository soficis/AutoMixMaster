#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace automix::ai {

struct ModelPack {
  int schemaVersion = 1;
  std::string id;
  std::string name;
  std::string type;
  std::string taskScope;
  std::string engine;
  std::string minAppVersion;
  std::string version;
  std::string licenseId;
  std::string source;
  std::string intendedUse;
  std::string featureSchemaVersion;
  std::string modelFile;
  // Additional artifacts carried by the pack alongside the primary model file,
  // consumed as one model pack (e.g. ITO-Master: fxencoder.onnx primary +
  // mastering_tcn.onnx + config.json).
  std::vector<std::string> auxiliaryFiles;
  std::string checksum;
  std::optional<size_t> inputFeatureCount;
  std::string preferredPrecision;
  std::vector<std::string> providerAffinity;
  std::optional<int> defaultIntraOpThreads;
  std::optional<int> defaultInterOpThreads;
  bool enableProfiling = false;
  std::vector<std::string> expectedOutputKeys;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  std::filesystem::path rootPath;
};

class ModelPackLoader {
 public:
  std::optional<ModelPack> load(const std::filesystem::path& directory) const;

 private:
  std::string computeChecksum(const std::filesystem::path& filePath) const;
};

} // namespace automix::ai
