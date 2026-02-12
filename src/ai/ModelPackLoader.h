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
  std::string engine;
  std::string minAppVersion;
  std::string version;
  std::string modelFile;
  std::string checksum;
  std::optional<size_t> inputFeatureCount;
  std::vector<std::string> expectedOutputKeys;
  std::filesystem::path rootPath;
};

class ModelPackLoader {
 public:
  std::optional<ModelPack> load(const std::filesystem::path& directory) const;

 private:
  std::string computeChecksum(const std::filesystem::path& filePath) const;
};

} // namespace automix::ai
