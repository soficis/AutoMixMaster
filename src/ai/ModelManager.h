#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "ai/ModelPackLoader.h"

namespace automix::ai {

class ModelManager {
 public:
  explicit ModelManager(std::filesystem::path rootPath = "ModelPacks");

  void setRootPath(const std::filesystem::path& rootPath);
  void setRootPaths(const std::vector<std::filesystem::path>& rootPaths);
  const std::vector<std::filesystem::path>& rootPaths() const;
  std::vector<ModelPack> scan();
  const std::vector<ModelPack>& availablePacks() const;
  std::vector<ModelPack> packsForType(const std::string& type) const;

  void setActivePackId(const std::string& task, const std::string& packId);
  std::string activePackId(const std::string& task) const;

 private:
  std::vector<std::filesystem::path> rootPaths_;
  ModelPackLoader loader_;
  std::vector<ModelPack> availablePacks_;
  std::unordered_map<std::string, std::string> activePackByTask_;
};

} // namespace automix::ai
