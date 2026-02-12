#include "ai/ModelManager.h"

#include <algorithm>

namespace automix::ai {

ModelManager::ModelManager(std::filesystem::path rootPath) : rootPath_(std::move(rootPath)) {}

void ModelManager::setRootPath(const std::filesystem::path& rootPath) { rootPath_ = rootPath; }

std::vector<ModelPack> ModelManager::scan() {
  availablePacks_.clear();

  std::error_code error;
  if (!std::filesystem::exists(rootPath_, error) || error) {
    return availablePacks_;
  }

  for (const auto& entry : std::filesystem::directory_iterator(rootPath_)) {
    if (!entry.is_directory()) {
      continue;
    }

    const auto pack = loader_.load(entry.path());
    if (pack.has_value()) {
      availablePacks_.push_back(pack.value());
    }
  }

  std::sort(availablePacks_.begin(), availablePacks_.end(),
            [](const ModelPack& a, const ModelPack& b) { return a.id < b.id; });
  return availablePacks_;
}

const std::vector<ModelPack>& ModelManager::availablePacks() const { return availablePacks_; }

std::vector<ModelPack> ModelManager::packsForType(const std::string& type) const {
  std::vector<ModelPack> matches;
  for (const auto& pack : availablePacks_) {
    if (pack.type == type) {
      matches.push_back(pack);
    }
  }
  return matches;
}

void ModelManager::setActivePackId(const std::string& task, const std::string& packId) {
  if (packId.empty()) {
    activePackByTask_.erase(task);
    return;
  }
  activePackByTask_[task] = packId;
}

std::string ModelManager::activePackId(const std::string& task) const {
  const auto it = activePackByTask_.find(task);
  return it != activePackByTask_.end() ? it->second : "";
}

} // namespace automix::ai
