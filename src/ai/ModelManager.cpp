#include "ai/ModelManager.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>
#include <sstream>
#include <unordered_set>

namespace automix::ai {
namespace {

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::vector<std::filesystem::path> defaultRoots() {
  return {
      "ModelPacks",
      "assets/models",
      "assets/modelpacks",
      "assets/ModelPacks",
      "Assets/ModelPacks",
  };
}

std::vector<std::filesystem::path> parseEnvRoots() {
  std::vector<std::filesystem::path> roots;
  const char* raw = std::getenv("AUTOMIX_MODELPACK_PATHS");
  if (raw == nullptr || *raw == '\0') {
    return roots;
  }

#if defined(_WIN32)
  constexpr char delimiter = ';';
#else
  constexpr char delimiter = ':';
#endif

  std::stringstream stream(raw);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    if (!token.empty()) {
      roots.emplace_back(token);
    }
  }
  return roots;
}

bool isModelMetadataFile(const std::filesystem::path& path) {
  return toLower(path.filename().string()) == "model.json";
}

std::vector<std::filesystem::path> expandRootCandidates(const std::filesystem::path& root) {
  std::vector<std::filesystem::path> candidates;
  if (root.empty()) {
    return candidates;
  }

  if (root.is_absolute()) {
    candidates.push_back(root);
    return candidates;
  }

  std::error_code error;
  auto base = std::filesystem::current_path(error);
  if (error) {
    candidates.push_back(root);
    return candidates;
  }

  for (int depth = 0; depth < 6; ++depth) {
    candidates.push_back(base / root);
    if (!base.has_parent_path()) {
      break;
    }
    const auto parent = base.parent_path();
    if (parent == base) {
      break;
    }
    base = parent;
  }

  return candidates;
}

} // namespace

ModelManager::ModelManager(std::filesystem::path rootPath) { rootPaths_.push_back(std::move(rootPath)); }

void ModelManager::setRootPath(const std::filesystem::path& rootPath) {
  rootPaths_.clear();
  rootPaths_.push_back(rootPath);
}

void ModelManager::setRootPaths(const std::vector<std::filesystem::path>& rootPaths) {
  rootPaths_.clear();
  for (const auto& root : rootPaths) {
    if (!root.empty()) {
      rootPaths_.push_back(root);
    }
  }
}

const std::vector<std::filesystem::path>& ModelManager::rootPaths() const { return rootPaths_; }

std::vector<ModelPack> ModelManager::scan() {
  availablePacks_.clear();

  std::vector<std::filesystem::path> scanRoots;
  scanRoots.reserve(rootPaths_.size() + 8);
  for (const auto& root : rootPaths_) {
    if (!root.empty()) {
      scanRoots.push_back(root);
    }
  }
  for (const auto& root : defaultRoots()) {
    scanRoots.push_back(root);
  }
  for (const auto& root : parseEnvRoots()) {
    scanRoots.push_back(root);
  }

  std::set<std::string> visitedRoots;
  std::unordered_set<std::string> seenPackIds;
  std::error_code error;
  for (const auto& root : scanRoots) {
    for (const auto& candidate : expandRootCandidates(root)) {
      error.clear();
      const auto absoluteRoot = std::filesystem::absolute(candidate, error);
      if (error) {
        continue;
      }
      const auto rootKey = toLower(absoluteRoot.string());
      if (!visitedRoots.insert(rootKey).second) {
        continue;
      }
      if (!std::filesystem::exists(absoluteRoot, error) || error) {
        continue;
      }

      std::filesystem::recursive_directory_iterator iterator(
          absoluteRoot,
          std::filesystem::directory_options::skip_permission_denied,
          error);
      if (error) {
        continue;
      }

      for (const auto& entry : iterator) {
        if (!entry.is_regular_file()) {
          continue;
        }
        if (!isModelMetadataFile(entry.path())) {
          continue;
        }
        const auto packDir = entry.path().parent_path();
        const auto pack = loader_.load(packDir);
        if (!pack.has_value()) {
          continue;
        }
        if (seenPackIds.insert(pack->id).second) {
          availablePacks_.push_back(pack.value());
        }
      }
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
