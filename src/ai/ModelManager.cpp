#include "ai/ModelManager.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <initializer_list>
#include <set>
#include <sstream>
#include <unordered_set>

#include "util/StringUtils.h"

namespace automix::ai {
namespace {

using ::automix::util::toLower;

std::vector<std::filesystem::path> defaultRoots() {
  return {
      "ModelPacks",
      "modelhub",
      "assets/modelhub",
      "assets/ModelHub",
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

  constexpr char kPathDelimiter =
#if defined(_WIN32)
      ';';
#else
      ':';
#endif

  std::stringstream stream(raw);
  std::string token;
  while (std::getline(stream, token, kPathDelimiter)) {
    if (!token.empty()) {
      roots.emplace_back(token);
    }
  }
  return roots;
}

bool isModelMetadataFile(const std::filesystem::path& path) {
  return toLower(path.filename().string()) == "model.json";
}

bool containsAny(const std::string& haystack, std::initializer_list<const char*> needles) {
  for (const auto* needle : needles) {
    if (haystack.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void applyLegacyTaskScopeFix(ModelPack& pack) {
  if (toLower(pack.taskScope) != "analysis") {
    return;
  }

  std::string joined;
  joined.reserve(pack.id.size() + pack.name.size() + pack.modelFile.size() + 3);
  joined += toLower(pack.id);
  joined += "|";
  joined += toLower(pack.name);
  joined += "|";
  joined += toLower(pack.modelFile);

  const bool looksLikeSeparation = containsAny(
      joined,
      {"demucs", "mdx", "roformer", "source-separation", "stem-separation", "separator"});

  if (looksLikeSeparation) {
    pack.taskScope = "separation";
  }
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
        auto normalizedPack = pack.value();
        applyLegacyTaskScopeFix(normalizedPack);
        if (seenPackIds.insert(normalizedPack.id).second) {
          availablePacks_.push_back(std::move(normalizedPack));
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
