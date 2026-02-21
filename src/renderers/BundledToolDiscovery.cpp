#include "renderers/BundledToolDiscovery.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <set>
#include <sstream>

#include <juce_core/juce_core.h>

#include "util/FileUtils.h"
#include "util/StringUtils.h"

namespace automix::renderers {
namespace {

using ::automix::util::isRegularFile;
using ::automix::util::toLower;
using ::automix::util::trim;

std::optional<std::string> readEnvironment(const char* key) {
#if defined(_WIN32)
  char* buffer = nullptr;
  size_t length = 0;
  if (_dupenv_s(&buffer, &length, key) != 0 || buffer == nullptr) {
    return std::nullopt;
  }
  std::string value(buffer, length > 0 ? length - 1 : 0);
  free(buffer);
#else
  const char* raw = std::getenv(key);
  if (raw == nullptr) {
    return std::nullopt;
  }
  std::string value(raw);
#endif
  value = trim(value);
  if (value.empty()) {
    return std::nullopt;
  }
  return value;
}

std::vector<std::string> platformSubdirectories() {
#if defined(_WIN32)
  return {"", "bin", "windows", "win", "win64", "x64", "x86_64", "arm64", "aarch64"};
#elif defined(__APPLE__)
  return {"", "bin", "mac", "macos", "darwin", "osx", "universal", "x64", "arm64"};
#else
  return {"", "bin", "linux", "linux64", "x64", "x86_64", "arm64", "aarch64"};
#endif
}

std::optional<BundledToolBinaryInfo> toBinaryInfo(const std::filesystem::path& executablePath) {
  if (!isRegularFile(executablePath)) {
    return std::nullopt;
  }

  const auto absolutePath = std::filesystem::absolute(executablePath);
  const auto parent = absolutePath.parent_path();
  auto installRoot = parent;
  if (toLower(parent.filename().string()) == "bin" && parent.has_parent_path()) {
    installRoot = parent.parent_path();
  }

  return BundledToolBinaryInfo{
      .executablePath = absolutePath,
      .installRoot = installRoot,
  };
}

std::optional<BundledToolBinaryInfo> scanDirectoryShallow(const std::filesystem::path& directory,
                                                          const std::vector<std::string>& executableNames) {
  for (const auto& subdirectory : platformSubdirectories()) {
    const auto candidateDirectory = subdirectory.empty() ? directory : (directory / subdirectory);
    for (const auto& executableName : executableNames) {
      if (const auto info = toBinaryInfo(candidateDirectory / executableName); info.has_value()) {
        return info;
      }
    }
  }
  return std::nullopt;
}

std::optional<BundledToolBinaryInfo> scanDirectoryRecursive(const std::filesystem::path& directory,
                                                            const std::vector<std::string>& executableNames,
                                                            const int maxDepth) {
  std::error_code error;
  if (!std::filesystem::exists(directory, error) || error) {
    return std::nullopt;
  }

  std::set<std::string> normalizedNames;
  for (const auto& name : executableNames) {
    normalizedNames.insert(toLower(name));
  }

  std::filesystem::recursive_directory_iterator iterator(
      directory,
      std::filesystem::directory_options::skip_permission_denied,
      error);
  std::filesystem::recursive_directory_iterator end;

  for (; iterator != end && !error; iterator.increment(error)) {
    if (iterator.depth() > maxDepth) {
      iterator.disable_recursion_pending();
      continue;
    }
    if (!iterator->is_regular_file(error) || error) {
      continue;
    }

    const auto fileName = toLower(iterator->path().filename().string());
    if (normalizedNames.find(fileName) == normalizedNames.end()) {
      continue;
    }

    if (const auto info = toBinaryInfo(iterator->path()); info.has_value()) {
      return info;
    }
  }

  return std::nullopt;
}

std::optional<BundledToolBinaryInfo> scanAssetsRoot(const std::filesystem::path& root,
                                                    const BundledToolDiscoverySpec& spec) {
  const std::array<std::filesystem::path, 6> assetRoots = {
      root / "assets",
      root / "Assets",
      root / "resources" / "assets",
      root / "Resources" / "assets",
      root / "Contents" / "Resources" / "assets",
      root / "resources" / "Assets",
  };

  for (const auto& assetRoot : assetRoots) {
    for (const auto& assetDirectory : spec.assetDirectoryNames) {
      const std::array<std::filesystem::path, 4> candidateDirs = {
          assetRoot / assetDirectory,
          assetRoot / "renderers" / assetDirectory,
          assetRoot / "limiters" / assetDirectory,
          assetRoot / "tools" / assetDirectory,
      };

      for (const auto& candidateDir : candidateDirs) {
        if (const auto info = scanDirectoryShallow(candidateDir, spec.executableNames); info.has_value()) {
          return info;
        }
        if (const auto info = scanDirectoryRecursive(candidateDir, spec.executableNames, 4); info.has_value()) {
          return info;
        }
      }
    }
  }

  return std::nullopt;
}

std::vector<std::filesystem::path> defaultRoots() {
  std::set<std::filesystem::path> roots;
  std::error_code error;
  roots.insert(std::filesystem::absolute(std::filesystem::current_path(error), error));

  const juce::File executable = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  if (executable.existsAsFile()) {
    const auto executableDirectory = std::filesystem::path(executable.getParentDirectory().getFullPathName().toStdString());
    roots.insert(executableDirectory);
    roots.insert(executableDirectory / ".." / "Resources");
  }

  if (const auto assetRoot = readEnvironment("AUTOMIX_ASSET_ROOT"); assetRoot.has_value()) {
    roots.insert(std::filesystem::path(assetRoot.value()));
  }

  std::vector<std::filesystem::path> result;
  result.reserve(roots.size());
  for (const auto& root : roots) {
    result.push_back(root);
  }
  return result;
}

std::optional<BundledToolBinaryInfo> scanPathEnvironment(const BundledToolDiscoverySpec& spec) {
  const char* rawPath = std::getenv("PATH");
  if (rawPath == nullptr || *rawPath == '\0') {
    return std::nullopt;
  }

#if defined(_WIN32)
  constexpr char delimiter = ';';
#else
  constexpr char delimiter = ':';
#endif

  std::stringstream stream(rawPath);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    const auto pathToken = trim(token);
    if (pathToken.empty()) {
      continue;
    }

    for (const auto& executableName : spec.executableNames) {
      if (const auto info = toBinaryInfo(std::filesystem::path(pathToken) / executableName); info.has_value()) {
        return info;
      }
    }
  }

  return std::nullopt;
}

} // namespace

std::optional<BundledToolBinaryInfo> BundledToolDiscovery::find(const BundledToolDiscoverySpec& spec) {
  if (spec.executableNames.empty() || spec.assetDirectoryNames.empty()) {
    return std::nullopt;
  }

  if (!spec.environmentVariable.empty()) {
    if (const auto value = readEnvironment(spec.environmentVariable.c_str()); value.has_value()) {
      const std::filesystem::path candidate(value.value());
      if (const auto info = toBinaryInfo(candidate); info.has_value()) {
        return info;
      }
      if (const auto info = scanDirectoryShallow(candidate, spec.executableNames); info.has_value()) {
        return info;
      }
      if (const auto info = scanDirectoryRecursive(candidate, spec.executableNames, 4); info.has_value()) {
        return info;
      }
    }
  }

  for (const auto& baseRoot : defaultRoots()) {
    auto current = baseRoot;
    for (int depth = 0; depth < 8; ++depth) {
      if (const auto info = scanAssetsRoot(current, spec); info.has_value()) {
        return info;
      }
      if (!current.has_parent_path()) {
        break;
      }
      const auto parent = current.parent_path();
      if (parent == current) {
        break;
      }
      current = parent;
    }
  }

  return scanPathEnvironment(spec);
}

} // namespace automix::renderers
