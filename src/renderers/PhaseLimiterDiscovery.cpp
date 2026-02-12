#include "renderers/PhaseLimiterDiscovery.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>

namespace automix::renderers {
namespace {

bool isRegularFile(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::vector<std::string> executableNames() {
#if defined(_WIN32)
  return {"phase_limiter.exe", "phaselimiter.exe", "phase_limiter", "phaselimiter"};
#else
  return {"phase_limiter", "phaselimiter", "phase_limiter.bin", "phaselimiter.bin", "phase_limiter.exe"};
#endif
}

bool isKnownExecutableName(const std::filesystem::path& path) {
  const auto lower = toLower(path.filename().string());
  const auto names = executableNames();
  return std::find(names.begin(), names.end(), lower) != names.end();
}

std::vector<std::string> platformDirectoryNames() {
#if defined(_WIN32)
  return {"", "windows", "win", "win64", "x64"};
#elif defined(__APPLE__)
  return {"", "mac", "macos", "darwin", "osx", "universal"};
#else
  return {"", "linux", "linux64", "x64"};
#endif
}

std::optional<PhaseLimiterBinaryInfo> toBinaryInfo(const std::filesystem::path& executablePath) {
  if (!isRegularFile(executablePath)) {
    return std::nullopt;
  }

  const auto absolutePath = std::filesystem::absolute(executablePath);
  const auto parent = absolutePath.parent_path();
  std::filesystem::path installRoot = parent;
  if (toLower(parent.filename().string()) == "bin" && parent.has_parent_path()) {
    installRoot = parent.parent_path();
  }

  return PhaseLimiterBinaryInfo{
      .executablePath = absolutePath,
      .installRoot = installRoot,
  };
}

std::optional<PhaseLimiterBinaryInfo> scanDirectoryShallow(const std::filesystem::path& directory) {
  const auto names = executableNames();
  for (const auto& platformDir : platformDirectoryNames()) {
    const auto candidateDir = platformDir.empty() ? directory : (directory / platformDir);
    for (const auto& name : names) {
      const auto candidate = candidateDir / name;
      if (const auto info = toBinaryInfo(candidate); info.has_value()) {
        return info;
      }
    }
  }
  return std::nullopt;
}

std::optional<PhaseLimiterBinaryInfo> scanDirectoryRecursive(const std::filesystem::path& directory, const int maxDepth) {
  std::error_code error;
  if (!std::filesystem::exists(directory, error) || error) {
    return std::nullopt;
  }

  std::filesystem::recursive_directory_iterator it(directory, error);
  std::filesystem::recursive_directory_iterator end;
  for (; it != end && !error; it.increment(error)) {
    if (it.depth() > maxDepth) {
      it.disable_recursion_pending();
      continue;
    }

    const auto path = it->path();
    if (!it->is_regular_file(error) || error) {
      continue;
    }
    if (!isKnownExecutableName(path)) {
      continue;
    }

    if (const auto info = toBinaryInfo(path); info.has_value()) {
      return info;
    }
  }

  return std::nullopt;
}

std::vector<std::filesystem::path> baseAssetDirectories(const std::filesystem::path& root) {
  return {
      root / "assets",
      root / "Assets",
      root / "resources" / "assets",
      root / "Resources" / "assets",
      root / "Contents" / "Resources" / "assets",
  };
}

std::vector<std::string> phaseLimiterDirectoryNames() {
  return {"phaselimiter", "phase_limiter", "PhaseLimiter", "phaseLimiter"};
}

std::optional<PhaseLimiterBinaryInfo> scanRoot(const std::filesystem::path& root) {
  const auto phaseDirs = phaseLimiterDirectoryNames();

  for (const auto& assetsDir : baseAssetDirectories(root)) {
    for (const auto& phaseName : phaseDirs) {
      const auto phaseRoot = assetsDir / phaseName;
      if (const auto info = scanDirectoryShallow(phaseRoot / "bin"); info.has_value()) {
        return info;
      }
      if (const auto info = scanDirectoryShallow(phaseRoot); info.has_value()) {
        return info;
      }
      if (const auto info = scanDirectoryRecursive(phaseRoot, 4); info.has_value()) {
        return info;
      }
    }
  }

  return std::nullopt;
}

std::optional<PhaseLimiterBinaryInfo> resolveFromEnvironment() {
  std::string envValue;
#if defined(_WIN32)
  char* buffer = nullptr;
  size_t length = 0;
  if (_dupenv_s(&buffer, &length, "PHASELIMITER_BIN") == 0 && buffer != nullptr) {
    envValue.assign(buffer, length > 0 ? length - 1 : 0);
    free(buffer);
  }
#else
  const char* value = std::getenv("PHASELIMITER_BIN");
  if (value != nullptr) {
    envValue = value;
  }
#endif
  if (envValue.empty()) {
    return std::nullopt;
  }

  const std::filesystem::path path(envValue);
  if (const auto info = toBinaryInfo(path); info.has_value()) {
    return info;
  }
  if (const auto info = scanDirectoryShallow(path); info.has_value()) {
    return info;
  }
  return scanDirectoryRecursive(path, 4);
}

std::vector<std::filesystem::path> defaultRoots() {
  std::set<std::filesystem::path> roots;
  roots.insert(std::filesystem::current_path());

  const juce::File executable = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  const std::filesystem::path executableDir(executable.getParentDirectory().getFullPathName().toStdString());
  roots.insert(executableDir);
  roots.insert(executableDir / ".." / "Resources");

  std::vector<std::filesystem::path> output;
  output.reserve(roots.size());
  for (const auto& root : roots) {
    output.push_back(root);
  }
  return output;
}

} // namespace

std::optional<PhaseLimiterBinaryInfo> PhaseLimiterDiscovery::find() const {
  if (const auto fromEnv = resolveFromEnvironment(); fromEnv.has_value()) {
    return fromEnv;
  }
  return findInRoots(defaultRoots());
}

std::optional<PhaseLimiterBinaryInfo> PhaseLimiterDiscovery::findInRoots(
    const std::vector<std::filesystem::path>& roots) const {
  for (const auto& root : roots) {
    auto current = root;
    for (int depth = 0; depth < 12; ++depth) {
      if (const auto info = scanRoot(current); info.has_value()) {
        return info;
      }
      if (!current.has_parent_path()) {
        break;
      }
      current = current.parent_path();
    }
  }
  return std::nullopt;
}

} // namespace automix::renderers
