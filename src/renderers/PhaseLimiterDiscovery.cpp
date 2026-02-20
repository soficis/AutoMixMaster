#include "renderers/PhaseLimiterDiscovery.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <vector>

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
  const char* rawValue = std::getenv(key);
  if (rawValue == nullptr) {
    return std::nullopt;
  }
  std::string value(rawValue);
#endif
  value = trim(value);
  if (value.empty()) {
    return std::nullopt;
  }
  return value;
}

bool flagEnabled(const char* key) {
  const auto value = readEnvironment(key);
  if (!value.has_value()) {
    return false;
  }
  const auto lower = toLower(value.value());
  return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

std::filesystem::path cacheToolsRoot() {
  const auto appDataDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
  const auto base = std::filesystem::path(appDataDir.getFullPathName().toStdString());
  return base / "AutoMixMaster" / "tools";
}

std::filesystem::path cacheInstallRoot() {
  return cacheToolsRoot() / "phaselimiter";
}

std::filesystem::path cacheBinaryPath() {
#if defined(_WIN32)
  return cacheInstallRoot() / "bin" / "phase_limiter.exe";
#else
  return cacheInstallRoot() / "bin" / "phase_limiter";
#endif
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
  const auto envValue = readEnvironment("PHASELIMITER_BIN");
  if (!envValue.has_value()) {
    return std::nullopt;
  }

  const std::filesystem::path path(envValue.value());
  if (const auto info = toBinaryInfo(path); info.has_value()) {
    return info;
  }
  if (const auto info = scanDirectoryShallow(path); info.has_value()) {
    return info;
  }
  return scanDirectoryRecursive(path, 4);
}

bool downloadToFile(const std::string& url, const std::filesystem::path& outputPath) {
  int statusCode = 0;
  const auto options =
      juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
          .withConnectionTimeoutMs(45000)
          .withNumRedirectsToFollow(6)
          .withStatusCode(&statusCode);
  const auto stream = juce::URL(url).createInputStream(options);
  if (stream == nullptr || statusCode >= 400) {
    return false;
  }

  std::error_code error;
  std::filesystem::create_directories(outputPath.parent_path(), error);
  if (error) {
    return false;
  }

  juce::File output(outputPath.string());
  auto outputStream = output.createOutputStream();
  if (outputStream == nullptr || !outputStream->openedOk()) {
    return false;
  }

  outputStream->writeFromInputStream(*stream, -1);
  outputStream->flush();
  if (!isRegularFile(outputPath)) {
    return false;
  }

#if !defined(_WIN32)
  const auto currentPermissions = std::filesystem::status(outputPath, error).permissions();
  if (!error) {
    constexpr auto executeFlags = std::filesystem::perms::owner_exec |
                                  std::filesystem::perms::group_exec |
                                  std::filesystem::perms::others_exec;
    std::filesystem::permissions(outputPath, currentPermissions | executeFlags, error);
  }
#endif

  return true;
}

std::optional<std::string> defaultDownloadUrl() {
  if (const auto manual = readEnvironment("AUTOMIX_PHASELIMITER_DOWNLOAD_URL"); manual.has_value()) {
    return manual;
  }

#if defined(_WIN32)
  return std::string("https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/phaselimiter-win.zip");
#elif defined(__linux__)
  return std::string("https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/release.tar.xz");
#else
  return std::nullopt;
#endif
}

bool isSafeArchivePath(const std::filesystem::path& relativePath) {
  if (relativePath.empty() || relativePath.is_absolute()) {
    return false;
  }
  for (const auto& part : relativePath) {
    if (part == "..") {
      return false;
    }
  }
  return true;
}

bool extractZipArchive(const std::filesystem::path& archivePath, const std::filesystem::path& destinationRoot) {
  juce::ZipFile zipFile(juce::File(archivePath.string()));
  const int entryCount = zipFile.getNumEntries();
  if (entryCount <= 0) {
    return false;
  }

  for (int index = 0; index < entryCount; ++index) {
    const auto* entry = zipFile.getEntry(index);
    if (entry == nullptr || entry->filename.isEmpty()) {
      continue;
    }

    const auto normalized = std::filesystem::path(entry->filename.toStdString()).lexically_normal();
    if (!isSafeArchivePath(normalized)) {
      continue;
    }

    juce::File outputFile((destinationRoot / normalized).string());
    if (entry->filename.endsWithChar('/')) {
      outputFile.createDirectory();
      continue;
    }

    outputFile.getParentDirectory().createDirectory();
    auto stream = zipFile.createStreamForEntry(index);
    if (stream == nullptr) {
      return false;
    }
    auto outputStream = outputFile.createOutputStream();
    if (outputStream == nullptr || !outputStream->openedOk()) {
      return false;
    }
    outputStream->writeFromInputStream(*stream, -1);
    outputStream->flush();
  }

  return true;
}

bool extractTarXzArchive(const std::filesystem::path& archivePath, const std::filesystem::path& destinationRoot) {
#if defined(_WIN32)
  (void)archivePath;
  (void)destinationRoot;
  return false;
#else
  juce::StringArray command;
  command.add("tar");
  command.add("-xJf");
  command.add(archivePath.string());
  command.add("-C");
  command.add(destinationRoot.string());

  juce::ChildProcess process;
  if (!process.start(command)) {
    return false;
  }
  if (!process.waitForProcessToFinish(180000)) {
    process.kill();
    return false;
  }
  return process.getExitCode() == 0;
#endif
}

std::optional<PhaseLimiterBinaryInfo> resolveFromCacheInstall() {
  if (const auto info = toBinaryInfo(cacheBinaryPath()); info.has_value()) {
    return info;
  }

  if (const auto info = scanDirectoryShallow(cacheInstallRoot() / "bin"); info.has_value()) {
    return info;
  }

  return scanDirectoryRecursive(cacheInstallRoot(), 6);
}

std::optional<PhaseLimiterBinaryInfo> resolveFromAutoDownload() {
  if (flagEnabled("AUTOMIX_PHASELIMITER_SKIP_DOWNLOAD")) {
    return std::nullopt;
  }

  const auto url = defaultDownloadUrl();
  if (!url.has_value()) {
    return std::nullopt;
  }

  if (const auto cachedInfo = resolveFromCacheInstall(); cachedInfo.has_value()) {
    return cachedInfo;
  }

  static std::mutex downloadMutex;
  static bool attemptedDownload = false;
  std::scoped_lock lock(downloadMutex);

  if (const auto cachedInfo = resolveFromCacheInstall(); cachedInfo.has_value()) {
    return cachedInfo;
  }
  if (attemptedDownload) {
    return std::nullopt;
  }
  attemptedDownload = true;

  std::error_code error;
  std::filesystem::create_directories(cacheToolsRoot(), error);
  if (error) {
    return std::nullopt;
  }

  const auto lowerUrl = toLower(url.value());
  const std::filesystem::path archivePath = cacheToolsRoot() / "phaselimiter_download";

  if (lowerUrl.ends_with(".zip")) {
    const auto zipPath = archivePath.string() + ".zip";
    if (!downloadToFile(url.value(), zipPath)) {
      return std::nullopt;
    }
    std::filesystem::remove_all(cacheInstallRoot(), error);
    if (!extractZipArchive(zipPath, cacheToolsRoot())) {
      return std::nullopt;
    }
  } else if (lowerUrl.ends_with(".tar.xz") || lowerUrl.ends_with(".txz")) {
    const auto tarPath = archivePath.string() + ".tar.xz";
    if (!downloadToFile(url.value(), tarPath)) {
      return std::nullopt;
    }
    std::filesystem::remove_all(cacheInstallRoot(), error);
    if (!extractTarXzArchive(tarPath, cacheToolsRoot())) {
      return std::nullopt;
    }
  } else if (lowerUrl.ends_with(".sh")) {
    if (!flagEnabled("AUTOMIX_PHASELIMITER_ALLOW_LINUX_INSTALL_SCRIPT")) {
      return std::nullopt;
    }
#if defined(__linux__)
    const auto scriptPath = archivePath.string() + ".sh";
    if (!downloadToFile(url.value(), scriptPath)) {
      return std::nullopt;
    }

    juce::StringArray command;
    command.add("bash");
    command.add(scriptPath);
    juce::ChildProcess process;
    if (!process.start(command) || !process.waitForProcessToFinish(180000) || process.getExitCode() != 0) {
      return std::nullopt;
    }
#else
    return std::nullopt;
#endif
  } else {
    std::filesystem::create_directories(cacheInstallRoot() / "bin", error);
    if (error || !downloadToFile(url.value(), cacheBinaryPath())) {
      return std::nullopt;
    }
  }

  if (const auto downloadedInfo = resolveFromCacheInstall(); downloadedInfo.has_value()) {
    return downloadedInfo;
  }
  return std::nullopt;
}

std::vector<std::filesystem::path> defaultRoots() {
  std::set<std::filesystem::path> roots;
  roots.insert(std::filesystem::current_path());

  const juce::File executable = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  const std::filesystem::path executableDir(executable.getParentDirectory().getFullPathName().toStdString());
  roots.insert(executableDir);
  roots.insert(executableDir / ".." / "Resources");
  if (const auto assetRoot = readEnvironment("AUTOMIX_ASSET_ROOT"); assetRoot.has_value()) {
    roots.insert(assetRoot.value());
  }

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
  if (const auto cached = toBinaryInfo(cacheBinaryPath()); cached.has_value()) {
    return cached;
  }
  if (const auto fromLocal = findInRoots(defaultRoots()); fromLocal.has_value()) {
    return fromLocal;
  }
  return resolveFromAutoDownload();
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
