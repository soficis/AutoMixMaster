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
#include "util/Sha256.h"
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
  return {"phase_limiter", "phaselimiter", "phase_limiter.bin", "phaselimiter.bin"};
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

} // namespace

std::string currentPhaseLimiterPlatformKey() {
#if defined(_WIN32)
  #if defined(_M_X64) || defined(__x86_64__)
    return "windows-x64";
  #elif defined(_M_ARM64) || defined(__aarch64__)
    return "windows-arm64";
  #else
    return "windows-x86";
  #endif
#elif defined(__APPLE__)
  #if defined(__aarch64__) || defined(__arm64__)
    return "macos-arm64";
  #elif defined(__x86_64__)
    return "macos-x86_64";
  #else
    return "macos-unknown";
  #endif
#elif defined(__linux__)
  #if defined(__x86_64__)
    return "linux-x64";
  #elif defined(__aarch64__)
    return "linux-arm64";
  #else
    return "linux-unknown";
  #endif
#else
  return "unknown";
#endif
}

std::vector<PhaseLimiterDownloadPin> phaseLimiterDownloadPinTable() {
  return {
      {"windows-x64",
       "https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/phaselimiter-win.zip",
       "cab2d30ad8d993a383749b30d9f6dc1911198d3aa309d5f631b70206e0162145"},
      {"linux-x64",
       "https://github.com/ai-mastering/phaselimiter/releases/download/v0.2.0/release.tar.xz",
       "0b382ba78b030926f706345d1b00d5f45890ba384c8a4e960320e3ee992bd163"},
      {"macos-arm64",
       "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native1/phaselimiter-0.2.0-macos-arm64.tar.xz",
       ""},
      {"macos-x86_64",
       "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native1/phaselimiter-0.2.0-macos-x86_64.tar.xz",
       ""},
      {"linux-arm64",
       "https://github.com/soficis/phaselimiter/releases/download/v0.2.0-native1/phaselimiter-0.2.0-linux-arm64.tar.xz",
       ""},
  };
}

std::optional<PhaseLimiterDownloadPin> defaultPhaseLimiterDownloadPin() {
  if (const auto manual = readEnvironment("AUTOMIX_PHASELIMITER_DOWNLOAD_URL"); manual.has_value()) {
    PhaseLimiterDownloadPin pin;
    pin.platformKey = currentPhaseLimiterPlatformKey();
    pin.url = manual.value();
    if (const auto manualSha = readEnvironment("AUTOMIX_PHASELIMITER_DOWNLOAD_SHA256"); manualSha.has_value()) {
      pin.sha256 = manualSha.value();
    }
    return pin;
  }

  const auto currentKey = currentPhaseLimiterPlatformKey();
  const auto table = phaseLimiterDownloadPinTable();
  for (const auto& pin : table) {
    if (pin.platformKey == currentKey) {
      if (pin.url.empty()) {
        return std::nullopt;
      }
      return pin;
    }
  }
  return std::nullopt;
}

namespace {

static std::optional<PhaseLimiterDiscovery::DownloadFetcher> gDownloadFetcher;
static bool gAttemptedDownload = false;
static std::mutex gDownloadMutex;

bool fetchArchive(const std::string& url, const std::filesystem::path& destination) {
  if (gDownloadFetcher.has_value()) {
    return (*gDownloadFetcher)(url, destination);
  }
  return downloadToFile(url, destination);
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

// Ordered by priority: AUTOMIX_ASSET_ROOT is an override, so it is searched
// first. (A std::set here once sorted roots lexicographically, which let the
// executable's own tree win whenever its path sorted before the override.)
std::vector<std::filesystem::path> defaultRoots() {
  std::vector<std::filesystem::path> candidates;
  if (const auto assetRoot = readEnvironment("AUTOMIX_ASSET_ROOT"); assetRoot.has_value()) {
    candidates.emplace_back(assetRoot.value());
  }
  candidates.push_back(std::filesystem::current_path());

  const juce::File executable = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  const std::filesystem::path executableDir(executable.getParentDirectory().getFullPathName().toStdString());
  candidates.push_back(executableDir);
  candidates.push_back(executableDir / ".." / "Resources");

  std::set<std::filesystem::path> seen;
  std::vector<std::filesystem::path> output;
  output.reserve(candidates.size());
  for (const auto& root : candidates) {
    if (seen.insert(root).second) {
      output.push_back(root);
    }
  }
  return output;
}

} // namespace

void PhaseLimiterDiscovery::setDownloadFetcherForTesting(DownloadFetcher fetcher) {
  gDownloadFetcher = std::move(fetcher);
}

void PhaseLimiterDiscovery::resetDownloadFetcherForTesting() {
  gDownloadFetcher.reset();
}

void PhaseLimiterDiscovery::resetAttemptedDownloadForTesting() {
  std::scoped_lock lock(gDownloadMutex);
  gAttemptedDownload = false;
}

std::optional<PhaseLimiterBinaryInfo> PhaseLimiterDiscovery::downloadAndInstall(
    const std::optional<PhaseLimiterDownloadPin>& customPin) {
  if (flagEnabled("AUTOMIX_PHASELIMITER_SKIP_DOWNLOAD")) {
    return std::nullopt;
  }

  const auto pin = customPin.has_value() ? customPin : defaultPhaseLimiterDownloadPin();
  if (!pin.has_value() || pin->url.empty()) {
    return std::nullopt;
  }

  if (const auto cachedInfo = resolveFromCacheInstall(); cachedInfo.has_value()) {
    return cachedInfo;
  }

  std::scoped_lock lock(gDownloadMutex);
  if (const auto cachedInfo = resolveFromCacheInstall(); cachedInfo.has_value()) {
    return cachedInfo;
  }
  if (gAttemptedDownload) {
    return std::nullopt;
  }
  gAttemptedDownload = true;

  std::error_code error;
  std::filesystem::create_directories(cacheToolsRoot(), error);
  if (error) {
    return std::nullopt;
  }

  const auto lowerUrl = toLower(pin->url);
  const std::filesystem::path archivePath = cacheToolsRoot() / "phaselimiter_download";
  std::filesystem::path destinationFile;
  bool isArchive = false;
  bool isZip = false;

  if (lowerUrl.ends_with(".zip")) {
    destinationFile = archivePath.string() + ".zip";
    isArchive = true;
    isZip = true;
  } else if (lowerUrl.ends_with(".tar.xz") || lowerUrl.ends_with(".txz")) {
    destinationFile = archivePath.string() + ".tar.xz";
    isArchive = true;
    isZip = false;
  } else {
    std::filesystem::create_directories(cacheInstallRoot() / "bin", error);
    if (error) {
      return std::nullopt;
    }
    destinationFile = cacheBinaryPath();
    isArchive = false;
  }

  if (!fetchArchive(pin->url, destinationFile)) {
    std::filesystem::remove(destinationFile, error);
    return std::nullopt;
  }

  // SHA-256 verification before extraction
  if (!pin->sha256.empty()) {
    const auto actualSha = toLower(automix::util::fileSha256(destinationFile));
    const auto expectedSha = toLower(pin->sha256);
    if (actualSha != expectedSha) {
      std::filesystem::remove(destinationFile, error);
      juce::Logger::writeToLog("PhaseLimiter download SHA-256 mismatch for " +
                               destinationFile.string() + " (expected " + expectedSha +
                               ", got " + (actualSha.empty() ? "unreadable" : actualSha) +
                               "); download discarded.");
      return std::nullopt;
    }
  } else {
    juce::Logger::writeToLog("WARNING: PhaseLimiter downloaded without SHA-256 verification (unpinned URL).");
  }

  if (isArchive) {
    std::filesystem::remove_all(cacheInstallRoot(), error);
    const bool extracted = isZip ? extractZipArchive(destinationFile, cacheToolsRoot())
                                 : extractTarXzArchive(destinationFile, cacheToolsRoot());
    std::filesystem::remove(destinationFile, error);
    if (!extracted) {
      return std::nullopt;
    }
  }

  if (const auto downloadedInfo = resolveFromCacheInstall(); downloadedInfo.has_value()) {
#if !defined(_WIN32)
    const auto currentPermissions = std::filesystem::status(downloadedInfo->executablePath, error).permissions();
    if (!error) {
      constexpr auto executeFlags = std::filesystem::perms::owner_exec |
                                    std::filesystem::perms::group_exec |
                                    std::filesystem::perms::others_exec;
      std::filesystem::permissions(downloadedInfo->executablePath, currentPermissions | executeFlags, error);
    }
#endif
    return downloadedInfo;
  }
  return std::nullopt;
}

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
  return downloadAndInstall();
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
