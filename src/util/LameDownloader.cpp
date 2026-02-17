#include "util/LameDownloader.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include <juce_core/juce_core.h>

#include "util/FileUtils.h"
#include "util/StringUtils.h"

namespace automix::util {
namespace {

using ::automix::util::toLower;
using ::automix::util::trim;
using ::automix::util::isRegularFile;

constexpr const char* kDefaultLameVersion = "3.100";
constexpr int kDownloadTimeoutMs = 180000;

enum class SourceType {
  DirectBinary,
  Zip,
  Debian,
  Ghcr,
};

struct DownloadSource {
  SourceType type = SourceType::Zip;
  std::string url;
  std::string ghcrOs;
  std::string ghcrArch;
};

struct TempDirectory {
  explicit TempDirectory(const std::string& prefix) {
    const auto base =
        std::filesystem::path(juce::File::getSpecialLocation(juce::File::tempDirectory).getFullPathName().toStdString());
    const auto nonce = std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    path = base / (prefix + "_" + nonce);

    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
      throw std::runtime_error("Failed to create temp directory: " + path.string());
    }
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }

  std::filesystem::path path;
};

std::string joinLines(const std::vector<std::string>& lines) {
  std::ostringstream os;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (i != 0) {
      os << '\n';
    }
    os << lines[i];
  }
  return os.str();
}

std::optional<std::string> readEnvironment(const char* key) {
#if defined(_WIN32)
  char* buffer = nullptr;
  size_t length = 0;
  if (_dupenv_s(&buffer, &length, key) != 0 || buffer == nullptr) {
    return std::nullopt;
  }

  std::string value(buffer, length > 0 ? length - 1 : 0);
  free(buffer);
  value = trim(value);
  if (value.empty()) {
    return std::nullopt;
  }
  return value;
#else
  const char* value = std::getenv(key);
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  const auto trimmed = trim(value);
  if (trimmed.empty()) {
    return std::nullopt;
  }
  return trimmed;
#endif
}

bool flagEnabled(const char* key) {
  const auto value = readEnvironment(key);
  if (!value.has_value()) {
    return false;
  }

  const auto lower = toLower(value.value());
  return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

std::string binaryName() {
#if defined(_WIN32)
  return "lame.exe";
#else
  return "lame";
#endif
}

std::string platformKey() {
#if defined(_WIN32)
#if defined(_M_ARM64) || defined(__aarch64__)
  return "win32-arm64";
#elif defined(_M_IX86) || defined(__i386__)
  return "win32-ia32";
#else
  return "win32-x64";
#endif
#elif defined(__APPLE__)
#if defined(__aarch64__) || defined(__arm64__)
  return "darwin-arm64";
#else
  return "darwin-x64";
#endif
#elif defined(__linux__)
#if defined(__aarch64__)
  return "linux-arm64";
#elif defined(__arm__)
  return "linux-arm";
#else
  return "linux-x64";
#endif
#else
  return "";
#endif
}

std::vector<DownloadSource> platformSources() {
  const auto version = readEnvironment("AUTOMIX_LAME_VERSION").value_or(kDefaultLameVersion);
  if (const auto manualUrl = readEnvironment("AUTOMIX_LAME_DOWNLOAD_URL"); manualUrl.has_value()) {
    const auto lower = toLower(*manualUrl);
    if (lower.ends_with(".zip")) {
      return {{SourceType::Zip, *manualUrl, "", ""}};
    }
    if (lower.ends_with(".deb")) {
      return {{SourceType::Debian, *manualUrl, "", ""}};
    }
    return {{SourceType::DirectBinary, *manualUrl, "", ""}};
  }

  const auto key = platformKey();
  if (key == "win32-x64") {
    return {
        {SourceType::Zip, "https://www.rarewares.org/files/mp3/lame" + version + ".1-x64.zip", "", ""},
    };
  }
  if (key == "win32-ia32") {
    return {
        {SourceType::Zip, "https://www.rarewares.org/files/mp3/lame" + version + ".1-win32.zip", "", ""},
    };
  }
  if (key == "win32-arm64") {
    return {
        {SourceType::Zip, "https://www.rarewares.org/files/mp3/LAME-" + version + "-Win-ARM64.zip", "", ""},
    };
  }
  if (key == "linux-x64") {
    return {
        {SourceType::Debian, "https://deb.debian.org/debian/pool/main/l/lame/lame_" + version + "-6_amd64.deb", "", ""},
        {SourceType::Ghcr, "", "linux", "amd64"},
    };
  }
  if (key == "linux-arm64") {
    return {
        {SourceType::Debian, "https://deb.debian.org/debian/pool/main/l/lame/lame_" + version + "-6_arm64.deb", "", ""},
        {SourceType::Ghcr, "", "linux", "arm64"},
    };
  }
  if (key == "linux-arm") {
    return {
        {SourceType::Debian, "https://deb.debian.org/debian/pool/main/l/lame/lame_" + version + "-6_armhf.deb", "", ""},
    };
  }
  if (key == "darwin-x64") {
    return {
        {SourceType::Ghcr, "", "darwin", "amd64"},
    };
  }
  if (key == "darwin-arm64") {
    return {
        {SourceType::Ghcr, "", "darwin", "arm64"},
    };
  }
  return {};
}

bool runProcess(const juce::StringArray& command,
                const int timeoutMs,
                std::string* processOutput,
                std::string* detail) {
  juce::ChildProcess process;
  if (!process.start(command)) {
    if (detail != nullptr) {
      *detail = "Failed to start process: " + command.joinIntoString(" ").toStdString();
    }
    return false;
  }

  if (!process.waitForProcessToFinish(timeoutMs)) {
    process.kill();
    if (detail != nullptr) {
      *detail = "Timed out waiting for process: " + command.joinIntoString(" ").toStdString();
    }
    return false;
  }

  const auto output = process.readAllProcessOutput().toStdString();
  if (processOutput != nullptr) {
    *processOutput = output;
  }

  const auto exitCode = process.getExitCode();
  if (exitCode != 0) {
    if (detail != nullptr) {
      std::ostringstream os;
      os << "Process failed (exit=" << exitCode << "): " << command.joinIntoString(" ").toStdString();
      if (!output.empty()) {
        os << " output=" << output;
      }
      *detail = os.str();
    }
    return false;
  }
  return true;
}

bool downloadToFile(const std::string& url,
                    const std::filesystem::path& outputPath,
                    const std::string& extraHeaders,
                    std::string* detail) {
  int statusCode = 0;
  const auto baseOptions =
      juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
          .withConnectionTimeoutMs(45000)
          .withNumRedirectsToFollow(8)
          .withStatusCode(&statusCode);
  const auto input = juce::URL(url).createInputStream(
      extraHeaders.empty() ? baseOptions : baseOptions.withExtraHeaders(extraHeaders));
  if (input == nullptr) {
    if (detail != nullptr) {
      *detail = "Download failed (no stream): " + url;
    }
    return false;
  }
  if (statusCode >= 400) {
    if (detail != nullptr) {
      *detail = "Download failed (HTTP " + std::to_string(statusCode) + "): " + url;
    }
    return false;
  }

  std::error_code error;
  std::filesystem::create_directories(outputPath.parent_path(), error);
  if (error) {
    if (detail != nullptr) {
      *detail = "Failed creating output folder: " + outputPath.parent_path().string();
    }
    return false;
  }

  juce::File outFile(outputPath.string());
  std::unique_ptr<juce::FileOutputStream> out(outFile.createOutputStream());
  if (out == nullptr || !out->openedOk()) {
    if (detail != nullptr) {
      *detail = "Failed opening output file: " + outputPath.string();
    }
    return false;
  }

  out->writeFromInputStream(*input, -1);
  out->flush();
  if (!isRegularFile(outputPath)) {
    if (detail != nullptr) {
      *detail = "Downloaded file missing after transfer: " + outputPath.string();
    }
    return false;
  }

  return true;
}

std::optional<nlohmann::json> fetchJson(const std::string& url,
                                        const std::string& extraHeaders,
                                        std::string* detail) {
  int statusCode = 0;
  const auto baseOptions =
      juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
          .withConnectionTimeoutMs(45000)
          .withNumRedirectsToFollow(8)
          .withStatusCode(&statusCode);
  const auto input = juce::URL(url).createInputStream(
      extraHeaders.empty() ? baseOptions : baseOptions.withExtraHeaders(extraHeaders));
  if (input == nullptr) {
    if (detail != nullptr) {
      *detail = "JSON fetch failed (no stream): " + url;
    }
    return std::nullopt;
  }
  if (statusCode >= 400) {
    if (detail != nullptr) {
      *detail = "JSON fetch failed (HTTP " + std::to_string(statusCode) + "): " + url;
    }
    return std::nullopt;
  }

  const auto text = input->readEntireStreamAsString().toStdString();
  if (text.empty()) {
    if (detail != nullptr) {
      *detail = "JSON fetch returned empty body: " + url;
    }
    return std::nullopt;
  }

  try {
    return nlohmann::json::parse(text);
  } catch (const std::exception& error) {
    if (detail != nullptr) {
      *detail = "JSON parse failed: " + std::string(error.what());
    }
    return std::nullopt;
  }
}

std::optional<std::filesystem::path> findFileRecursive(const std::filesystem::path& root,
                                                       const std::vector<std::string>& names) {
  std::error_code error;
  if (!std::filesystem::exists(root, error) || error) {
    return std::nullopt;
  }

  std::filesystem::recursive_directory_iterator it(root, std::filesystem::directory_options::skip_permission_denied, error);
  std::filesystem::recursive_directory_iterator end;
  for (; it != end && !error; it.increment(error)) {
    if (!it->is_regular_file(error) || error) {
      continue;
    }

    const auto fileName = toLower(it->path().filename().string());
    for (const auto& candidateName : names) {
      if (fileName == toLower(candidateName)) {
        return it->path();
      }
    }
  }

  return std::nullopt;
}

bool ensureExecutable(const std::filesystem::path& path, std::string* detail) {
  std::error_code error;
  if (!isRegularFile(path)) {
    if (detail != nullptr) {
      *detail = "File is not present: " + path.string();
    }
    return false;
  }

#if !defined(_WIN32)
  const auto perms = std::filesystem::status(path, error).permissions();
  if (!error) {
    std::filesystem::permissions(path,
                                 perms | std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec |
                                     std::filesystem::perms::others_exec,
                                 std::filesystem::perm_options::replace,
                                 error);
  }
  if (error && detail != nullptr) {
    *detail = "Failed to set executable permissions: " + path.string();
    return false;
  }
#endif

  juce::StringArray versionCommand;
  versionCommand.add(path.string());
  versionCommand.add("--version");

  std::string output;
  std::string versionDetail;
  if (!runProcess(versionCommand, 15000, &output, &versionDetail)) {
    if (detail != nullptr) {
      *detail = "Downloaded binary failed validation. " + versionDetail;
    }
    return false;
  }

  if (toLower(output).find("lame") == std::string::npos && detail != nullptr) {
    *detail = "Binary validation succeeded without 'lame' in --version output.";
  }

  return true;
}

bool copyBinaryToCache(const std::filesystem::path& source, const std::filesystem::path& destination, std::string* detail) {
  std::error_code error;
  std::filesystem::create_directories(destination.parent_path(), error);
  if (error) {
    if (detail != nullptr) {
      *detail = "Failed creating cache directory: " + destination.parent_path().string();
    }
    return false;
  }

  std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing, error);
  if (error) {
    if (detail != nullptr) {
      *detail = "Failed copying binary to cache: " + destination.string();
    }
    return false;
  }

  return ensureExecutable(destination, detail);
}

std::optional<std::filesystem::path> extractArMember(const std::filesystem::path& archivePath,
                                                      const std::string& memberPrefix,
                                                      const std::filesystem::path& outputDirectory,
                                                      std::string* detail) {
  std::ifstream input(archivePath, std::ios::binary);
  if (!input.is_open()) {
    if (detail != nullptr) {
      *detail = "Unable to open .deb archive: " + archivePath.string();
    }
    return std::nullopt;
  }

  char magic[8] = {};
  input.read(magic, sizeof(magic));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(magic)) || std::string(magic, sizeof(magic)) != "!<arch>\n") {
    if (detail != nullptr) {
      *detail = "Invalid ar archive header: " + archivePath.string();
    }
    return std::nullopt;
  }

  while (true) {
    char header[60] = {};
    input.read(header, sizeof(header));
    if (input.eof()) {
      break;
    }
    if (!input.good()) {
      if (detail != nullptr) {
        *detail = "Failed reading ar header from: " + archivePath.string();
      }
      return std::nullopt;
    }

    std::string name = trim(std::string(header, 16));
    if (!name.empty() && name.back() == '/') {
      name.pop_back();
    }

    const auto sizeText = trim(std::string(header + 48, 10));
    std::int64_t size = 0;
    try {
      size = std::stoll(sizeText);
    } catch (...) {
      if (detail != nullptr) {
        *detail = "Failed parsing ar member size for: " + name;
      }
      return std::nullopt;
    }
    if (size < 0) {
      if (detail != nullptr) {
        *detail = "Encountered negative ar member size for: " + name;
      }
      return std::nullopt;
    }

    if (name.rfind(memberPrefix, 0) == 0) {
      std::vector<char> payload(static_cast<size_t>(size));
      input.read(payload.data(), static_cast<std::streamsize>(payload.size()));
      if (!input.good()) {
        if (detail != nullptr) {
          *detail = "Failed reading ar member payload for: " + name;
        }
        return std::nullopt;
      }

      std::error_code error;
      std::filesystem::create_directories(outputDirectory, error);
      if (error) {
        if (detail != nullptr) {
          *detail = "Failed creating extraction directory: " + outputDirectory.string();
        }
        return std::nullopt;
      }

      const auto outputPath = outputDirectory / name;
      std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
      if (!output.is_open()) {
        if (detail != nullptr) {
          *detail = "Failed writing extracted member: " + outputPath.string();
        }
        return std::nullopt;
      }
      output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
      output.flush();
      if (!output.good()) {
        if (detail != nullptr) {
          *detail = "Failed flushing extracted member: " + outputPath.string();
        }
        return std::nullopt;
      }

      return outputPath;
    }

    input.seekg(size, std::ios::cur);
    if (!input.good()) {
      if (detail != nullptr) {
        *detail = "Failed skipping ar member: " + name;
      }
      return std::nullopt;
    }

    if ((size % 2) != 0) {
      input.seekg(1, std::ios::cur);
      if (!input.good()) {
        if (detail != nullptr) {
          *detail = "Failed skipping ar padding byte.";
        }
        return std::nullopt;
      }
    }
  }

  if (detail != nullptr) {
    *detail = "No data.tar member found in archive: " + archivePath.string();
  }
  return std::nullopt;
}

bool installFromZip(const DownloadSource& source, const std::filesystem::path& targetBinary, std::string* detail) {
  TempDirectory temp("automix_lame_zip");
  const auto archivePath = temp.path / "lame.zip";
  if (!downloadToFile(source.url, archivePath, "", detail)) {
    return false;
  }

  juce::ZipFile zip(juce::File(archivePath.string()));
  if (zip.getNumEntries() <= 0) {
    if (detail != nullptr) {
      *detail = "ZIP archive appears empty: " + source.url;
    }
    return false;
  }

  const auto extractPath = temp.path / "extract";
  std::error_code error;
  std::filesystem::create_directories(extractPath, error);
  if (error) {
    if (detail != nullptr) {
      *detail = "Failed creating ZIP extract directory: " + extractPath.string();
    }
    return false;
  }

  const auto unzipResult = zip.uncompressTo(juce::File(extractPath.string()), true);
  if (unzipResult.failed()) {
    if (detail != nullptr) {
      *detail = "Failed extracting ZIP archive: " + unzipResult.getErrorMessage().toStdString();
    }
    return false;
  }

  const auto foundBinary = findFileRecursive(extractPath, {binaryName(), "lame"});
  if (!foundBinary.has_value()) {
    if (detail != nullptr) {
      *detail = "No LAME executable found in ZIP archive: " + source.url;
    }
    return false;
  }

  return copyBinaryToCache(*foundBinary, targetBinary, detail);
}

bool extractTarArchive(const std::filesystem::path& archivePath,
                       const std::filesystem::path& outputDirectory,
                       const std::string& compressionFlag,
                       const std::string& memberHint,
                       std::string* detail) {
  juce::StringArray command;
  command.add("tar");
  command.add("-x" + compressionFlag + "f");
  command.add(archivePath.string());
  command.add("-C");
  command.add(outputDirectory.string());
  if (!memberHint.empty()) {
    command.add(memberHint);
  }
  return runProcess(command, kDownloadTimeoutMs, nullptr, detail);
}

bool installFromDebian(const DownloadSource& source, const std::filesystem::path& targetBinary, std::string* detail) {
#if !defined(__linux__)
  (void)source;
  (void)targetBinary;
  if (detail != nullptr) {
    *detail = "Debian package install is only available on Linux.";
  }
  return false;
#else
  TempDirectory temp("automix_lame_deb");
  const auto debPath = temp.path / "lame.deb";
  if (!downloadToFile(source.url, debPath, "", detail)) {
    return false;
  }

  const auto memberPath = extractArMember(debPath, "data.tar", temp.path, detail);
  if (!memberPath.has_value()) {
    return false;
  }

  const auto lowerName = toLower(memberPath->filename().string());
  std::string compression = "J";
  if (lowerName.ends_with(".tar.gz")) {
    compression = "z";
  } else if (lowerName.ends_with(".tar.xz")) {
    compression = "J";
  } else if (lowerName.ends_with(".tar")) {
    compression.clear();
  } else {
    if (detail != nullptr) {
      *detail = "Unsupported data tar format in .deb: " + memberPath->filename().string();
    }
    return false;
  }

  const auto extractDir = temp.path / "extract";
  std::error_code error;
  std::filesystem::create_directories(extractDir, error);
  if (error) {
    if (detail != nullptr) {
      *detail = "Failed creating extraction dir: " + extractDir.string();
    }
    return false;
  }

  if (!extractTarArchive(*memberPath, extractDir, compression, "", detail)) {
    return false;
  }

  const auto binaryPath = extractDir / "usr" / "bin" / "lame";
  if (!isRegularFile(binaryPath)) {
    if (detail != nullptr) {
      *detail = "Debian package did not contain usr/bin/lame.";
    }
    return false;
  }

  return copyBinaryToCache(binaryPath, targetBinary, detail);
#endif
}

bool installFromGhcr(const DownloadSource& source, const std::filesystem::path& targetBinary, std::string* detail) {
  const auto version = readEnvironment("AUTOMIX_LAME_VERSION").value_or(kDefaultLameVersion);

  const auto tokenJson = fetchJson("https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/lame:pull", "", detail);
  if (!tokenJson.has_value() || !tokenJson->contains("token")) {
    if (detail != nullptr && detail->empty()) {
      *detail = "Failed to resolve GHCR token.";
    }
    return false;
  }

  const std::string token = tokenJson->value("token", "");
  if (token.empty()) {
    if (detail != nullptr) {
      *detail = "GHCR token response did not include a token.";
    }
    return false;
  }

  const std::string authHeader = "Authorization: Bearer " + token + "\n";
  const auto manifestList = fetchJson(
      "https://ghcr.io/v2/homebrew/core/lame/manifests/" + version,
      authHeader + "Accept: application/vnd.oci.image.index.v1+json\n",
      detail);
  if (!manifestList.has_value() || !manifestList->contains("manifests")) {
    if (detail != nullptr && detail->empty()) {
      *detail = "Failed to fetch GHCR manifest list.";
    }
    return false;
  }

  std::string manifestDigest;
  for (const auto& manifest : (*manifestList)["manifests"]) {
    const auto platform = manifest.value("platform", nlohmann::json::object());
    if (platform.value("os", "") == source.ghcrOs && platform.value("architecture", "") == source.ghcrArch) {
      manifestDigest = manifest.value("digest", "");
      break;
    }
  }
  if (manifestDigest.empty()) {
    if (detail != nullptr) {
      *detail = "No GHCR manifest found for " + source.ghcrOs + "/" + source.ghcrArch;
    }
    return false;
  }

  const auto manifest = fetchJson(
      "https://ghcr.io/v2/homebrew/core/lame/manifests/" + manifestDigest,
      authHeader + "Accept: application/vnd.oci.image.manifest.v1+json\n",
      detail);
  if (!manifest.has_value() || !manifest->contains("layers")) {
    if (detail != nullptr && detail->empty()) {
      *detail = "Failed to fetch GHCR image manifest.";
    }
    return false;
  }

  std::string layerDigest;
  std::string mediaType;
  for (const auto& layer : (*manifest)["layers"]) {
    mediaType = layer.value("mediaType", "");
    if (mediaType.find("tar") != std::string::npos) {
      layerDigest = layer.value("digest", "");
      break;
    }
  }
  if (layerDigest.empty()) {
    if (detail != nullptr) {
      *detail = "No tar layer found in GHCR image manifest.";
    }
    return false;
  }

  TempDirectory temp("automix_lame_ghcr");
  const bool gzipLayer = mediaType.find("gzip") != std::string::npos;
  const auto layerPath = temp.path / (gzipLayer ? "layer.tar.gz" : "layer.tar");
  if (!downloadToFile("https://ghcr.io/v2/homebrew/core/lame/blobs/" + layerDigest,
                      layerPath,
                      authHeader + "Accept: application/octet-stream\n",
                      detail)) {
    return false;
  }

  const auto extractDir = temp.path / "extract";
  std::error_code error;
  std::filesystem::create_directories(extractDir, error);
  if (error) {
    if (detail != nullptr) {
      *detail = "Failed creating GHCR extract folder: " + extractDir.string();
    }
    return false;
  }

  if (!extractTarArchive(layerPath, extractDir, gzipLayer ? "z" : "", "", detail)) {
    return false;
  }

  const auto binaryPath = findFileRecursive(extractDir, {"lame", "lame.exe"});
  if (!binaryPath.has_value()) {
    if (detail != nullptr) {
      *detail = "Unable to locate LAME binary in GHCR layer.";
    }
    return false;
  }

  return copyBinaryToCache(*binaryPath, targetBinary, detail);
}

std::filesystem::path internalCacheBinaryPath() {
  const auto key = platformKey().empty() ? "unknown" : platformKey();
  const auto appData =
      std::filesystem::path(juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory).getFullPathName().toStdString());
  return appData / "AutoMixMaster" / "codecs" / "lame" / key / binaryName();
}

} // namespace

std::filesystem::path LameDownloader::cacheBinaryPath() { return internalCacheBinaryPath(); }

bool LameDownloader::isSupportedOnCurrentPlatform() { return !platformSources().empty(); }

LameDownloader::DownloadResult LameDownloader::ensureAvailable(const bool forceDownload) {
  static std::mutex mutex;
  const std::lock_guard<std::mutex> lock(mutex);

  DownloadResult result;
  const auto targetBinary = cacheBinaryPath();
  if (!forceDownload && !flagEnabled("AUTOMIX_LAME_FORCE_DOWNLOAD") && isRegularFile(targetBinary)) {
    std::string detail;
    if (ensureExecutable(targetBinary, &detail)) {
      result.success = true;
      result.executablePath = targetBinary;
      result.detail = "Using cached LAME binary at " + targetBinary.string();
      return result;
    }
  }

  if (!forceDownload && flagEnabled("AUTOMIX_LAME_SKIP_DOWNLOAD")) {
    result.detail = "Skipped LAME download because AUTOMIX_LAME_SKIP_DOWNLOAD is enabled.";
    return result;
  }

  const auto sources = platformSources();
  if (sources.empty()) {
    result.detail = "No fallback LAME downloader source configured for this platform.";
    return result;
  }

  std::vector<std::string> failures;
  result.attempted = true;
  for (const auto& source : sources) {
    std::string attemptDetail;
    bool installed = false;
    switch (source.type) {
      case SourceType::DirectBinary: {
        TempDirectory temp("automix_lame_direct");
        const auto downloadedPath = temp.path / binaryName();
        if (downloadToFile(source.url, downloadedPath, "", &attemptDetail)) {
          installed = copyBinaryToCache(downloadedPath, targetBinary, &attemptDetail);
        }
        break;
      }
      case SourceType::Zip:
        installed = installFromZip(source, targetBinary, &attemptDetail);
        break;
      case SourceType::Debian:
        installed = installFromDebian(source, targetBinary, &attemptDetail);
        break;
      case SourceType::Ghcr:
        installed = installFromGhcr(source, targetBinary, &attemptDetail);
        break;
    }

    if (installed) {
      result.success = true;
      result.executablePath = targetBinary;
      result.detail = "Downloaded fallback LAME binary to " + targetBinary.string();
      return result;
    }

    std::string sourceLabel;
    switch (source.type) {
      case SourceType::DirectBinary:
        sourceLabel = "direct";
        break;
      case SourceType::Zip:
        sourceLabel = "zip";
        break;
      case SourceType::Debian:
        sourceLabel = "debian";
        break;
      case SourceType::Ghcr:
        sourceLabel = "ghcr";
        break;
    }

    failures.push_back("[" + sourceLabel + "] " + (attemptDetail.empty() ? "download attempt failed" : attemptDetail));
  }

  result.detail = failures.empty() ? "LAME download failed." : joinLines(failures);
  return result;
}

} // namespace automix::util
