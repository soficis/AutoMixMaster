#include "ai/GitHubReleaseModelHub.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>

#include <juce_core/juce_core.h>
#include <nlohmann/json.hpp>

#include "ai/ModelCatalogValidator.h"
#include "util/StringUtils.h"

namespace automix::ai {
namespace {

using ::automix::util::toLower;
using ::automix::util::trim;

struct CuratedGitHubModel {
  std::string repo;
  std::string assetName;
  std::string displayName;
  std::string useCase;
  std::string taskScope;
};

std::vector<CuratedGitHubModel> curatedGitHubModels() {
  return {
      CuratedGitHubModel{
          .repo = "smartdaze/otowake-oto",
          .assetName = "htdemucs_6s.onnx",
          .displayName = "OtoWake Demucs 6s ONNX",
          .useCase = "stem-separation",
          .taskScope = "separation",
      },
      CuratedGitHubModel{
          .repo = "instant-high/Resemble-Denoiser-ONNX",
          .assetName = "denoiser_fp16.onnx",
          .displayName = "Resemble Denoiser FP16 ONNX",
          .useCase = "analysis-denoise",
          .taskScope = "analysis",
      },
  };
}

std::optional<CuratedGitHubModel> findCuratedGitHubModel(const std::string& repo,
                                                         const std::string& assetName) {
  const auto repoLower = toLower(repo);
  const auto assetLower = toLower(assetName);
  for (const auto& curated : curatedGitHubModels()) {
    if (toLower(curated.repo) == repoLower && toLower(curated.assetName) == assetLower) {
      return curated;
    }
  }
  return std::nullopt;
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

  const auto normalized = trim(value);
  if (normalized.empty()) {
    return std::nullopt;
  }
  return normalized;
#endif
}

std::string sanitizeToken(std::string token) {
  token = trim(std::move(token));
  if (!token.empty() && token.rfind("Bearer ", 0) == 0) {
    token = token.substr(7);
  }
  return token;
}

std::string buildHeaders(const std::string& token) {
  std::string headers;
  headers += "Accept: application/vnd.github+json\n";
  headers += "User-Agent: AutoMixMaster\n";
  headers += "X-GitHub-Api-Version: 2022-11-28\n";
  if (!token.empty()) {
    headers += "Authorization: Bearer " + token + "\n";
  }
  return headers;
}

std::optional<nlohmann::json> fetchJson(const std::string& url,
                                        const std::string& token,
                                        std::string* detail = nullptr) {
  int statusCode = 0;
  const auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                           .withConnectionTimeoutMs(45000)
                           .withNumRedirectsToFollow(8)
                           .withStatusCode(&statusCode)
                           .withExtraHeaders(buildHeaders(token));
  const auto input = juce::URL(url).createInputStream(options);
  if (input == nullptr) {
    if (detail != nullptr) {
      *detail = "GitHub request failed (no stream): " + url;
    }
    return std::nullopt;
  }

  if (statusCode >= 400) {
    if (detail != nullptr) {
      *detail = "GitHub request failed (HTTP " + std::to_string(statusCode) + "): " + url;
    }
    return std::nullopt;
  }

  const auto text = input->readEntireStreamAsString().toStdString();
  if (text.empty()) {
    if (detail != nullptr) {
      *detail = "GitHub request returned empty response";
    }
    return std::nullopt;
  }

  try {
    return nlohmann::json::parse(text);
  } catch (const std::exception& error) {
    if (detail != nullptr) {
      *detail = std::string("GitHub JSON parse failed: ") + error.what();
    }
    return std::nullopt;
  }
}

bool downloadToFile(const std::string& url,
                    const std::filesystem::path& outputPath,
                    const std::string& token,
                    std::string* detail = nullptr) {
  int statusCode = 0;
  const auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                           .withConnectionTimeoutMs(60000)
                           .withNumRedirectsToFollow(8)
                           .withStatusCode(&statusCode)
                           .withExtraHeaders(buildHeaders(token));
  const auto input = juce::URL(url).createInputStream(options);
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
      *detail = "Failed creating output directory: " + outputPath.parent_path().string();
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

  if (!std::filesystem::is_regular_file(outputPath, error) || error) {
    if (detail != nullptr) {
      *detail = "Downloaded output missing: " + outputPath.string();
    }
    return false;
  }

  return true;
}

std::string iso8601NowUtc() {
  return juce::Time::getCurrentTime().toISO8601(true).toStdString();
}

nlohmann::json loadJsonIfPresent(const std::filesystem::path& path) {
  try {
    std::ifstream in(path);
    if (!in.is_open()) {
      return nlohmann::json::array();
    }
    nlohmann::json parsed;
    in >> parsed;
    return parsed;
  } catch (...) {
    return nlohmann::json::array();
  }
}

void writeJson(const std::filesystem::path& path, const nlohmann::json& json) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  std::ofstream out(path);
  out << json.dump(2);
}

std::string sanitizePathToken(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (const auto ch : text) {
    const bool alphaNum = std::isalnum(static_cast<unsigned char>(ch)) != 0;
    if (alphaNum || ch == '-' || ch == '_' || ch == '.') {
      out.push_back(ch);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    return "model";
  }
  return out;
}

std::string makeModelId(const std::string& repo, const std::string& assetName) {
  return "github:" + repo + "@" + assetName;
}

struct ParsedModelId {
  std::string repo;
  std::string assetName;
};

std::optional<ParsedModelId> parseModelId(const std::string& modelId) {
  if (modelId.empty()) {
    return std::nullopt;
  }

  std::string raw = modelId;
  if (raw.rfind("github:", 0) == 0) {
    raw = raw.substr(7);
  }

  const auto atPos = raw.find('@');
  if (atPos == std::string::npos || atPos == 0 || atPos + 1 >= raw.size()) {
    return std::nullopt;
  }

  ParsedModelId parsed;
  parsed.repo = raw.substr(0, atPos);
  parsed.assetName = raw.substr(atPos + 1);
  if (parsed.repo.find('/') == std::string::npos) {
    return std::nullopt;
  }
  return parsed;
}

std::optional<nlohmann::json> fetchLatestRelease(const std::string& repo,
                                                 const std::string& token,
                                                 std::string* detail = nullptr) {
  if (repo.empty()) {
    return std::nullopt;
  }
  const auto url = "https://api.github.com/repos/" + repo + "/releases/latest";
  return fetchJson(url, token, detail);
}

std::optional<nlohmann::json> findAsset(const nlohmann::json& release,
                                        const std::string& assetName,
                                        const bool allowFallback) {
  if (!release.is_object() || !release.contains("assets") || !release.at("assets").is_array()) {
    return std::nullopt;
  }

  for (const auto& item : release.at("assets")) {
    if (!item.is_object()) {
      continue;
    }
    if (item.value("name", "") == assetName) {
      return item;
    }
  }

  if (!allowFallback) {
    return std::nullopt;
  }

  static const std::unordered_set<std::string> supportedExt = {
      ".onnx", ".safetensors", ".bin", ".pt", ".pth", ".th", ".ckpt"};

  for (const auto& item : release.at("assets")) {
    if (!item.is_object()) {
      continue;
    }
    const auto name = item.value("name", "");
    const auto ext = toLower(std::filesystem::path(name).extension().string());
    if (supportedExt.find(ext) != supportedExt.end()) {
      return item;
    }
  }

  return std::nullopt;
}

void updateInstallRegistry(const std::filesystem::path& root,
                           const HubModelInfo& model,
                           const HubInstallResult& result) {
  const auto registryPath = root / "install_registry.json";
  auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array()) {
    registry = nlohmann::json::array();
  }

  const auto modelKey = !result.modelId.empty() ? result.modelId : model.modelId;
  bool updated = false;
  for (auto& item : registry) {
    if (!item.is_object() || item.value("modelId", item.value("repoId", "")) != modelKey) {
      continue;
    }
    item = {
        {"modelId", modelKey},
        {"source", result.source.empty() ? model.source : result.source},
        {"repoId", model.repoId},
        {"revision", result.revision},
        {"installedAtUtc", iso8601NowUtc()},
        {"installPath", result.installPath.string()},
        {"primaryFile", result.primaryFilePath.filename().string()},
        {"taskScope", result.taskScope.empty() ? model.taskScope : result.taskScope},
        {"useCase", model.useCase},
        {"license", model.license},
        {"sourceUrl", model.sourceUrl},
        {"downloads", model.downloads},
        {"likes", model.likes},
    };
    updated = true;
    break;
  }

  if (!updated) {
    registry.push_back({
        {"modelId", modelKey},
        {"source", result.source.empty() ? model.source : result.source},
        {"repoId", model.repoId},
        {"revision", result.revision},
        {"installedAtUtc", iso8601NowUtc()},
        {"installPath", result.installPath.string()},
        {"primaryFile", result.primaryFilePath.filename().string()},
        {"taskScope", result.taskScope.empty() ? model.taskScope : result.taskScope},
        {"useCase", model.useCase},
        {"license", model.license},
        {"sourceUrl", model.sourceUrl},
        {"downloads", model.downloads},
        {"likes", model.likes},
    });
  }

  std::sort(registry.begin(), registry.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
    return a.value("modelId", a.value("repoId", "")) < b.value("modelId", b.value("repoId", ""));
  });
  writeJson(registryPath, registry);
}

void appendInstallLog(const std::filesystem::path& root,
                      const HubModelInfo& model,
                      const HubInstallResult& result) {
  const auto logPath = root / "install_log.jsonl";
  std::error_code error;
  std::filesystem::create_directories(logPath.parent_path(), error);

  nlohmann::json event = {
      {"timestampUtc", iso8601NowUtc()},
      {"modelId", result.modelId.empty() ? model.modelId : result.modelId},
      {"source", result.source.empty() ? model.source : result.source},
      {"repoId", model.repoId},
      {"revision", result.revision},
      {"success", result.success},
      {"message", result.message},
      {"installPath", result.installPath.string()},
      {"primaryFile", result.primaryFilePath.filename().string()},
      {"taskScope", result.taskScope.empty() ? model.taskScope : result.taskScope},
      {"license", model.license},
  };

  std::ofstream out(logPath, std::ios::app);
  out << event.dump() << "\n";
}

HubModelInfo buildModelInfoFromRelease(const std::string& repo,
                                       const nlohmann::json& release,
                                       const nlohmann::json& asset,
                                       const std::string& displayName,
                                       const std::string& useCase,
                                       const std::string& taskScope,
                                       const bool curated,
                                       const bool recommended,
                                       const int likes = 0) {
  HubModelInfo info;
  info.source = "github";
  info.repoId = repo;
  info.displayName = displayName.empty() ? (repo + "/" + asset.value("name", "")) : displayName;
  info.useCase = useCase;
  info.taskScope = taskScope;
  info.license = "unknown";
  info.revision = release.value("tag_name", "latest");
  info.downloads = asset.value("download_count", 0);
  info.likes = likes;
  info.privateRepo = false;
  info.gated = false;
  info.disabled = false;
  info.hasOnnx = toLower(std::filesystem::path(asset.value("name", "")).extension().string()) == ".onnx";
  info.recommended = recommended;
  info.curated = curated;
  info.lastModified = release.value("published_at", "");
  info.sourceUrl = release.value("html_url", "https://github.com/" + repo + "/releases");
  info.primaryFile = asset.value("name", "");
  info.files.push_back(info.primaryFile);
  info.tags = {"github-release", curated ? "curated" : "raw"};
  info.modelId = makeModelId(repo, info.primaryFile);

  const auto compatibility = validateCatalogModel(info);
  info.compatible = compatibility.compatible;
  info.taskScope = compatibility.taskScope;
  info.compatibilityReport = compatibility.reason;

  return info;
}

} // namespace

std::string GitHubReleaseModelHub::resolveToken() const {
  for (const auto* key : {"AUTOMIX_GITHUB_TOKEN", "GITHUB_TOKEN"}) {
    if (const auto token = readEnvironment(key); token.has_value()) {
      const auto clean = sanitizeToken(*token);
      if (!clean.empty()) {
        return clean;
      }
    }
  }
  return "";
}

std::vector<HubModelInfo> GitHubReleaseModelHub::discoverRecommended(const HubModelQueryOptions& options) const {
  const auto token = resolveToken();
  std::vector<HubModelInfo> discovered;
  std::set<std::string> seenModelIds;

  if (options.curatedOnly || trim(options.searchText).empty()) {
    for (const auto& curated : curatedGitHubModels()) {
      std::string detail;
      const auto release = fetchLatestRelease(curated.repo, token, &detail);
      if (!release.has_value() || !release->is_object()) {
        continue;
      }

      const auto asset = findAsset(*release, curated.assetName, false);
      if (!asset.has_value()) {
        continue;
      }

      auto info = buildModelInfoFromRelease(curated.repo,
                                            *release,
                                            *asset,
                                            curated.displayName,
                                            curated.useCase,
                                            curated.taskScope,
                                            true,
                                            true);
      if (!info.compatible) {
        continue;
      }
      if (seenModelIds.insert(info.modelId).second) {
        discovered.push_back(std::move(info));
      }
    }
    return discovered;
  }

  const auto rawQuery = trim(options.searchText);
  const auto encodedQuery = juce::URL::addEscapeChars(rawQuery + " audio onnx", false).toStdString();
  const auto searchUrl = "https://api.github.com/search/repositories?q=" + encodedQuery +
                         "&sort=stars&order=desc&per_page=" +
                         std::to_string(std::max<size_t>(1, options.maxResultsPerQuery));

  const auto searchResults = fetchJson(searchUrl, token);
  if (!searchResults.has_value() || !searchResults->is_object() ||
      !searchResults->contains("items") || !searchResults->at("items").is_array()) {
    return discovered;
  }

  for (const auto& repoItem : searchResults->at("items")) {
    if (!repoItem.is_object()) {
      continue;
    }

    const auto repo = repoItem.value("full_name", "");
    if (repo.empty()) {
      continue;
    }

    const auto release = fetchLatestRelease(repo, token);
    if (!release.has_value() || !release->is_object()) {
      continue;
    }

    const auto asset = findAsset(*release, "", true);
    if (!asset.has_value()) {
      continue;
    }

    auto info = buildModelInfoFromRelease(repo,
                                          *release,
                                          *asset,
                                          "",
                                          "",
                                          "",
                                          false,
                                          false,
                                          repoItem.value("stargazers_count", 0));
    if (!info.compatible) {
      continue;
    }

    if (seenModelIds.insert(info.modelId).second) {
      discovered.push_back(std::move(info));
    }

    if (discovered.size() >= 20) {
      break;
    }
  }

  return discovered;
}

std::optional<HubModelInfo> GitHubReleaseModelHub::modelInfo(const std::string& modelId) const {
  const auto parsed = parseModelId(modelId);
  if (!parsed.has_value()) {
    return std::nullopt;
  }

  const auto token = resolveToken();
  const auto release = fetchLatestRelease(parsed->repo, token);
  if (!release.has_value() || !release->is_object()) {
    return std::nullopt;
  }

  const auto asset = findAsset(*release, parsed->assetName, false);
  if (!asset.has_value()) {
    return std::nullopt;
  }

  const auto curated = findCuratedGitHubModel(parsed->repo, parsed->assetName);
  auto info = buildModelInfoFromRelease(parsed->repo,
                                        *release,
                                        *asset,
                                        curated.has_value() ? curated->displayName : "",
                                        curated.has_value() ? curated->useCase : "",
                                        curated.has_value() ? curated->taskScope : "",
                                        curated.has_value(),
                                        curated.has_value());
  info.modelId = modelId;
  return info;
}

HubInstallResult GitHubReleaseModelHub::installModel(const std::string& modelId,
                                                      const HubInstallOptions& options) const {
  HubInstallResult result;
  result.modelId = modelId;
  result.source = "github";

  const auto parsed = parseModelId(modelId);
  if (!parsed.has_value()) {
    result.message = "Invalid GitHub model id.";
    return result;
  }

  const auto token = resolveToken();
  const auto release = fetchLatestRelease(parsed->repo, token);
  if (!release.has_value() || !release->is_object()) {
    result.message = "Unable to fetch GitHub release metadata.";
    return result;
  }

  const auto asset = findAsset(*release, parsed->assetName, false);
  if (!asset.has_value()) {
    result.message = "GitHub release asset not found: " + parsed->assetName;
    return result;
  }

  const auto curated = findCuratedGitHubModel(parsed->repo, parsed->assetName);
  auto info = buildModelInfoFromRelease(parsed->repo,
                                        *release,
                                        *asset,
                                        curated.has_value() ? curated->displayName : "",
                                        curated.has_value() ? curated->useCase : "",
                                        curated.has_value() ? curated->taskScope : "",
                                        curated.has_value(),
                                        curated.has_value());
  info.modelId = modelId;

  const auto compatibility = validateCatalogModel(info);
  if (!compatibility.compatible) {
    result.message = "Model is not compatible for turnkey install: " + compatibility.reason;
    return result;
  }

  result.repoId = info.repoId;
  result.taskScope = compatibility.taskScope;
  result.revision = info.revision;

  const auto destinationRoot = options.destinationRoot.empty() ? std::filesystem::path("assets/modelhub") : options.destinationRoot;
  const auto installKey = sanitizePathToken(info.modelId.empty() ? info.repoId : info.modelId);
  const auto installPath = destinationRoot / installKey;
  const auto primaryPath = installPath / std::filesystem::path(info.primaryFile).filename();
  result.installPath = installPath;
  result.primaryFilePath = primaryPath;

  std::error_code error;
  if (std::filesystem::is_regular_file(primaryPath, error) && !error && !options.overwrite) {
    result.success = true;
    result.message = "Model already installed.";
    result.downloadedFiles = {primaryPath.filename().string()};
    updateInstallRegistry(destinationRoot, info, result);
    appendInstallLog(destinationRoot, info, result);
    return result;
  }

  const auto downloadUrl = asset->value("browser_download_url", "");
  if (downloadUrl.empty()) {
    result.message = "GitHub release asset has no download URL.";
    return result;
  }

  std::string detail;
  if (!downloadToFile(downloadUrl, primaryPath, token, &detail)) {
    result.message = detail.empty() ? "Model download failed." : detail;
    appendInstallLog(destinationRoot, info, result);
    return result;
  }

  result.downloadedFiles.push_back(primaryPath.filename().string());

  nlohmann::json metadata = {
      {"schemaVersion", 1},
      {"modelId", info.modelId},
      {"source", info.source},
      {"repoId", info.repoId},
      {"revision", result.revision},
      {"name", info.displayName},
      {"taskScope", compatibility.taskScope},
      {"useCase", info.useCase},
      {"license", info.license},
      {"sourceUrl", info.sourceUrl},
      {"downloads", info.downloads},
      {"likes", info.likes},
      {"installedAtUtc", iso8601NowUtc()},
      {"primaryFile", primaryPath.filename().string()},
      {"availableFiles", info.files},
      {"tags", info.tags},
  };
  result.metadataPath = installPath / "modelhub.json";
  writeJson(result.metadataPath, metadata);

  std::string manifestError;
  if (!writeTurnkeyModelPackManifest(installPath, info, result, compatibility, &manifestError)) {
    std::filesystem::remove(primaryPath, error);
    result.message = "Failed writing turnkey model pack metadata: " + manifestError;
    appendInstallLog(destinationRoot, info, result);
    return result;
  }

  result.success = true;
  result.message = "Model installed successfully.";
  updateInstallRegistry(destinationRoot, info, result);
  appendInstallLog(destinationRoot, info, result);
  return result;
}

} // namespace automix::ai
