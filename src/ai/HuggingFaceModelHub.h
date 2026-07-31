#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace automix::ai {

struct HubModelInfo {
  std::string modelId;
  std::string source = "huggingface";
  std::string repoId;
  std::string displayName;
  std::string useCase;
  std::string taskScope;
  std::string license;
  std::string revision;
  int downloads = 0;
  int likes = 0;
  bool privateRepo = false;
  bool gated = false;
  bool disabled = false;
  bool hasOnnx = false;
  bool recommended = false;
  bool curated = false;
  bool compatible = false;
  std::string compatibilityReport;
  std::string lastModified;
  std::string sourceUrl;
  std::string primaryFile;
  std::vector<std::string> tags;
  std::vector<std::string> files;
  std::unordered_map<std::string, std::string> fileSha256;
};

struct HubModelQueryOptions {
  size_t maxResultsPerQuery = 8;
  bool includeGated = false;
  bool curatedOnly = true;
  std::string searchText;
  std::string token;
};

struct HubInstallOptions {
  std::filesystem::path destinationRoot = "assets/modelhub";
  std::string token;
  bool downloadReadme = true;
  bool overwrite = false;
};

struct HubInstallResult {
  bool success = false;
  std::string modelId;
  std::string source = "huggingface";
  std::string taskScope;
  std::string repoId;
  std::filesystem::path installPath;
  std::filesystem::path primaryFilePath;
  std::filesystem::path metadataPath;
  std::string revision;
  std::string message;
  std::vector<std::string> downloadedFiles;
};

class HuggingFaceModelHub {
 public:
  std::vector<HubModelInfo> discoverRecommended(const HubModelQueryOptions& options = {}) const;
  std::optional<HubModelInfo> modelInfo(const std::string& modelIdOrRepoId, const std::string& token = "") const;
  HubInstallResult installModel(const std::string& modelIdOrRepoId, const HubInstallOptions& options = {}) const;

  // Single filter shared by both curated and search discovery so catalog and
  // search hide gated/private/disabled, file-less, and incompatible models
  // identically (matching install-time rejection in installModel).
  static bool passesDiscoveryFilters(const HubModelInfo& info, const HubModelQueryOptions& options);

  std::string resolveToken(const std::string& explicitToken = "") const;
  static std::vector<std::string> defaultRecommendedSearchTerms();

  // Maps a repo id + tags + fallback search query onto a hub use-case label.
  // Public so tests can pin curated entries (e.g. ITO-Master -> mastering-assistant).
  static std::string inferUseCase(const std::string& repoId,
                                  const std::vector<std::string>& tags,
                                  const std::string& fallbackQuery);
};

// Curated model catalogue (catalog-only discovery source). Exposed for
// license-coverage tests and downstream hub tooling.
std::vector<std::string> curatedModelIds();

} // namespace automix::ai
