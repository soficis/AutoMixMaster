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
  // Non-primary artifacts fetched alongside the primary model file (e.g.
  // mastering_tcn.onnx + config.json for the ITO-Master pack).
  std::vector<std::string> auxiliaryFiles;
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

// Overrides the live revision and file hash with the pinned values for curated
// repos that have them (Open-Unmix); other repos are left untouched.
void applyCuratedPin(HubModelInfo& info);

// Curated model catalogue (catalog-only discovery source). Exposed for
// license-coverage tests and downstream hub tooling.
std::vector<std::string> curatedModelIds();

// The repo file an install downloads as the pack's model file; empty when the
// repo offers nothing installable. Generic preference order, except for repos
// whose correct file cannot be inferred from names (BS-RoFormer, spec D4).
// preferGpuBuild selects a GPU-oriented variant where a repo has one.
std::string primaryFileForRepo(const std::string& repoId,
                               const std::vector<std::string>& files,
                               bool* hasOnnxOut = nullptr,
                               bool preferGpuBuild = false);

// True when this machine should get BS-RoFormer's GPU (fp32) build: a CUDA
// session opens and the device is large enough for the model.
bool bsRoformerGpuBuildQualifies();

// After GPU support appears (e.g. the GPU runtime pack was installed),
// reinstalls a BS-RoFormer pack under `destinationRoot` that is still on its CPU
// (quantized) build as the GPU build, and removes the superseded file. Empty
// when there is no such pack or the machine does not qualify.
std::optional<HubInstallResult> upgradeBsRoformerForGpu(const std::filesystem::path& destinationRoot);

// Files that must be downloaded next to `primaryFile` to make a complete pack:
// per-repo extras, plus "<primaryFile>.data" whenever the repo publishes one
// (ONNX external weights, which the installer then inlines).
std::vector<std::string> auxiliaryAssetsFor(const std::string& repoId,
                                            const std::string& primaryFile,
                                            const std::vector<std::string>& files);

// Where a repo file lands on disk: always directly inside `installPath`, using
// only the file's own name. Repo subfolders ("onnx/model.onnx") are dropped and
// ".." segments can never escape the install directory.
std::filesystem::path localAssetPath(const std::filesystem::path& installPath, const std::string& repoPath);

} // namespace automix::ai
