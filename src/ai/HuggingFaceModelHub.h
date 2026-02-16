#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace automix::ai {

struct HubModelInfo {
  std::string repoId;
  std::string displayName;
  std::string useCase;
  std::string license;
  std::string revision;
  int downloads = 0;
  int likes = 0;
  bool privateRepo = false;
  bool gated = false;
  bool disabled = false;
  bool hasOnnx = false;
  bool recommended = false;
  std::string lastModified;
  std::string sourceUrl;
  std::string primaryFile;
  std::vector<std::string> tags;
  std::vector<std::string> files;
};

struct HubModelQueryOptions {
  size_t maxResultsPerQuery = 8;
  bool includeGated = false;
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
  std::optional<HubModelInfo> modelInfo(const std::string& repoId, const std::string& token = "") const;
  HubInstallResult installModel(const std::string& repoId, const HubInstallOptions& options = {}) const;

  std::string resolveToken(const std::string& explicitToken = "") const;
  static std::vector<std::string> defaultRecommendedSearchTerms();
};

} // namespace automix::ai
