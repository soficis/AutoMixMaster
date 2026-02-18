#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace automix::domain {

// Stem count constraints for audio separation.
// Maximum of 6 stems is based on common separation models (vocals, drums, bass, guitar, keys, other)
// and practical UI/performance limits for real-time mixing workflows.
constexpr int kMinPreferredStemCount = 2;
constexpr int kMaxPreferredStemCount = 6;

struct ProjectProfile {
  std::string id;
  std::string name;
  std::string platformPreset = "spotify";
  std::string rendererName = "BuiltIn";
  std::string outputFormat = "wav";
  int lossyBitrateKbps = 320;
  bool mp3UseVbr = false;
  int mp3VbrQuality = 4;
  std::string gpuProvider = "auto";
  std::string roleModelPackId = "none";
  std::string mixModelPackId = "none";
  std::string masterModelPackId = "none";
  std::string safetyPolicyId = "balanced";
  int preferredStemCount = 4;
  std::string metadataPolicy = "copy_common";
  std::map<std::string, std::string> metadataTemplate;
  std::vector<std::string> pinnedRendererIds;
};

std::vector<ProjectProfile> defaultProjectProfiles();
std::vector<ProjectProfile> loadProjectProfiles(const std::filesystem::path& repositoryRoot);
std::optional<ProjectProfile> findProjectProfile(const std::vector<ProjectProfile>& profiles, const std::string& id);

} // namespace automix::domain
