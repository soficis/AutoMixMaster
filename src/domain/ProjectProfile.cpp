#include "domain/ProjectProfile.h"

#include <algorithm>
#include <fstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace automix::domain {
namespace {

ProjectProfile profileFromJson(const nlohmann::json& json) {
  ProjectProfile profile;
  profile.id = json.value("id", "");
  profile.name = json.value("name", profile.id);
  profile.platformPreset = json.value("platformPreset", json.value("platform", "spotify"));
  profile.rendererName = json.value("rendererName", "BuiltIn");
  profile.outputFormat = json.value("outputFormat", "wav");
  profile.lossyBitrateKbps = std::clamp(json.value("lossyBitrateKbps", 320), 64, 320);
  profile.mp3UseVbr = json.value("mp3UseVbr", false);
  profile.mp3VbrQuality = std::clamp(json.value("mp3VbrQuality", 4), 0, 9);
  profile.gpuProvider = json.value("gpuProvider", "auto");
  profile.roleModelPackId = json.value("roleModelPackId", "none");
  profile.mixModelPackId = json.value("mixModelPackId", "none");
  profile.masterModelPackId = json.value("masterModelPackId", "none");
  profile.safetyPolicyId = json.value("safetyPolicyId", "balanced");
  profile.preferredStemCount = std::clamp(json.value("preferredStemCount", 4), 2, 6);
  profile.metadataPolicy = json.value("metadataPolicy", "copy_common");
  if (profile.metadataPolicy != "copy_all" &&
      profile.metadataPolicy != "copy_common" &&
      profile.metadataPolicy != "copy_common_only" &&
      profile.metadataPolicy != "strip" &&
      profile.metadataPolicy != "override_template") {
    profile.metadataPolicy = "copy_common";
  }
  if (json.contains("metadataTemplate") && json.at("metadataTemplate").is_object()) {
    profile.metadataTemplate = json.at("metadataTemplate").get<std::map<std::string, std::string>>();
  }
  if (json.contains("pinnedRendererIds") && json.at("pinnedRendererIds").is_array()) {
    profile.pinnedRendererIds = json.at("pinnedRendererIds").get<std::vector<std::string>>();
  }
  return profile;
}

void appendUnique(std::vector<ProjectProfile>* profiles, const ProjectProfile& profile) {
  if (profile.id.empty() || profile.name.empty()) {
    return;
  }
  const auto exists = std::any_of(profiles->begin(), profiles->end(), [&](const ProjectProfile& existing) {
    return existing.id == profile.id;
  });
  if (!exists) {
    profiles->push_back(profile);
  }
}

std::vector<std::filesystem::path> profileFileCandidates(const std::filesystem::path& repositoryRoot) {
  std::vector<std::filesystem::path> candidates;
  std::error_code error;
  auto current = repositoryRoot.empty() ? std::filesystem::current_path(error) : repositoryRoot;
  if (error) {
    return candidates;
  }

  for (int depth = 0; depth < 5; ++depth) {
    candidates.push_back(current / "assets" / "profiles" / "project_profiles.json");
    candidates.push_back(current / "Assets" / "Profiles" / "project_profiles.json");
    if (!current.has_parent_path()) {
      break;
    }
    const auto parent = current.parent_path();
    if (parent == current) {
      break;
    }
    current = parent;
  }

  return candidates;
}

} // namespace

std::vector<ProjectProfile> defaultProjectProfiles() {
  return {
      ProjectProfile{
          .id = "default",
          .name = "Default Balanced",
          .platformPreset = "spotify",
          .rendererName = "BuiltIn",
          .outputFormat = "wav",
          .lossyBitrateKbps = 320,
          .mp3UseVbr = false,
          .mp3VbrQuality = 4,
          .gpuProvider = "auto",
          .roleModelPackId = "none",
          .mixModelPackId = "none",
          .masterModelPackId = "none",
          .safetyPolicyId = "balanced",
          .preferredStemCount = 4,
          .metadataPolicy = "copy_common",
          .metadataTemplate = {},
          .pinnedRendererIds = {"BuiltIn"},
      },
      ProjectProfile{
          .id = "streaming_spotify",
          .name = "Streaming Spotify",
          .platformPreset = "spotify",
          .rendererName = "BuiltIn",
          .outputFormat = "mp3",
          .lossyBitrateKbps = 256,
          .mp3UseVbr = true,
          .mp3VbrQuality = 2,
          .gpuProvider = "auto",
          .roleModelPackId = "demo-role-v1",
          .mixModelPackId = "demo-mix-v1",
          .masterModelPackId = "demo-master-v1",
          .safetyPolicyId = "strict",
          .preferredStemCount = 4,
          .metadataPolicy = "copy_common",
          .metadataTemplate = {},
          .pinnedRendererIds = {"BuiltIn", "PhaseLimiter"},
      },
      ProjectProfile{
          .id = "mobile_fast_turn",
          .name = "Mobile Fast Turn",
          .platformPreset = "youtube",
          .rendererName = "BuiltIn",
          .outputFormat = "ogg",
          .lossyBitrateKbps = 192,
          .mp3UseVbr = false,
          .mp3VbrQuality = 4,
          .gpuProvider = "cpu",
          .roleModelPackId = "none",
          .mixModelPackId = "none",
          .masterModelPackId = "none",
          .safetyPolicyId = "balanced",
          .preferredStemCount = 2,
          .metadataPolicy = "strip",
          .metadataTemplate = {},
          .pinnedRendererIds = {"BuiltIn"},
      },
  };
}

std::vector<ProjectProfile> loadProjectProfiles(const std::filesystem::path& repositoryRoot) {
  std::vector<ProjectProfile> profiles = defaultProjectProfiles();

  std::unordered_set<std::string> loadedFiles;
  for (const auto& candidate : profileFileCandidates(repositoryRoot)) {
    const auto key = candidate.lexically_normal().string();
    if (!loadedFiles.insert(key).second) {
      continue;
    }

    std::error_code error;
    if (!std::filesystem::is_regular_file(candidate, error) || error) {
      continue;
    }

    try {
      std::ifstream in(candidate);
      nlohmann::json json;
      in >> json;
      if (!json.is_array()) {
        continue;
      }
      for (const auto& entry : json) {
        appendUnique(&profiles, profileFromJson(entry));
      }
    } catch (...) {
    }
  }

  std::sort(profiles.begin(), profiles.end(), [](const ProjectProfile& a, const ProjectProfile& b) {
    if (a.id == "default") {
      return true;
    }
    if (b.id == "default") {
      return false;
    }
    return a.name < b.name;
  });

  return profiles;
}

std::optional<ProjectProfile> findProjectProfile(const std::vector<ProjectProfile>& profiles, const std::string& id) {
  const auto it = std::find_if(profiles.begin(), profiles.end(), [&](const ProjectProfile& profile) {
    return profile.id == id;
  });
  if (it == profiles.end()) {
    return std::nullopt;
  }
  return *it;
}

} // namespace automix::domain
