#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "domain/ProjectProfile.h"
#include "domain/Session.h"

namespace automix::app {

struct AppliedProfileSettings {
  std::string gpuProvider;
  std::string roleModelPackId;
  std::string mixModelPackId;
  std::string masterModelPackId;
  std::string rendererName;
  std::string platformPreset;
};

class ProfileController {
 public:
  std::vector<domain::ProjectProfile> loadProfiles(const std::filesystem::path& repositoryRoot) const;
  std::optional<domain::ProjectProfile> selectedProfile(const std::vector<domain::ProjectProfile>& profiles,
                                                        const std::string& preferredProfileId) const;
  AppliedProfileSettings applyProfile(domain::Session& session, const domain::ProjectProfile& profile) const;
};

} // namespace automix::app
