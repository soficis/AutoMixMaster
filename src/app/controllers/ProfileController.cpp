#include "app/controllers/ProfileController.h"

namespace automix::app {

std::vector<domain::ProjectProfile> ProfileController::loadProfiles(
    const std::filesystem::path& repositoryRoot) const {
  return domain::loadProjectProfiles(repositoryRoot);
}

std::optional<domain::ProjectProfile> ProfileController::selectedProfile(
    const std::vector<domain::ProjectProfile>& profiles,
    const std::string& preferredProfileId) const {
  if (const auto profile = domain::findProjectProfile(profiles, preferredProfileId); profile.has_value()) {
    return profile;
  }
  if (!profiles.empty()) {
    return profiles.front();
  }
  return std::nullopt;
}

AppliedProfileSettings ProfileController::applyProfile(domain::Session& session,
                                                       const domain::ProjectProfile& profile) const {
  session.projectProfileId = profile.id;
  session.safetyPolicyId = profile.safetyPolicyId;
  session.preferredStemCount = profile.preferredStemCount;

  session.renderSettings.gpuExecutionProvider = profile.gpuProvider;
  session.renderSettings.outputFormat = profile.outputFormat;
  session.renderSettings.lossyBitrateKbps = profile.lossyBitrateKbps;
  session.renderSettings.mp3UseVbr = profile.mp3UseVbr;
  session.renderSettings.mp3VbrQuality = profile.mp3VbrQuality;
  session.renderSettings.metadataPolicy = profile.metadataPolicy;
  session.renderSettings.metadataTemplate = profile.metadataTemplate;
  session.renderSettings.rendererName = profile.rendererName;

  AppliedProfileSettings settings;
  settings.gpuProvider = profile.gpuProvider;
  settings.roleModelPackId = profile.roleModelPackId;
  settings.mixModelPackId = profile.mixModelPackId;
  settings.masterModelPackId = profile.masterModelPackId;
  settings.rendererName = profile.rendererName;
  settings.platformPreset = profile.platformPreset;
  return settings;
}

} // namespace automix::app
