#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace automix::ai {

// Where downloaded models and their records (install_registry.json,
// license_consents.json) live: %LOCALAPPDATA%\AutoMixMaster\modelhub on
// Windows, <user application data>/AutoMixMaster/modelhub elsewhere, or
// AUTOMIX_MODEL_HUB_ROOT when set (portable installs, tests). Per user and
// writable, unlike the application folder under Program Files - and unlike the
// old cwd-relative "assets/modelhub", which moved with the working directory.
std::filesystem::path defaultModelHubRoot();

// The pre-2026-10 location, relative to the working directory.
std::filesystem::path legacyModelHubRoot();

struct ModelHubMigration {
  bool attempted = false;                // a legacy hub with content was found
  int movedPacks = 0;
  std::vector<std::string> keptInPlace;  // already present at the target; left untouched
  std::string error;
};

// Moves a legacy hub into `targetRoot`: every pack directory, plus the hub's
// records. install_registry.json and license_consents.json are merged by
// modelId (the target's entry wins), so recorded licence consents survive, and
// every registry installPath is rewritten to the pack's new directory. Packs
// moved across volumes are copied, then deleted at the source. Leaves a
// MIGRATED.txt note behind and is a no-op once that note exists.
ModelHubMigration migrateModelHub(const std::filesystem::path& legacyRoot, const std::filesystem::path& targetRoot);

// True when `candidate` is strictly inside `root` (after normalisation). Used to
// refuse deleting anything a registry entry points at outside the hub.
bool isInsideDirectory(const std::filesystem::path& candidate, const std::filesystem::path& root);

} // namespace automix::ai
