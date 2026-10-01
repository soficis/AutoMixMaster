#include "ai/ModelStorage.h"

#include <fstream>
#include <set>
#include <system_error>

#include <juce_core/juce_core.h>
#include <nlohmann/json.hpp>

namespace automix::ai {
namespace {

constexpr const char* kRegistryFile = "install_registry.json";
constexpr const char* kConsentsFile = "license_consents.json";
constexpr const char* kInstallLogFile = "install_log.jsonl";
constexpr const char* kMigratedNote = "MIGRATED.txt";

std::filesystem::path toPath(const juce::File& file) {
  return std::filesystem::path(file.getFullPathName().toWideCharPointer());
}

nlohmann::json readArray(const std::filesystem::path& path) {
  try {
    std::ifstream in(path);
    if (in.is_open()) {
      auto parsed = nlohmann::json::parse(in);
      if (parsed.is_array()) {
        return parsed;
      }
    }
  } catch (...) {
  }
  return nlohmann::json::array();
}

std::string recordKey(const nlohmann::json& item) {
  return item.is_object() ? item.value("modelId", item.value("repoId", "")) : std::string();
}

// rename() where possible; across volumes, copy then delete the source.
bool movePath(const std::filesystem::path& from, const std::filesystem::path& to, std::string& error) {
  std::error_code code;
  std::filesystem::rename(from, to, code);
  if (!code) {
    return true;
  }
  code.clear();
  std::filesystem::copy(from, to, std::filesystem::copy_options::recursive, code);
  if (code) {
    error = "cannot copy '" + from.string() + "' to '" + to.string() + "': " + code.message();
    std::error_code ignored;
    std::filesystem::remove_all(to, ignored);
    return false;
  }
  std::filesystem::remove_all(from, code);  // the copy is complete; a leftover source is harmless
  return true;
}

// Target records win on the same key; legacy-only records are appended.
nlohmann::json mergeRecords(nlohmann::json target, const nlohmann::json& legacy) {
  std::set<std::string> present;
  for (const auto& item : target) {
    present.insert(recordKey(item));
  }
  for (const auto& item : legacy) {
    const auto key = recordKey(item);
    if (key.empty() || present.insert(key).second) {
      target.push_back(item);
    }
  }
  return target;
}

bool writeJson(const std::filesystem::path& path, const nlohmann::json& value) {
  std::ofstream out(path, std::ios::trunc);
  out << value.dump(2);
  return static_cast<bool>(out);
}

} // namespace

std::filesystem::path defaultModelHubRoot() {
  const auto overrideRoot = juce::SystemStats::getEnvironmentVariable("AUTOMIX_MODEL_HUB_ROOT", {});
  if (overrideRoot.isNotEmpty()) {
    return std::filesystem::path(overrideRoot.toWideCharPointer());
  }
#if defined(_WIN32)
  const auto base = juce::File::getSpecialLocation(juce::File::windowsLocalAppData);
#else
  const auto base = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
#endif
  return toPath(base) / "AutoMixMaster" / "modelhub";
}

std::filesystem::path legacyModelHubRoot() { return std::filesystem::path("assets") / "modelhub"; }

bool isInsideDirectory(const std::filesystem::path& candidate, const std::filesystem::path& root) {
  std::error_code error;
  const auto child = std::filesystem::weakly_canonical(std::filesystem::absolute(candidate, error), error);
  const auto parent = std::filesystem::weakly_canonical(std::filesystem::absolute(root, error), error);
  if (error || child.empty() || parent.empty()) {
    return false;
  }
  const auto relative = child.lexically_relative(parent);
  return !relative.empty() && relative != "." && *relative.begin() != "..";
}

ModelHubMigration migrateModelHub(const std::filesystem::path& legacyRoot, const std::filesystem::path& targetRoot) {
  ModelHubMigration result;
  std::error_code error;
  if (!std::filesystem::is_directory(legacyRoot, error) || std::filesystem::exists(legacyRoot / kMigratedNote, error)) {
    return result;
  }
  if (std::filesystem::equivalent(legacyRoot, targetRoot, error) && !error) {
    return result;  // already the same place (e.g. AUTOMIX_MODEL_HUB_ROOT points at it)
  }
  error.clear();
  if (std::filesystem::is_empty(legacyRoot, error)) {
    return result;
  }
  result.attempted = true;
  std::filesystem::create_directories(targetRoot, error);
  if (error) {
    result.error = "cannot create '" + targetRoot.string() + "': " + error.message();
    return result;
  }

  // Pack directories first; the records describing them follow.
  for (const auto& entry : std::filesystem::directory_iterator(legacyRoot, error)) {
    if (!entry.is_directory(error)) {
      continue;
    }
    const auto name = entry.path().filename();
    if (std::filesystem::exists(targetRoot / name, error)) {
      result.keptInPlace.push_back(name.string());
      continue;
    }
    if (!movePath(entry.path(), targetRoot / name, result.error)) {
      return result;  // nothing is marked migrated, so the next start retries
    }
    ++result.movedPacks;
  }

  // Registry: merge, then point every installPath at the pack's directory here.
  auto registry = mergeRecords(readArray(targetRoot / kRegistryFile), readArray(legacyRoot / kRegistryFile));
  for (auto& item : registry) {
    if (!item.is_object() || !item.contains("installPath")) {
      continue;
    }
    auto installPath = std::filesystem::path(item.value("installPath", "")).lexically_normal();
    if (installPath.filename().empty()) {
      installPath = installPath.parent_path();  // trailing separator
    }
    const auto relocated = targetRoot / installPath.filename();
    if (!installPath.filename().empty() && std::filesystem::is_directory(relocated, error)) {
      item["installPath"] = relocated.string();
    }
  }
  if (!registry.empty() && !writeJson(targetRoot / kRegistryFile, registry)) {
    result.error = "cannot write " + (targetRoot / kRegistryFile).string();
    return result;
  }
  const auto consents = mergeRecords(readArray(targetRoot / kConsentsFile), readArray(legacyRoot / kConsentsFile));
  if (!consents.empty() && !writeJson(targetRoot / kConsentsFile, consents)) {
    result.error = "cannot write " + (targetRoot / kConsentsFile).string();
    return result;
  }
  std::filesystem::remove(legacyRoot / kRegistryFile, error);
  std::filesystem::remove(legacyRoot / kConsentsFile, error);

  // The install log is append-only history: keep both.
  if (std::filesystem::exists(legacyRoot / kInstallLogFile, error)) {
    std::ifstream in(legacyRoot / kInstallLogFile, std::ios::binary);
    std::ofstream out(targetRoot / kInstallLogFile, std::ios::binary | std::ios::app);
    out << in.rdbuf();
    in.close();
    if (out) {
      std::filesystem::remove(legacyRoot / kInstallLogFile, error);
    }
  }

  std::ofstream note(legacyRoot / kMigratedNote, std::ios::trunc);
  note << "Downloaded models moved to " << targetRoot.string() << "\n"
       << "AutoMixMaster no longer reads this folder.\n";
  return result;
}

} // namespace automix::ai
