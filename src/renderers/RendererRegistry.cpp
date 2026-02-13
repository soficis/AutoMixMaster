#include "renderers/RendererRegistry.h"

#include <algorithm>
#include <fstream>
#include <optional>
#include <set>

#include <nlohmann/json.hpp>

#include "renderers/PhaseLimiterDiscovery.h"

namespace automix::renderers {
namespace {

bool hasBinary(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

RendererInfo makeBuiltInInfo() {
  return RendererInfo{
      .id = "BuiltIn",
      .name = "BuiltIn",
      .version = "internal",
      .licenseId = "Project",
      .linkMode = RendererLinkMode::InProcess,
      .bundledByDefault = true,
      .available = true,
      .discovery = "Always available (core renderer).",
  };
}

RendererInfo makePhaseLimiterInfo() {
  PhaseLimiterDiscovery discovery;
  const auto binaryInfo = discovery.find();

  RendererInfo info{
      .id = "PhaseLimiter",
      .name = "PhaseLimiter",
      .version = "external",
      .licenseId = "See assets/phaselimiter/licenses",
      .linkMode = RendererLinkMode::External,
      .bundledByDefault = false,
      .available = binaryInfo.has_value(),
      .discovery = binaryInfo.has_value() ? "Auto-discovered in assets or PHASELIMITER_BIN."
                                          : "Not found in assets. Set PHASELIMITER_BIN or install under assets.",
  };

  if (binaryInfo.has_value()) {
    info.binaryPath = binaryInfo->executablePath;
  }

  return info;
}

RendererInfo makeExternalInfo(const ExternalRendererConfig& config) {
  RendererInfo info;
  info.id = config.id;
  info.name = config.name;
  info.version = config.version;
  info.licenseId = config.licenseId;
  info.linkMode = RendererLinkMode::External;
  info.bundledByDefault = config.bundledByDefault;
  info.available = hasBinary(config.binaryPath);
  info.binaryPath = config.binaryPath;
  info.discovery = info.available ? "User-supplied external binary path."
                                  : "Configured path is missing or not executable.";
  return info;
}

std::optional<ExternalRendererConfig> loadExternalRendererDescriptor(const std::filesystem::path& descriptorPath) {
  try {
    std::ifstream in(descriptorPath);
    if (!in.is_open()) {
      return std::nullopt;
    }

    nlohmann::json json;
    in >> json;

    ExternalRendererConfig config;
    config.id = json.value("id", descriptorPath.parent_path().filename().string());
    config.name = json.value("name", config.id);
    config.version = json.value("version", "unknown");
    config.licenseId = json.value("licenseId", json.value("license", "unknown"));
    config.bundledByDefault = json.value("bundledByDefault", false);

    std::string binaryPath = json.value("binaryPath", json.value("binary", ""));
    if (binaryPath.empty()) {
      return std::nullopt;
    }

    const std::filesystem::path binary(binaryPath);
    config.binaryPath = binary.is_absolute() ? binary : (descriptorPath.parent_path() / binary);
    if (config.id.empty() || config.name.empty()) {
      return std::nullopt;
    }
    return config;
  } catch (...) {
    return std::nullopt;
  }
}

std::vector<std::filesystem::path> assetLimiterRoots() {
  std::vector<std::filesystem::path> roots;
  std::error_code error;
  auto current = std::filesystem::absolute(std::filesystem::current_path(error), error);
  if (error) {
    return roots;
  }

  for (int depth = 0; depth < 5; ++depth) {
    roots.push_back(current / "assets" / "limiters");
    roots.push_back(current / "assets" / "renderers");
    roots.push_back(current / "Assets" / "Limiters");
    if (!current.has_parent_path()) {
      break;
    }
    current = current.parent_path();
  }
  return roots;
}

std::vector<ExternalRendererConfig> discoverAssetExternalRenderers() {
  std::vector<ExternalRendererConfig> configs;
  std::set<std::string> seenDescriptors;
  std::error_code error;

  for (const auto& root : assetLimiterRoots()) {
    if (!std::filesystem::exists(root, error) || error) {
      continue;
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(
             root,
             std::filesystem::directory_options::skip_permission_denied,
             error)) {
      if (error || !entry.is_regular_file()) {
        continue;
      }
      if (entry.path().filename() != "renderer.json") {
        continue;
      }

      const auto descriptorKey = entry.path().lexically_normal().string();
      if (!seenDescriptors.insert(descriptorKey).second) {
        continue;
      }

      const auto config = loadExternalRendererDescriptor(entry.path());
      if (config.has_value()) {
        configs.push_back(config.value());
      }
    }
  }

  return configs;
}

} // namespace

std::vector<RendererInfo> RendererRegistry::list(const std::vector<ExternalRendererConfig>& externalConfigs) const {
  std::vector<RendererInfo> infos;
  const auto assetConfigs = discoverAssetExternalRenderers();
  infos.reserve(2 + externalConfigs.size() + assetConfigs.size());
  infos.push_back(makeBuiltInInfo());
  infos.push_back(makePhaseLimiterInfo());

  std::set<std::string> seenExternalIds;
  auto appendConfig = [&](const ExternalRendererConfig& config) {
    if (config.id.empty() || config.name.empty()) {
      return;
    }
    if (!seenExternalIds.insert(config.id).second) {
      return;
    }
    infos.push_back(makeExternalInfo(config));
  };

  for (const auto& config : externalConfigs) {
    appendConfig(config);
  }
  for (const auto& config : assetConfigs) {
    appendConfig(config);
  }

  std::stable_sort(infos.begin(), infos.end(), [](const RendererInfo& left, const RendererInfo& right) {
    if (left.id == "BuiltIn") {
      return true;
    }
    if (right.id == "BuiltIn") {
      return false;
    }
    if (left.available != right.available) {
      return left.available > right.available;
    }
    return left.name < right.name;
  });
  return infos;
}

std::string toString(const RendererLinkMode mode) {
  switch (mode) {
    case RendererLinkMode::InProcess:
      return "in-process";
    case RendererLinkMode::External:
      return "external";
  }

  return "unknown";
}

} // namespace automix::renderers
