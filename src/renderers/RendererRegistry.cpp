#include "renderers/RendererRegistry.h"

#include <algorithm>
#include <fstream>
#include <optional>
#include <set>

#include <nlohmann/json.hpp>

#include "renderers/ExternalLimiterRenderer.h"
#include "renderers/PhaseLimiterDiscovery.h"

namespace automix::renderers {
namespace {

bool hasBinary(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

RendererInfo makeBuiltInInfo() {
  RendererInfo info;
  info.id = "BuiltIn";
  info.name = "BuiltIn";
  info.version = "internal";
  info.licenseId = "Project";
  info.linkMode = RendererLinkMode::InProcess;
  info.bundledByDefault = true;
  info.available = true;
  info.discovery = "Always available (core renderer).";
  return info;
}

RendererInfo makePhaseLimiterInfo() {
  PhaseLimiterDiscovery discovery;
  const auto binaryInfo = discovery.find();

  RendererInfo info;
  info.id = "PhaseLimiter";
  info.name = "PhaseLimiter";
  info.version = "external";
  info.licenseId = "See assets/phaselimiter/licenses";
  info.linkMode = RendererLinkMode::External;
  info.bundledByDefault = false;
  info.available = binaryInfo.has_value();
  info.discovery = binaryInfo.has_value() ? "Auto-discovered in assets or PHASELIMITER_BIN."
                                          : "Not found in assets. Set PHASELIMITER_BIN or install under assets.";

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
  info.binaryPath = config.binaryPath;

  if (!hasBinary(config.binaryPath)) {
    info.available = false;
    info.discovery = "Configured path is missing or not executable.";
    return info;
  }

  const auto validation = ExternalLimiterRenderer::validateBinary(config.binaryPath);
  info.available = validation.valid;
  if (!validation.version.empty() && (info.version.empty() || info.version == "unknown")) {
    info.version = validation.version;
  }

  if (validation.valid) {
    info.discovery = "External limiter validated (--validate contract).";
  } else {
    info.discovery = "Validation failed [" + validation.errorCode + "]: " + validation.diagnostics;
  }

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
