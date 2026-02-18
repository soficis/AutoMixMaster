#include "renderers/RendererRegistry.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "renderers/ExternalLimiterRenderer.h"
#include "renderers/PhaseLimiterDiscovery.h"
#include "util/HashUtils.h"

namespace automix::renderers {
namespace {

struct TrustPolicy {
  bool enforceSignedDescriptors = false;
  std::unordered_set<std::string> trustedSigners;
  std::unordered_map<std::string, std::vector<std::string>> profileRendererPins;
};

bool hasBinary(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

std::vector<std::filesystem::path> trustPolicyCandidates() {
  std::vector<std::filesystem::path> candidates;
  std::error_code error;
  auto current = std::filesystem::absolute(std::filesystem::current_path(error), error);
  if (error) {
    return candidates;
  }

  for (int depth = 0; depth < 5; ++depth) {
    candidates.push_back(current / "assets" / "renderers" / "trust_policy.json");
    candidates.push_back(current / "assets" / "limiters" / "trust_policy.json");
    candidates.push_back(current / "Assets" / "Renderers" / "trust_policy.json");
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

TrustPolicy loadTrustPolicy() {
  TrustPolicy policy;

  std::set<std::string> visited;
  for (const auto& candidate : trustPolicyCandidates()) {
    const auto key = candidate.lexically_normal().string();
    if (!visited.insert(key).second) {
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
      policy.enforceSignedDescriptors = json.value("enforceSignedDescriptors", false);

      if (json.contains("trustedSigners") && json.at("trustedSigners").is_array()) {
        for (const auto& signer : json.at("trustedSigners")) {
          if (signer.is_string()) {
            policy.trustedSigners.insert(signer.get<std::string>());
          }
        }
      }

      if (json.contains("profileRendererPins") && json.at("profileRendererPins").is_object()) {
        for (const auto& [profileId, rendererIds] : json.at("profileRendererPins").items()) {
          if (!rendererIds.is_array()) {
            continue;
          }
          policy.profileRendererPins[profileId] = rendererIds.get<std::vector<std::string>>();
        }
      }

      return policy;
    } catch (...) {
      continue;
    }
  }

  return policy;
}

bool verifyDescriptorSignature(const nlohmann::json& descriptor,
                               const std::string& algorithm,
                               const std::string& signatureValue) {
  if (algorithm.empty() || signatureValue.empty()) {
    return false;
  }

  if (algorithm != "fnv1a64" && algorithm != "FNV1A64") {
    return false;
  }

  auto canonical = descriptor;
  canonical.erase("signature");
  const auto digest = util::toHex(util::fnv1a64(canonical.dump()));
  return digest == signatureValue;
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
  info.trustPolicyStatus = "trusted:built-in";
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
  info.trustPolicyStatus = "unsigned";

  if (binaryInfo.has_value()) {
    info.binaryPath = binaryInfo->executablePath;
  }

  return info;
}

std::filesystem::path capabilitySnapshotPath(const ExternalRendererConfig& config) {
  const auto safeId = config.id.empty() ? std::string("external") : config.id;
  return config.binaryPath.parent_path() / (safeId + ".capabilities.snapshot.json");
}

void writeCapabilitySnapshot(const ExternalRendererConfig& config,
                             const ExternalLimiterRenderer::ValidationResult& validation,
                             const std::filesystem::path& snapshotPath) {
  nlohmann::json snapshot = {
      {"id", config.id},
      {"name", config.name},
      {"binaryPath", config.binaryPath.string()},
      {"version", validation.version},
      {"supportedFeatures", validation.supportedFeatures},
      {"validated", validation.valid},
      {"errorCode", validation.errorCode},
      {"diagnostics", validation.diagnostics},
      {"signatureValid", config.signatureValid},
      {"signerId", config.signerId},
      {"timestampEpochMs", std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::high_resolution_clock::now().time_since_epoch())
                               .count()},
  };
  std::ofstream out(snapshotPath);
  if (out.is_open()) {
    out << snapshot.dump(2);
  }
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
  info.pinnedProfileIds = config.pinnedProfileIds;
  info.trustPolicyStatus = config.trustPolicyStatus;

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

  const auto snapshotPath = capabilitySnapshotPath(config);
  writeCapabilitySnapshot(config, validation, snapshotPath);
  info.capabilitySnapshotPath = snapshotPath;

  if (validation.valid) {
    info.discovery = "External limiter validated (--validate contract).";
  } else {
    info.discovery = "Validation failed [" + validation.errorCode + "]: " + validation.diagnostics;
  }

  if (!info.trustPolicyStatus.empty()) {
    info.discovery += " Trust=" + info.trustPolicyStatus + ".";
  }

  return info;
}

std::optional<ExternalRendererConfig> loadExternalRendererDescriptor(const std::filesystem::path& descriptorPath,
                                                                     const TrustPolicy& trustPolicy) {
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

    if (json.contains("signature") && json.at("signature").is_object()) {
      const auto& signature = json.at("signature");
      config.signerId = signature.value("signer", "");
      config.signatureAlgorithm = signature.value("algorithm", "");
      config.signatureValue = signature.value("value", "");
      config.signatureValid = verifyDescriptorSignature(json, config.signatureAlgorithm, config.signatureValue);
    }

    if (json.contains("pinnedProfileIds") && json.at("pinnedProfileIds").is_array()) {
      config.pinnedProfileIds = json.at("pinnedProfileIds").get<std::vector<std::string>>();
    }

    if (config.id.empty() || config.name.empty()) {
      return std::nullopt;
    }

    if (!config.signerId.empty() && !trustPolicy.trustedSigners.empty()) {
      if (trustPolicy.trustedSigners.find(config.signerId) == trustPolicy.trustedSigners.end()) {
        config.trustPolicyStatus = "signer_untrusted";
      }
    }

    if (!config.signatureAlgorithm.empty() && !config.signatureValid) {
      config.trustPolicyStatus = "signature_invalid";
    } else if (!config.signatureAlgorithm.empty() && config.signatureValid) {
      config.trustPolicyStatus = "signature_valid";
    }

    if (trustPolicy.enforceSignedDescriptors) {
      if (config.signatureAlgorithm.empty()) {
        config.trustPolicyStatus = "signature_missing";
      }
      if (!config.signerId.empty() && !trustPolicy.trustedSigners.empty() &&
          trustPolicy.trustedSigners.find(config.signerId) == trustPolicy.trustedSigners.end()) {
        config.trustPolicyStatus = "signer_untrusted";
      }
    }

    if (config.trustPolicyStatus.empty()) {
      config.trustPolicyStatus = config.signatureAlgorithm.empty() ? "unsigned" : "signature_valid";
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

std::vector<ExternalRendererConfig> discoverAssetExternalRenderers(const TrustPolicy& trustPolicy) {
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

      const auto config = loadExternalRendererDescriptor(entry.path(), trustPolicy);
      if (config.has_value()) {
        if (trustPolicy.enforceSignedDescriptors) {
          if (config->trustPolicyStatus == "signature_missing" ||
              config->trustPolicyStatus == "signature_invalid" ||
              config->trustPolicyStatus == "signer_untrusted") {
            continue;
          }
        }
        configs.push_back(config.value());
      }
    }
  }

  return configs;
}

} // namespace

std::vector<RendererInfo> RendererRegistry::list(const std::vector<ExternalRendererConfig>& externalConfigs) const {
  const auto trustPolicy = loadTrustPolicy();

  std::vector<RendererInfo> infos;
  const auto assetConfigs = discoverAssetExternalRenderers(trustPolicy);
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

    ExternalRendererConfig effective = config;
    if (effective.trustPolicyStatus.empty()) {
      effective.trustPolicyStatus = "unsigned";
    }
    infos.push_back(makeExternalInfo(effective));
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
