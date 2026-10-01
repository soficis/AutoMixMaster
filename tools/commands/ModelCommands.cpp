#include "commands/CommandRegistry.h"
#include "commands/DevToolsUtils.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include "ai/FeatureSchema.h"
#include "ai/GpuRuntimePack.h"
#include "ai/HuggingFaceModelHub.h"
#include "ai/ModelPackLoader.h"
#include "ai/OnnxModelInference.h"
#include "ai/OnnxTensorInference.h"
#include "ai/StemSeparator.h"
#include "renderers/ExternalLimiterRenderer.h"
#include "util/LameDownloader.h"

namespace {

using namespace automix::devtools;

int commandListSupportedModels(const CommandArgs&) {
  std::cout << "Supported model packs:\n";
  for (const auto& pack : supportedModelPacks()) {
    std::cout << "  - " << pack.id << " [" << pack.type << "] " << pack.description << "\n";
  }
  return 0;
}

int commandInstallSupportedModel(const CommandArgs& args) {
  const auto idArg = argValue(args, "--id");
  if (!idArg.has_value()) {
    std::cerr << "install-supported-model requires --id <model_id>\n";
    return 2;
  }
  const std::filesystem::path destinationRoot = argValue(args, "--dest").value_or("assets/models");

  const auto it = std::find_if(supportedModelPacks().begin(), supportedModelPacks().end(),
                               [&](const SupportedModelPack& pack) { return pack.id == *idArg; });
  if (it == supportedModelPacks().end()) {
    std::cerr << "Unknown supported model id: " << *idArg << "\n";
    return 2;
  }

  const auto source = findRepoPath(std::filesystem::path("tools/catalog/modelpacks") / it->id);
  if (!source.has_value()) {
    std::cerr << "Catalog source not found for model: " << it->id << "\n";
    return 1;
  }

  const auto destination = destinationRoot / it->id;
  copyDirectory(source.value(), destination);
  std::cout << "Installed model pack '" << it->id << "' to " << destination.string() << "\n";
  return 0;
}

int commandListSupportedLimiters(const CommandArgs&) {
  std::cout << "Supported limiters:\n";
  for (const auto& limiter : supportedLimiters()) {
    std::cout << "  - " << limiter.id << " [" << limiter.name << "] " << limiter.description << "\n";
  }
  return 0;
}

int commandInstallSupportedLimiter(const CommandArgs& args) {
  const auto idArg = argValue(args, "--id");
  if (!idArg.has_value()) {
    std::cerr << "install-supported-limiter requires --id <limiter_id>\n";
    return 2;
  }
  const std::filesystem::path destinationRoot = argValue(args, "--dest").value_or("assets/limiters");
  std::filesystem::create_directories(destinationRoot);

  if (*idArg == "phaselimiter") {
    const auto source = findRepoPath("assets/phaselimiter");
    if (!source.has_value()) {
      std::cerr << "PhaseLimiter source package not found under assets/phaselimiter\n";
      return 1;
    }
    const auto destination = destinationRoot / "phaselimiter";
    std::filesystem::create_directories(destination);
    copyDirectory(source.value(), destination / "runtime");

    nlohmann::json descriptor = {
        {"id", "PhaseLimiterPack"},
        {"name", "PhaseLimiter (pack)"},
        {"version", "external"},
        {"licenseId", "See assets/phaselimiter/licenses"},
        {"binaryPath", "runtime/phase_limiter"},
        {"bundledByDefault", false},
    };
    std::ofstream out(destination / "renderer.json");
    out << descriptor.dump(2);
    std::cout << "Installed limiter '" << *idArg << "' to " << destination.string() << "\n";
    return 0;
  }

  if (*idArg == "external-template") {
    const auto destination = destinationRoot / "external-template";
    std::filesystem::create_directories(destination);
    nlohmann::json descriptor = {
        {"id", "ExternalTemplate"},
        {"name", "External Limiter Template"},
        {"version", "1.0"},
        {"licenseId", "User-supplied"},
        {"binaryPath", "your_limiter_binary_here"},
        {"bundledByDefault", false},
    };
    std::ofstream out(destination / "renderer.json");
    out << descriptor.dump(2);
    std::cout << "Installed limiter template to " << destination.string() << "\n";
    return 0;
  }

  if (*idArg == "ffmpeg" || *idArg == "sox" || *idArg == "rsgain") {
    const auto source = findRepoPath(std::filesystem::path("assets") / *idArg);
    if (!source.has_value()) {
      std::cerr << "Bundled source package not found under assets/" << *idArg << "\n";
      return 1;
    }

    const auto destination = destinationRoot / *idArg;
    try {
      std::filesystem::create_directories(destination);
      copyDirectory(source.value(), destination);
    } catch (const std::exception& ex) {
      std::cerr << "Failed to install limiter '" << *idArg << "': " << ex.what() << "\n";
      return 1;
    }
    std::cout << "Installed limiter '" << *idArg << "' to " << destination.string() << "\n";
    return 0;
  }

  std::cerr << "Unknown supported limiter id: " << *idArg << "\n";
  return 2;
}

int commandInstallLameFallback(const CommandArgs& args) {
  const bool force = hasFlag(args, "--force");
  const bool jsonOutput = hasFlag(args, "--json");

  const auto result = automix::util::LameDownloader::ensureAvailable(force);
  const auto cachePath = automix::util::LameDownloader::cacheBinaryPath();

  if (jsonOutput) {
    nlohmann::json payload = {
        {"success", result.success},
        {"attempted", result.attempted},
        {"path", result.success ? result.executablePath.string() : cachePath.string()},
        {"detail", result.detail},
    };
    std::cout << payload.dump(2) << "\n";
    return result.success ? 0 : 1;
  }

  std::cout << "LAME fallback installation:\n";
  std::cout << "  Success: " << (result.success ? "yes" : "no") << "\n";
  std::cout << "  Attempted download: " << (result.attempted ? "yes" : "no") << "\n";
  std::cout << "  Binary path: " << (result.success ? result.executablePath.string() : cachePath.string()) << "\n";
  if (!result.detail.empty()) {
    std::cout << "  Detail: " << result.detail << "\n";
  }

  return result.success ? 0 : 1;
}

int commandValidateModelPack(const CommandArgs& args) {
  const auto packArg = argValue(args, "--pack");
  if (!packArg.has_value()) {
    std::cerr << "validate-modelpack requires --pack <directory>\n";
    return 2;
  }

  const std::filesystem::path packDir(*packArg);
  automix::ai::ModelPackLoader loader;
  const auto maybePack = loader.load(packDir);
  if (!maybePack.has_value()) {
    std::cerr << "Model pack validation failed: could not load model.json or model file.\n";
    return 1;
  }
  const auto& pack = maybePack.value();

  std::cout << "Model pack loaded: " << pack.id << " engine=" << pack.engine
            << " type=" << pack.type << " version=" << pack.version << "\n";

  std::unique_ptr<automix::ai::IModelInference> inference = std::make_unique<automix::ai::NullModelInference>();
  if (pack.engine == "onnxruntime") {
    inference = std::make_unique<automix::ai::OnnxModelInference>();
  } else if (pack.engine != "unknown") {
    std::cerr << "Warning: model engine '" << pack.engine
              << "' is not supported by validate-modelpack; runtime inference validation will be skipped.\n";
  }

  const auto modelPath = pack.rootPath / pack.modelFile;
  if (!inference->loadModel(modelPath)) {
    if (pack.engine == "unknown") {
      std::cout << "Warning: unknown model engine; skipped runtime inference validation.\n";
      return 0;
    }
    if (!inference->isAvailable()) {
      return 0;
    }
    std::cerr << "Model pack validation failed: backend refused to load model file.\n";
    return 1;
  }

  const size_t featureCount = pack.inputFeatureCount.value_or(automix::ai::FeatureSchemaV1::featureCount());
  const automix::ai::InferenceRequest request{
      .task = taskFromModelType(pack.type),
      .features = deterministicFeatures(featureCount),
  };
  const auto result = inference->run(request);
  if (!result.usedModel) {
    std::cerr << "Model pack validation failed: sample inference did not use model (" << result.logMessage << ")\n";
    return 1;
  }

  for (const auto& key : pack.expectedOutputKeys) {
    if (!result.outputs.contains(key)) {
      std::cerr << "Model pack validation failed: missing expected output key '" << key << "'\n";
      return 1;
    }
  }

  std::cout << "Model pack validation passed.\n";
  return 0;
}

int commandValidateExternalLimiter(const CommandArgs& args) {
  const auto binaryArg = argValue(args, "--binary");
  if (!binaryArg.has_value()) {
    std::cerr << "validate-external-limiter requires --binary <path>\n";
    return 2;
  }

  const bool jsonOutput = hasFlag(args, "--json");
  const std::filesystem::path binaryPath(*binaryArg);
  const auto validation = automix::renderers::ExternalLimiterRenderer::validateBinary(binaryPath);

  if (jsonOutput) {
    nlohmann::json payload = {
        {"binary", binaryPath.string()},
        {"valid", validation.valid},
        {"version", validation.version},
        {"errorCode", validation.errorCode},
        {"diagnostics", validation.diagnostics},
        {"supportedFeatures", validation.supportedFeatures},
    };
    std::cout << payload.dump(2) << "\n";
    return validation.valid ? 0 : 1;
  }

  std::cout << "External limiter validation summary:\n";
  std::cout << "  Binary: " << binaryPath.string() << "\n";
  std::cout << "  Valid: " << (validation.valid ? "yes" : "no") << "\n";
  std::cout << "  Version: " << (validation.version.empty() ? "(none)" : validation.version) << "\n";
  std::cout << "  Error code: " << (validation.errorCode.empty() ? "(none)" : validation.errorCode) << "\n";
  std::cout << "  Diagnostics: " << validation.diagnostics << "\n";

  if (!validation.supportedFeatures.empty()) {
    std::cout << "  Supported features:\n";
    for (const auto& feature : validation.supportedFeatures) {
      std::cout << "    - " << feature << "\n";
    }
  } else {
    std::cout << "  Supported features: (none reported)\n";
  }

  return validation.valid ? 0 : 1;
}

int commandExternalLimiterCompat(const CommandArgs& args) {
  const auto binaryArg = argValue(args, "--binary");
  if (!binaryArg.has_value()) {
    std::cerr << "external-limiter-compat requires --binary <path>\n";
    return 2;
  }

  const auto timeoutMs = parseIntArg(args, "--timeout-ms").value_or(5000);
  const auto requiredFeatures =
      argValue(args, "--required-features").has_value()
          ? splitCommaSeparated(argValue(args, "--required-features").value())
          : std::vector<std::string>{};

  const std::filesystem::path binaryPath(*binaryArg);
  const auto validation = automix::renderers::ExternalLimiterRenderer::validateBinary(binaryPath, timeoutMs);

  std::unordered_set<std::string> supported;
  for (const auto& feature : validation.supportedFeatures) {
    supported.insert(toLower(feature));
  }
  std::vector<std::string> missingRequired;
  for (const auto& required : requiredFeatures) {
    if (supported.find(toLower(required)) == supported.end()) {
      missingRequired.push_back(required);
    }
  }

  const bool strictFeatureCheck = !requiredFeatures.empty();
  const bool featureCompatible = missingRequired.empty();
  const bool compatible = validation.valid && (!strictFeatureCheck || featureCompatible);
  const std::string tier = !validation.valid ? "incompatible"
                                             : (featureCompatible ? "compatible" : "partial");

  nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"binary", binaryPath.string()},
      {"timeoutMs", timeoutMs},
      {"valid", validation.valid},
      {"tier", tier},
      {"version", validation.version},
      {"errorCode", validation.errorCode},
      {"diagnostics", validation.diagnostics},
      {"supportedFeatures", validation.supportedFeatures},
      {"requiredFeatures", requiredFeatures},
      {"missingRequiredFeatures", missingRequired},
  };

  if (const auto outArg = argValue(args, "--out"); outArg.has_value()) {
    writeJsonFile(*outArg, payload);
    std::cout << "Compatibility report: " << *outArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "External limiter compatibility\n";
    std::cout << "  Binary: " << binaryPath.string() << "\n";
    std::cout << "  Tier: " << tier << "\n";
    std::cout << "  Valid: " << (validation.valid ? "yes" : "no") << "\n";
    std::cout << "  Version: " << (validation.version.empty() ? "(none)" : validation.version) << "\n";
    if (!missingRequired.empty()) {
      std::cout << "  Missing required features:\n";
      for (const auto& feature : missingRequired) {
        std::cout << "    - " << feature << "\n";
      }
    }
    std::cout << "  Diagnostics: " << validation.diagnostics << "\n";
  }

  return compatible ? 0 : 1;
}

int commandModelBrowse(const CommandArgs& args) {
  automix::ai::HuggingFaceModelHub hub;
  automix::ai::HubModelQueryOptions options;
  options.maxResultsPerQuery = static_cast<size_t>(std::clamp(parseIntArg(args, "--limit").value_or(6), 1, 20));

  if (const auto tokenEnvArg = argValue(args, "--token-env"); tokenEnvArg.has_value()) {
    options.token = readEnvironment(*tokenEnvArg).value_or("");
  }

  const auto models = hub.discoverRecommended(options);
  nlohmann::json payload = nlohmann::json::array();
  for (const auto& model : models) {
    payload.push_back({
        {"repoId", model.repoId},
        {"useCase", model.useCase},
        {"license", model.license},
        {"downloads", model.downloads},
        {"likes", model.likes},
        {"revision", model.revision},
        {"primaryFile", model.primaryFile},
        {"recommended", model.recommended},
        {"gated", model.gated},
        {"sourceUrl", model.sourceUrl},
    });
  }

  if (const auto outArg = argValue(args, "--out"); outArg.has_value()) {
    writeJsonFile(*outArg, payload);
    std::cout << "Wrote model catalog to " << *outArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Recommended model catalog entries: " << payload.size() << "\n";
    for (const auto& model : payload) {
      std::cout << "  - " << model.value("repoId", "")
                << " useCase=" << model.value("useCase", "")
                << " downloads=" << model.value("downloads", 0)
                << " license=" << model.value("license", "")
                << (model.value("recommended", false) ? " [recommended]" : "")
                << "\n";
    }
  }

  return payload.empty() ? 1 : 0;
}

int commandModelInstall(const CommandArgs& args) {
  const auto repoArg = argValue(args, "--repo");
  if (!repoArg.has_value()) {
    std::cerr << "model-install requires --repo <huggingface_repo_id>\n";
    return 2;
  }

  automix::ai::HubInstallOptions options;
  options.destinationRoot = argValue(args, "--dest").value_or("assets/modelhub");
  options.overwrite = hasFlag(args, "--force");
  options.downloadReadme = !hasFlag(args, "--no-readme");
  if (const auto tokenEnvArg = argValue(args, "--token-env"); tokenEnvArg.has_value()) {
    options.token = readEnvironment(*tokenEnvArg).value_or("");
  }

  automix::ai::HuggingFaceModelHub hub;
  const auto installed = hub.installModel(*repoArg, options);

  nlohmann::json payload = {
      {"success", installed.success},
      {"repoId", installed.repoId},
      {"revision", installed.revision},
      {"installPath", installed.installPath.string()},
      {"primaryFilePath", installed.primaryFilePath.string()},
      {"metadataPath", installed.metadataPath.string()},
      {"message", installed.message},
      {"downloadedFiles", installed.downloadedFiles},
  };

  if (const auto outArg = argValue(args, "--out"); outArg.has_value()) {
    writeJsonFile(*outArg, payload);
    std::cout << "Model install report: " << *outArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Model install result: " << (installed.success ? "success" : "failed") << "\n";
    std::cout << "  Repo: " << installed.repoId << "\n";
    std::cout << "  Revision: " << installed.revision << "\n";
    std::cout << "  Path: " << installed.installPath.string() << "\n";
    std::cout << "  Detail: " << installed.message << "\n";
  }

  return installed.success ? 0 : 1;
}

int commandModelHealth(const CommandArgs& args) {
  const std::filesystem::path root = argValue(args, "--root").value_or("assets/modelhub");
  const auto registryPath = root / "install_registry.json";
  const auto registry = loadJsonFile(registryPath);
  if (!registry.has_value() || !registry->is_array()) {
    std::cerr << "Model registry not found: " << registryPath.string() << "\n";
    return 1;
  }

  nlohmann::json checks = nlohmann::json::array();
  int ok = 0;
  int failed = 0;

  for (const auto& item : *registry) {
    if (!item.is_object()) {
      continue;
    }
    const auto repoId = item.value("repoId", "");
    const std::filesystem::path installPath(item.value("installPath", ""));
    const std::filesystem::path primaryPath = installPath / item.value("primaryFile", "");
    std::error_code error;
    const bool downloaded = std::filesystem::is_regular_file(primaryPath, error) && !error;
    bool loadable = false;
    std::string detail;

    if (downloaded) {
      const auto extension = toLower(primaryPath.extension().string());
      if (extension == ".onnx") {
        automix::ai::OnnxModelInference inference;
        loadable = inference.loadModel(primaryPath);
        detail = loadable ? "ONNX load ok" : "ONNX load failed";
      } else {
        loadable = true;
        detail = "Non-ONNX model file present (schema/runtime check skipped)";
      }
    } else {
      detail = "Primary model file missing";
    }

    if (downloaded && loadable) {
      ++ok;
    } else {
      ++failed;
    }

    checks.push_back({
        {"repoId", repoId},
        {"downloaded", downloaded},
        {"loadable", loadable},
        {"expectedIoSchema", "modelhub.json metadata"},
        {"detail", detail},
    });
  }

  nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"root", root.string()},
      {"ok", ok},
      {"failed", failed},
      {"checks", checks},
  };

  if (const auto outArg = argValue(args, "--out"); outArg.has_value()) {
    writeJsonFile(*outArg, payload);
    std::cout << "Model health report: " << *outArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Model health: ok=" << ok << " failed=" << failed << "\n";
  }

  return failed == 0 ? 0 : 1;
}

// separate --mix <file> --out <dir> [--pack <dir>] [--tensor] [--provider auto|cpu|cuda] [--cancel-after-ms N] [--json]
// Runs StemSeparator exactly as single-mix import does, outside the UI.
int commandSeparate(const std::vector<std::string>& args) {
  const auto mixArg = argValue(args, "--mix");
  const auto outArg = argValue(args, "--out");
  if (!mixArg.has_value() || !outArg.has_value()) {
    std::cerr << "Usage: separate --mix <file> --out <dir> [--pack <dir>] [--tensor] [--provider auto|cpu|cuda] [--cancel-after-ms N] [--json]\n";
    return 2;
  }

  automix::ai::StemSeparator separator(argValue(args, "--pack").value_or("assets/models/stem-separator"));
  automix::ai::StemSeparator::SeparationOptions options;
  options.useTensorModel = hasFlag(args, "--tensor");
  options.executionProvider = argValue(args, "--provider").value_or("auto");
  const bool jsonOutput = hasFlag(args, "--json");
  // --cancel-after-ms N: request cancellation N ms after start, to measure how
  // quickly a running separation actually stops.
  const auto started = std::chrono::steady_clock::now();
  if (const auto cancelAfter = parseIntArg(args, "--cancel-after-ms"); cancelAfter.has_value()) {
    const auto deadline = started + std::chrono::milliseconds(*cancelAfter);
    options.cancelRequested = [deadline] { return std::chrono::steady_clock::now() >= deadline; };
  }
  if (options.useTensorModel && !jsonOutput) {
    options.tensorProgress = [](const int done, const int total) {
      std::cout << "  chunk " << done << "/" << total << "\n" << std::flush;
    };
  }

  const auto result = separator.separate(*mixArg, *outArg, options);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  nlohmann::json stems = nlohmann::json::array();
  for (const auto& stem : result.stems) {
    stems.push_back({{"name", stem.name}, {"path", stem.filePath}, {"role", automix::domain::toString(stem.role)}});
  }
  const nlohmann::json payload = {
      {"success", result.success},
      {"usedModel", result.usedModel},
      {"cancelled", result.cancelled},
      {"seconds", seconds},
      {"stems", stems},
      {"energyLeakage", result.qaMetrics.energyLeakage},
      {"residualDistortion", result.qaMetrics.residualDistortion},
      {"transientRetention", result.qaMetrics.transientRetention},
      {"log", result.logMessage},
  };
  if (jsonOutput) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Separation: success=" << (result.success ? "yes" : "no")
              << " usedModel=" << (result.usedModel ? "yes" : "no") << " seconds=" << seconds << "\n";
    for (const auto& stem : result.stems) {
      std::cout << "  " << automix::domain::toString(stem.role) << " -> " << stem.filePath << "\n";
    }
    std::cout << "  " << result.logMessage << "\n";
  }
  return result.success ? 0 : 1;
}

// gpu-runtime status|install|remove|upgrade-models [--root <dir>] [--hub <dir>]
// The per-user NVIDIA CUDA libraries the CUDA execution provider loads.
int commandGpuRuntime(const std::vector<std::string>& args) {
  namespace pack = automix::ai::GpuRuntimePack;
  const auto has = [&args](const char* word) { return std::find(args.begin(), args.end(), word) != args.end(); };
  const std::string action = has("install")          ? "install"
                             : has("remove")         ? "remove"
                             : has("upgrade-models") ? "upgrade-models"
                             : (has("status") || args.size() <= 1) ? "status"
                                                                    : "";
  const std::filesystem::path root = argValue(args, "--root").value_or(pack::defaultRoot().string());
  if (action == "status") {
    const auto adapter = pack::largestNvidiaAdapter();
    const auto driver = adapter.has_value() && adapter->driverMajor.has_value()
                            ? std::to_string(*adapter->driverMajor) + "." +
                                  (*adapter->driverMinor < 10 ? "0" : "") + std::to_string(*adapter->driverMinor)
                            : std::string("unknown");
    std::cout << "GPU runtime " << pack::version() << " at " << root.string() << "\n"
              << "  installed: " << (pack::isInstalled(root) ? "yes" : "no") << "\n"
              << "  largest NVIDIA adapter: "
              << (adapter.has_value() ? std::to_string(adapter->dedicatedBytes / (1024 * 1024)) + " MiB, driver " + driver
                                      : std::string("none"))
              << " (CUDA 13 needs >= " << pack::kMinimumDriverMajor << ")\n"
              << "  runtime build has CUDA: " << (automix::ai::runtimeReportsProvider("cuda") ? "yes" : "no") << "\n"
              << "  preload: " << (pack::preload(root) ? "ok" : "not loaded") << "\n";
    std::string provider;
    const bool gpu = automix::ai::gpuTensorSessionAvailable(&provider);
    std::cout << "  GPU tensor session: " << (gpu ? provider : std::string("unavailable")) << "\n";
    return 0;
  }
  if (action == "install") {
    int lastDecile = -1;
    const auto result = pack::install(root, [&](std::uint64_t done, std::uint64_t total) {
      const int percent = total > 0 ? static_cast<int>(done * 100 / total) : 0;
      if (percent / 10 != lastDecile) {
        lastDecile = percent / 10;
        std::cout << "  " << percent << "%\n" << std::flush;
      }
      return true;
    });
    std::cout << result.message << "\n";
    return result.success ? 0 : 1;
  }
  if (action == "remove") {
    const auto removed = pack::uninstall(root);
    std::cout << removed.message << "\n";
    return removed.removedNow ? 0 : 3;
  }
  if (action == "upgrade-models") {
    const std::filesystem::path hub = argValue(args, "--hub").value_or("assets/modelhub");
    const auto upgraded = automix::ai::upgradeBsRoformerForGpu(hub);
    std::cout << (upgraded.has_value() ? upgraded->message : std::string("Nothing to upgrade.")) << "\n";
    return !upgraded.has_value() || upgraded->success ? 0 : 1;
  }
  std::cerr << "Usage: gpu-runtime status|install|remove|upgrade-models [--root <dir>] [--hub <dir>]\n";
  return 2;
}

} // namespace

void registerModelCommands(automix::devtools::CommandRegistry& registry) {
  registry.add("list-supported-models", commandListSupportedModels);
  registry.add("install-supported-model", commandInstallSupportedModel);
  registry.add("list-supported-limiters", commandListSupportedLimiters);
  registry.add("install-supported-limiter", commandInstallSupportedLimiter);
  registry.add("install-lame-fallback", commandInstallLameFallback);
  registry.add("validate-modelpack", commandValidateModelPack);
  registry.add("validate-external-limiter", commandValidateExternalLimiter);
  registry.add("external-limiter-compat", commandExternalLimiterCompat);
  registry.add("model-browse", commandModelBrowse);
  registry.add("model-install", commandModelInstall);
  registry.add("model-health", commandModelHealth);
  registry.add("separate", commandSeparate);
  registry.add("gpu-runtime", commandGpuRuntime);
}
