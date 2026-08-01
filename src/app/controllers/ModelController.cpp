#include "app/controllers/ModelController.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "ai/GitHubReleaseModelHub.h"
#include "util/CallbackDispatch.h"

namespace automix::app {
namespace {

std::filesystem::path defaultModelHubRoot() {
  return std::filesystem::path("assets") / "modelhub";
}

nlohmann::json loadJsonIfPresent(const std::filesystem::path& path) {
  try {
    std::ifstream in(path);
    if (!in.is_open()) {
      return nlohmann::json::array();
    }
    nlohmann::json parsed;
    in >> parsed;
    return parsed;
  } catch (...) {
    return nlohmann::json::array();
  }
}

bool writeJson(const std::filesystem::path& path, const nlohmann::json& value, std::string* errorMessage) {
  try {
    std::ofstream out(path);
    if (!out.is_open()) {
      if (errorMessage != nullptr) {
        *errorMessage = "Unable to open file for writing: " + path.string();
      }
      return false;
    }
    out << value.dump(2);
    out << "\n";
    return true;
  } catch (const std::exception& e) {
    if (errorMessage != nullptr) {
      *errorMessage = e.what();
    }
    return false;
  } catch (...) {
    if (errorMessage != nullptr) {
      *errorMessage = "Unknown error writing json";
    }
    return false;
  }
}

std::set<std::string> installedModelIdsFromRegistry(const std::filesystem::path& root) {
  std::set<std::string> ids;
  const auto registryPath = root / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array()) {
    return ids;
  }

  for (const auto& item : registry) {
    if (!item.is_object()) {
      continue;
    }
    const auto modelId = item.value("modelId", item.value("repoId", ""));
    if (!modelId.empty()) {
      ids.insert(modelId);
    }
  }
  return ids;
}

std::string normalizeTaskScopeValue(std::string scope) {
  std::transform(scope.begin(), scope.end(), scope.begin(), [](const char value) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
  });
  if (scope == "mix" || scope == "master" || scope == "analysis" || scope == "separation") {
    return scope;
  }
  if (scope == "stem-separation" || scope == "source-separation") {
    return "separation";
  }
  return {};
}

struct RegistryInstallSelection {
  std::string modelId;
  std::string taskScope;
  std::filesystem::path installPath;
};

std::optional<RegistryInstallSelection> findRegistryInstall(const std::filesystem::path& root, const std::string& modelId) {
  if (modelId.empty()) {
    return std::nullopt;
  }
  const auto registryPath = root / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array()) {
    return std::nullopt;
  }

  for (const auto& item : registry) {
    if (!item.is_object()) {
      continue;
    }
    const auto itemModelId = item.value("modelId", item.value("repoId", ""));
    if (itemModelId != modelId) {
      continue;
    }
    RegistryInstallSelection selection;
    selection.modelId = itemModelId;
    selection.taskScope = normalizeTaskScopeValue(item.value("taskScope", ""));
    selection.installPath = std::filesystem::path(item.value("installPath", ""));
    return selection;
  }
  return std::nullopt;
}

double clampProgress(const double progress) {
  return std::clamp(progress, 0.0, 1.0);
}

void emitProgress(const ModelController::Callbacks& callbacks, const double progress) {
  if (callbacks.onProgress) {
    callbacks.onProgress(clampProgress(progress));
  }
}

// ── License consent gating (non-commercial / CC BY-NC models) ──

std::string repoIdFromModelId(const std::string& modelId) {
  if (modelId.rfind("huggingface:", 0) == 0) {
    return modelId.substr(12);
  }
  return modelId;
}

constexpr const char* kNonCommercialLicenseLabel = "CC BY-NC 4.0";

constexpr const char* kItoMasterAttribution =
    "ITO-Master, Koo et al., Sony Research (github.com/SonyResearch/ITO-Master); "
    "ONNX export by kramp (huggingface.co/kramp/ito-master-onnx)";

std::string attributionForRepo(const std::string& repoId) {
  if (repoId == "kramp/ito-master-onnx") {
    return kItoMasterAttribution;
  }
  return "See upstream source: https://huggingface.co/" + repoId;
}

std::filesystem::path licenseConsentsPath(const std::filesystem::path& root) {
  return root / "license_consents.json";
}

std::string iso8601NowUtc() {
  return juce::Time::getCurrentTime().toISO8601(true).toStdString();
}

} // namespace

ModelController::ModelHubOps ModelController::createDefaultHubOps() {
  ModelHubOps ops;
  ops.discoverRecommended = [](const ai::HubModelQueryOptions& options) {
    ai::HuggingFaceModelHub hfHub;
    ai::GitHubReleaseModelHub ghHub;

    auto discovered = hfHub.discoverRecommended(options);
    auto gitHubDiscovered = ghHub.discoverRecommended(options);
    discovered.insert(discovered.end(), gitHubDiscovered.begin(), gitHubDiscovered.end());

    std::set<std::string> seen;
    std::vector<ai::HubModelInfo> deduped;
    deduped.reserve(discovered.size());
    for (auto& model : discovered) {
      const auto key = !model.modelId.empty() ? model.modelId : model.repoId;
      if (!seen.insert(key).second) {
        continue;
      }
      deduped.push_back(std::move(model));
    }
    return deduped;
  };
  ops.installModel = [](const std::string& modelId, const ai::HubInstallOptions& options) {
    if (modelId.rfind("github:", 0) == 0) {
      ai::GitHubReleaseModelHub hub;
      return hub.installModel(modelId, options);
    }

    ai::HuggingFaceModelHub hub;
    return hub.installModel(modelId, options);
  };
  ops.modelInfo = [](const std::string& modelId) {
    if (modelId.rfind("github:", 0) == 0) {
      ai::GitHubReleaseModelHub hub;
      return hub.modelInfo(modelId);
    }

    ai::HuggingFaceModelHub hub;
    return hub.modelInfo(modelId);
  };
  return ops;
}

ModelController::ModelController(ai::ModelManager& modelManager,
                                 juce::ThreadPool& threadPool,
                                 Callbacks callbacks,
                                 ModelHubOps hubOps)
    : modelManager_(modelManager),
      threadPool_(threadPool),
      callbacks_(std::move(callbacks)),
      hubOps_(std::move(hubOps)),
      modelHubRoot_(defaultModelHubRoot()) {
  if (!hubOps_.discoverRecommended || !hubOps_.installModel || !hubOps_.modelInfo) {
    throw std::invalid_argument("ModelController requires complete model-hub operations");
  }
}

namespace {

void reportModelTaskCancelled(const ModelController::Callbacks& callbacks, const std::string& taskName) {
  if (callbacks.onStatus) {
    callbacks.onStatus("Models: " + taskName + " cancelled");
  }
  emitProgress(callbacks, 1.0);
  if (callbacks.onTaskHistory) {
    callbacks.onTaskHistory("Models " + taskName + " cancelled");
  }
}

} // namespace

void ModelController::setModelHubRoot(const std::filesystem::path& root) {
  modelHubRoot_ = root;
}

const std::vector<ai::HubModelInfo>& ModelController::discoveredModels() const {
  return discoveredModels_;
}

std::set<std::string> ModelController::installedModelIds() const {
  return installedModelIdsFromRegistry(modelHubRoot_);
}

bool ModelController::modelRequiresLicenseConsent(const std::string& repoId) {
  const auto normalized = repoIdFromModelId(repoId);
  static const std::vector<std::string> nonCommercialRepoIds = {
      "kramp/ito-master-onnx",
      "SonyCSLParis/music2latent",
  };
  return std::find(nonCommercialRepoIds.begin(), nonCommercialRepoIds.end(), normalized) !=
         nonCommercialRepoIds.end();
}

bool ModelController::hasModelLicenseConsent(const std::string& modelId) const {
  if (modelId.empty()) {
    return false;
  }
  const auto consents = loadJsonIfPresent(licenseConsentsPath(modelHubRoot_));
  if (!consents.is_array()) {
    return false;
  }
  for (const auto& item : consents) {
    if (item.is_object() && item.value("modelId", "") == modelId) {
      return true;
    }
  }
  return false;
}

bool ModelController::acknowledgeModelLicenseConsent(const std::string& modelId) {
  const auto repoId = repoIdFromModelId(modelId);
  if (repoId.empty() || !modelRequiresLicenseConsent(repoId)) {
    return false;
  }

  auto consents = loadJsonIfPresent(licenseConsentsPath(modelHubRoot_));
  if (!consents.is_array()) {
    consents = nlohmann::json::array();
  }

  const nlohmann::json record = {
      {"modelId", modelId},
      {"repoId", repoId},
      {"license", kNonCommercialLicenseLabel},
      {"attribution", attributionForRepo(repoId)},
      {"acknowledgedAtUtc", iso8601NowUtc()},
  };

  bool updated = false;
  for (auto& item : consents) {
    if (item.is_object() && item.value("modelId", "") == modelId) {
      item = record;
      updated = true;
      break;
    }
  }
  if (!updated) {
    consents.push_back(record);
  }

  std::string writeError;
  return writeJson(licenseConsentsPath(modelHubRoot_), consents, &writeError);
}

bool ModelController::activateInstalledModelForTask(const std::string& modelId, const std::string& taskScope) {
  const auto normalizedRequestedScope = normalizeTaskScopeValue(taskScope);
  const auto registrySelection = findRegistryInstall(modelHubRoot_, modelId);
  if (!registrySelection.has_value()) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: selected model is not installed");
    }
    if (callbacks_.onTaskHistory) {
      callbacks_.onTaskHistory("Model activation failed (not installed): " + modelId);
    }
    return false;
  }

  if (ModelController::modelRequiresLicenseConsent(repoIdFromModelId(modelId)) && !hasModelLicenseConsent(modelId)) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: license consent required for " + modelId);
    }
    if (callbacks_.onTaskHistory) {
      callbacks_.onTaskHistory("Model activation blocked (CC BY-NC consent not acknowledged): " + modelId);
    }
    return false;
  }

  auto packScope = normalizedRequestedScope.empty() ? registrySelection->taskScope : normalizedRequestedScope;
  if (packScope.empty()) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: unable to determine task scope");
    }
    if (callbacks_.onTaskHistory) {
      callbacks_.onTaskHistory("Model activation failed (scope missing): " + modelId);
    }
    return false;
  }

  const auto packs = modelManager_.scan();
  const auto selectedPack = std::find_if(packs.begin(), packs.end(), [&](const ai::ModelPack& pack) {
    if (normalizeTaskScopeValue(pack.taskScope) != packScope) {
      return false;
    }
    std::error_code error;
    const auto packRoot = std::filesystem::weakly_canonical(pack.rootPath, error);
    if (error) {
      return false;
    }
    const auto installRoot = std::filesystem::weakly_canonical(registrySelection->installPath, error);
    if (error) {
      return false;
    }
    return packRoot == installRoot;
  });

  if (selectedPack == packs.end()) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: installed pack is not loadable");
    }
    if (callbacks_.onTaskHistory) {
      callbacks_.onTaskHistory("Model activation failed (pack missing): " + modelId);
    }
    return false;
  }

  modelManager_.setActivePackId(packScope, selectedPack->id);
  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: active " + packScope + " pack set to " + selectedPack->id);
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Model activated for " + packScope + ": " + selectedPack->id);
  }
  if (callbacks_.onReport) {
    callbacks_.onReport("Activated model pack for task '" + packScope + "': " + selectedPack->id +
                        "\nSource model: " + modelId +
                        "\nPath: " + registrySelection->installPath.string());
  }
  if (callbacks_.onModelPacksChanged) {
    callbacks_.onModelPacksChanged();
  }
  return true;
}

void ModelController::fetchCatalog(std::atomic_bool& cancelFlag, const bool curatedOnly, std::string searchText) {
  if (cancelFlag.load()) {
    auto capturedCallbacks = callbacks_;
    util::dispatchCallback([capturedCallbacks]() {
      reportModelTaskCancelled(capturedCallbacks, "catalog fetch");
      if (capturedCallbacks.onCatalogReady) {
        capturedCallbacks.onCatalogReady(true);
      }
    });
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus(curatedOnly ? "Models: fetching curated catalog..." : "Models: searching catalog...");
  }
  if (callbacks_.onTaskHistory) {
    if (curatedOnly) {
      callbacks_.onTaskHistory("Models curated catalog fetch started");
    } else {
      callbacks_.onTaskHistory("Models raw catalog search started: " + searchText);
    }
  }
  emitProgress(callbacks_, 0.05);

  struct CatalogJob final : juce::ThreadPoolJob {
    Callbacks callbacks;
    std::vector<ai::HubModelInfo>* modelsOut;
    std::atomic_bool* cancelFlag;
    ModelHubOps hubOps;
    bool curatedOnly;
    std::string searchText;

    CatalogJob(Callbacks cb,
               std::vector<ai::HubModelInfo>* out,
               std::atomic_bool* cancel,
               ModelHubOps ops,
               const bool curated,
               std::string search)
        : juce::ThreadPoolJob("FetchCatalog"),
          callbacks(std::move(cb)),
          modelsOut(out),
          cancelFlag(cancel),
          hubOps(std::move(ops)),
          curatedOnly(curated),
          searchText(std::move(search)) {}

    bool isCancellationRequested() const {
      return shouldExit() || (cancelFlag != nullptr && cancelFlag->load());
    }

    void requestCancellation() const {
      if (cancelFlag != nullptr) {
        cancelFlag->store(true);
      }
    }

    JobStatus runJob() override {
      bool cancelled = false;
      std::vector<ai::HubModelInfo> models;

      if (isCancellationRequested()) {
        requestCancellation();
        cancelled = true;
      }

      if (!cancelled) {
        emitProgress(callbacks, 0.2);
        ai::HubModelQueryOptions options;
        options.maxResultsPerQuery = 10;
        options.curatedOnly = curatedOnly;
        options.searchText = searchText;
        models = hubOps.discoverRecommended(options);
        if (models.size() > 20) {
          models.resize(20);
        }
        emitProgress(callbacks, 0.85);
        if (isCancellationRequested()) {
          requestCancellation();
          cancelled = true;
          models.clear();
        }
      }

      auto capturedCallbacks = callbacks;
      auto capturedModelsOut = modelsOut;
      const bool curatedMode = curatedOnly;
      const auto query = searchText;
      util::dispatchCallback([capturedCallbacks, capturedModelsOut, cancelled, models = std::move(models), curatedMode, query]() mutable {
        emitProgress(capturedCallbacks, 1.0);
        if (cancelled) {
          reportModelTaskCancelled(capturedCallbacks, "catalog fetch");
          if (capturedCallbacks.onCatalogReady) {
            capturedCallbacks.onCatalogReady(true);
          }
          return;
        }

        *capturedModelsOut = std::move(models);

        if (capturedModelsOut->empty()) {
          if (capturedCallbacks.onStatus) {
            capturedCallbacks.onStatus("Models: no compatible catalog results");
          }
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory("Models catalog returned no results");
          }
          if (capturedCallbacks.onReport) {
            std::string report = "Model browser: no compatible entries were returned.\n";
            report += "Mode: ";
            report += curatedMode ? "curated catalog" : "raw search";
            report += "\n";
            if (!curatedMode && !query.empty()) {
              report += "Query: " + query + "\n";
            }
            report += "Sources queried: Hugging Face + GitHub Releases.\n";
            report += "Tip: check internet access and optional tokens "
                      "(AUTOMIX_HF_TOKEN/HF_TOKEN/HUGGINGFACE_TOKEN, "
                      "AUTOMIX_GITHUB_TOKEN/GITHUB_TOKEN).";
            capturedCallbacks.onReport(report);
          }
          if (capturedCallbacks.onCatalogReady) {
            capturedCallbacks.onCatalogReady(false);
          }
          return;
        }

        if (capturedCallbacks.onStatus) {
          capturedCallbacks.onStatus("Models: select model to download");
        }
        if (capturedCallbacks.onTaskHistory) {
          capturedCallbacks.onTaskHistory(
              "Models catalog loaded (" + std::to_string(capturedModelsOut->size()) + " entries)");
        }
        if (capturedCallbacks.onCatalogReady) {
          capturedCallbacks.onCatalogReady(false);
        }
      });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new CatalogJob(callbacks_, &discoveredModels_, &cancelFlag, hubOps_, curatedOnly, std::move(searchText)), true);
}

void ModelController::installModel(const std::string& modelId, std::atomic_bool& cancelFlag) {
  if (modelId.empty()) {
    emitProgress(callbacks_, 1.0);
    if (callbacks_.onInstallComplete) {
      callbacks_.onInstallComplete(false);
    }
    return;
  }

  if (cancelFlag.load()) {
    auto capturedCallbacks = callbacks_;
    auto capturedModelId = modelId;
    util::dispatchCallback([capturedCallbacks, capturedModelId]() {
      reportModelTaskCancelled(capturedCallbacks, "install");
      if (capturedCallbacks.onTaskHistory) {
        capturedCallbacks.onTaskHistory("Model install cancelled: " + capturedModelId);
      }
      if (capturedCallbacks.onInstallComplete) {
        capturedCallbacks.onInstallComplete(true);
      }
    });
    return;
  }

  const auto selectedIt = std::find_if(discoveredModels_.begin(), discoveredModels_.end(), [&](const auto& entry) {
    const auto key = !entry.modelId.empty() ? entry.modelId : entry.repoId;
    return key == modelId;
  });
  if (selectedIt != discoveredModels_.end() && !selectedIt->compatible) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: incompatible model");
    }
    if (callbacks_.onTaskHistory) {
      callbacks_.onTaskHistory("Model install rejected by compatibility gate: " + modelId);
    }
    if (callbacks_.onReport) {
      callbacks_.onReport("Install blocked for " + modelId + ": " + selectedIt->compatibilityReport);
    }
    emitProgress(callbacks_, 1.0);
    if (callbacks_.onInstallComplete) {
      callbacks_.onInstallComplete(false);
    }
    return;
  }

  if (ModelController::modelRequiresLicenseConsent(repoIdFromModelId(modelId)) && !hasModelLicenseConsent(modelId)) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: license consent required for " + modelId);
    }
    if (callbacks_.onTaskHistory) {
      callbacks_.onTaskHistory("Model install blocked (CC BY-NC consent not acknowledged): " + modelId);
    }
    if (callbacks_.onReport) {
      std::string report = "Model install blocked for " + modelId + "\n";
      report += "License: " + std::string(kNonCommercialLicenseLabel) + " (non-commercial use only)\n";
      report += "Attribution: " + attributionForRepo(repoIdFromModelId(modelId)) + "\n";
      report += "Download is refused until you explicitly acknowledge the CC BY-NC license for this model.\n";
      report += "Never bundle NC weights in a commercial installer/redistribution; runtime hub download under the user's license is how this stays legal.\n";
      callbacks_.onReport(report);
    }
    emitProgress(callbacks_, 1.0);
    if (callbacks_.onInstallComplete) {
      callbacks_.onInstallComplete(false);
    }
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: installing " + modelId);
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Model install started: " + modelId);
  }
  emitProgress(callbacks_, 0.05);

  struct InstallJob final : juce::ThreadPoolJob {
    std::string modelId;
    std::filesystem::path hubRoot;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;
    ModelHubOps hubOps;

    InstallJob(std::string id,
               std::filesystem::path root,
               std::atomic_bool* cancel,
               Callbacks cb,
               ModelHubOps ops)
        : juce::ThreadPoolJob("InstallModel"),
          modelId(std::move(id)),
          hubRoot(std::move(root)),
          cancelFlag(cancel),
          callbacks(std::move(cb)),
          hubOps(std::move(ops)) {}

    bool isCancellationRequested() const {
      return shouldExit() || (cancelFlag != nullptr && cancelFlag->load());
    }

    void requestCancellation() const {
      if (cancelFlag != nullptr) {
        cancelFlag->store(true);
      }
    }

    JobStatus runJob() override {
      bool cancelled = false;
      ai::HubInstallResult install;

      if (isCancellationRequested()) {
        requestCancellation();
        cancelled = true;
      }

      if (!cancelled) {
        emitProgress(callbacks, 0.2);
        ai::HubInstallOptions installOptions;
        installOptions.destinationRoot = hubRoot;
        install = hubOps.installModel(modelId, installOptions);
        emitProgress(callbacks, 0.9);
        if (isCancellationRequested()) {
          requestCancellation();
          cancelled = true;
        }
      }

      auto capturedCallbacks = callbacks;
      auto capturedModelId = modelId;
      util::dispatchCallback([capturedCallbacks, capturedModelId, cancelled, install]() {
        emitProgress(capturedCallbacks, 1.0);
        if (cancelled) {
          reportModelTaskCancelled(capturedCallbacks, "install");
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory("Model install cancelled: " + capturedModelId);
          }
          if (capturedCallbacks.onInstallComplete) {
            capturedCallbacks.onInstallComplete(true);
          }
          return;
        }

        if (install.success) {
          if (capturedCallbacks.onStatus) {
            capturedCallbacks.onStatus("Model installed: " + capturedModelId);
          }
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory("Model installed: " + capturedModelId);
          }
        } else {
          if (capturedCallbacks.onStatus) {
            capturedCallbacks.onStatus("Model install failed");
          }
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory("Model install failed: " + capturedModelId);
          }
        }

        std::string report;
        report += "Model install: " + capturedModelId + "\n";
        report += "Source: " + install.source + "\n";
        report += "Task scope: " + install.taskScope + "\n";
        report += "Revision: " + install.revision + "\n";
        report += "Result: " + std::string(install.success ? "success" : "failed") + "\n";
        report += "Detail: " + install.message + "\n";
        if (!install.installPath.empty()) {
          report += "Path: " + install.installPath.string() + "\n";
        }
        report += "Token usage: env-based token is supported via AUTOMIX_HF_TOKEN/HF_TOKEN/HUGGINGFACE_TOKEN.\n";
        if (capturedCallbacks.onReport) {
          capturedCallbacks.onReport(report);
        }

        if (install.success && capturedCallbacks.onModelPacksChanged) {
          capturedCallbacks.onModelPacksChanged();
        }
        if (capturedCallbacks.onInstallComplete) {
          capturedCallbacks.onInstallComplete(false);
        }
      });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new InstallJob(modelId, modelHubRoot_, &cancelFlag, callbacks_, hubOps_), true);
}

void ModelController::uninstallModel(const std::string& modelId, std::atomic_bool& cancelFlag) {
  if (modelId.empty()) {
    emitProgress(callbacks_, 1.0);
    if (callbacks_.onUninstallComplete) {
      callbacks_.onUninstallComplete(false);
    }
    return;
  }

  if (cancelFlag.load()) {
    auto capturedCallbacks = callbacks_;
    auto capturedModelId = modelId;
    util::dispatchCallback([capturedCallbacks, capturedModelId]() {
      reportModelTaskCancelled(capturedCallbacks, "uninstall");
      if (capturedCallbacks.onTaskHistory) {
        capturedCallbacks.onTaskHistory("Model uninstall cancelled: " + capturedModelId);
      }
      if (capturedCallbacks.onUninstallComplete) {
        capturedCallbacks.onUninstallComplete(true);
      }
    });
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: uninstalling " + modelId);
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Model uninstall started: " + modelId);
  }
  emitProgress(callbacks_, 0.05);

  struct UninstallJob final : juce::ThreadPoolJob {
    std::string modelId;
    std::filesystem::path hubRoot;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;

    UninstallJob(std::string id,
                 std::filesystem::path root,
                 std::atomic_bool* cancel,
                 Callbacks cb)
        : juce::ThreadPoolJob("UninstallModel"),
          modelId(std::move(id)),
          hubRoot(std::move(root)),
          cancelFlag(cancel),
          callbacks(std::move(cb)) {}

    bool isCancellationRequested() const {
      return shouldExit() || (cancelFlag != nullptr && cancelFlag->load());
    }

    void requestCancellation() const {
      if (cancelFlag != nullptr) {
        cancelFlag->store(true);
      }
    }

    JobStatus runJob() override {
      bool cancelled = false;
      bool success = false;
      std::string detail;
      std::filesystem::path installPath;
      const auto registryPath = hubRoot / "install_registry.json";

      if (isCancellationRequested()) {
        requestCancellation();
        cancelled = true;
      }

      if (!cancelled) {
        emitProgress(callbacks, 0.2);
        auto registry = loadJsonIfPresent(registryPath);
        if (!registry.is_array()) {
          detail = "No install registry found.";
        } else {
          auto matchIt = registry.end();
          for (auto it = registry.begin(); it != registry.end(); ++it) {
            if (!it->is_object()) {
              continue;
            }
            const auto itemModelId = it->value("modelId", it->value("repoId", ""));
            if (itemModelId == modelId) {
              matchIt = it;
              installPath = std::filesystem::path(it->value("installPath", ""));
              break;
            }
          }

          if (matchIt == registry.end()) {
            detail = "Model is not currently installed.";
          } else {
            std::error_code error;
            if (!installPath.empty() && std::filesystem::exists(installPath, error)) {
              std::filesystem::remove_all(installPath, error);
            }
            if (error) {
              detail = "Failed removing install directory: " + installPath.string();
            } else {
              registry.erase(matchIt);
              std::string writeError;
              if (!writeJson(registryPath, registry, &writeError)) {
                detail = "Failed updating install registry: " + writeError;
              } else {
                success = true;
                detail = "Model uninstalled successfully.";
              }
            }
          }
        }
        emitProgress(callbacks, 0.9);
        if (isCancellationRequested()) {
          requestCancellation();
          cancelled = true;
        }
      }

      auto capturedCallbacks = callbacks;
      auto capturedModelId = modelId;
      auto capturedInstallPath = installPath;
      auto capturedDetail = detail;
      util::dispatchCallback([capturedCallbacks, capturedModelId, capturedInstallPath, capturedDetail, success, cancelled]() {
        emitProgress(capturedCallbacks, 1.0);
        if (cancelled) {
          reportModelTaskCancelled(capturedCallbacks, "uninstall");
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory("Model uninstall cancelled: " + capturedModelId);
          }
          if (capturedCallbacks.onUninstallComplete) {
            capturedCallbacks.onUninstallComplete(true);
          }
          return;
        }

        if (capturedCallbacks.onStatus) {
          capturedCallbacks.onStatus(success ? "Model uninstalled: " + capturedModelId : "Model uninstall failed");
        }
        if (capturedCallbacks.onTaskHistory) {
          capturedCallbacks.onTaskHistory(success ? "Model uninstalled: " + capturedModelId
                                                  : "Model uninstall failed: " + capturedModelId);
        }
        if (capturedCallbacks.onReport) {
          std::string report;
          report += "Model uninstall: " + capturedModelId + "\n";
          report += "Result: " + std::string(success ? "success" : "failed") + "\n";
          report += "Detail: " + capturedDetail + "\n";
          if (!capturedInstallPath.empty()) {
            report += "Path: " + capturedInstallPath.string() + "\n";
          }
          capturedCallbacks.onReport(report);
        }
        if (success && capturedCallbacks.onModelPacksChanged) {
          capturedCallbacks.onModelPacksChanged();
        }
        if (capturedCallbacks.onUninstallComplete) {
          capturedCallbacks.onUninstallComplete(false);
        }
      });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new UninstallJob(modelId, modelHubRoot_, &cancelFlag, callbacks_), true);
}

void ModelController::showInstalled() {
  const auto registryPath = modelHubRoot_ / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array() || registry.empty()) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: no installed hub models");
    }
    if (callbacks_.onReport) {
      callbacks_.onReport("No installed modelhub entries found at " + registryPath.string());
    }
    return;
  }

  std::string report = "Installed Models\n";
  report += "Registry: " + registryPath.string() + "\n\n";
  for (const auto& item : registry) {
    if (!item.is_object()) {
      continue;
    }
    report += "- " + item.value("modelId", item.value("repoId", "")) +
              " source=" + item.value("source", "huggingface") +
              " task=" + item.value("taskScope", "analysis") +
              " rev=" + item.value("revision", "") +
              " useCase=" + item.value("useCase", "") +
              " license=" + item.value("license", "") + "\n";
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: installed list loaded");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Loaded installed model list");
  }
  if (callbacks_.onReport) {
    callbacks_.onReport(report);
  }
}

void ModelController::checkUpdates(std::atomic_bool& cancelFlag) {
  if (cancelFlag.load()) {
    auto capturedCallbacks = callbacks_;
    util::dispatchCallback([capturedCallbacks]() {
      reportModelTaskCancelled(capturedCallbacks, "update check");
      if (capturedCallbacks.onUpdateCheckComplete) {
        capturedCallbacks.onUpdateCheckComplete(true);
      }
    });
    return;
  }

  const auto registryPath = modelHubRoot_ / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array() || registry.empty()) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: no installed hub models");
    }
    emitProgress(callbacks_, 1.0);
    if (callbacks_.onUpdateCheckComplete) {
      callbacks_.onUpdateCheckComplete(false);
    }
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: checking updates...");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Model update check started");
  }
  emitProgress(callbacks_, 0.05);

  std::vector<std::string> modelIds;
  modelIds.reserve(registry.size());
  for (const auto& item : registry) {
    if (item.is_object()) {
      const auto modelId = item.value("modelId", item.value("repoId", ""));
      if (!modelId.empty()) {
        modelIds.push_back(modelId);
      }
    }
  }

  struct UpdateCheckJob final : juce::ThreadPoolJob {
    std::vector<std::string> modelIds;
    std::filesystem::path registryPath;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;
    ModelHubOps hubOps;

    UpdateCheckJob(std::vector<std::string> ids,
                   std::filesystem::path regPath,
                   std::atomic_bool* cancel,
                   Callbacks cb,
                   ModelHubOps ops)
        : juce::ThreadPoolJob("CheckUpdates"),
          modelIds(std::move(ids)),
          registryPath(std::move(regPath)),
          cancelFlag(cancel),
          callbacks(std::move(cb)),
          hubOps(std::move(ops)) {}

    bool isCancellationRequested() const {
      return shouldExit() || (cancelFlag != nullptr && cancelFlag->load());
    }

    void requestCancellation() const {
      if (cancelFlag != nullptr) {
        cancelFlag->store(true);
      }
    }

    JobStatus runJob() override {
      bool cancelled = false;
      std::string report = "Model Update Check\n";
      report += "Registry: " + registryPath.string() + "\n\n";
      int updatesAvailable = 0;
      const auto totalRepoCount = std::max<size_t>(1, modelIds.size());

      for (size_t repoIndex = 0; repoIndex < modelIds.size(); ++repoIndex) {
        const auto& modelId = modelIds[repoIndex];
        if (isCancellationRequested()) {
          requestCancellation();
          cancelled = true;
          break;
        }

        const auto localRegistry = loadJsonIfPresent(registryPath);
        std::string localRevision;
        std::string localRepoId;
        if (localRegistry.is_array()) {
          for (const auto& item : localRegistry) {
            if (item.is_object() && item.value("modelId", item.value("repoId", "")) == modelId) {
              localRevision = item.value("revision", "");
              localRepoId = item.value("repoId", "");
              break;
            }
          }
        }

        const auto remote = hubOps.modelInfo(modelId);
        if (!remote.has_value()) {
          report += "- " + modelId + ": unable to fetch remote metadata\n";
          emitProgress(callbacks, 0.15 + (0.75 * (static_cast<double>(repoIndex + 1) / static_cast<double>(totalRepoCount))));
          continue;
        }

        if (isCancellationRequested()) {
          requestCancellation();
          cancelled = true;
          break;
        }

        const bool changed = !localRevision.empty() && localRevision != remote->revision;
        if (changed) {
          ++updatesAvailable;
        }
        report += "- " + modelId +
                  " repo=" + localRepoId +
                  " local=" + localRevision +
                  " remote=" + remote->revision +
                  " status=" + std::string(changed ? "update-available" : "up-to-date") + "\n";
        emitProgress(callbacks, 0.15 + (0.75 * (static_cast<double>(repoIndex + 1) / static_cast<double>(totalRepoCount))));
      }

      auto capturedCallbacks = callbacks;
      auto capturedReport = std::move(report);
      auto capturedUpdates = updatesAvailable;

      util::dispatchCallback([capturedCallbacks, capturedReport, capturedUpdates, cancelled]() {
        emitProgress(capturedCallbacks, 1.0);
        if (cancelled) {
          reportModelTaskCancelled(capturedCallbacks, "update check");
          if (capturedCallbacks.onReport) {
            capturedCallbacks.onReport(capturedReport);
          }
          if (capturedCallbacks.onUpdateCheckComplete) {
            capturedCallbacks.onUpdateCheckComplete(true);
          }
          return;
        }

        if (capturedCallbacks.onStatus) {
          capturedCallbacks.onStatus(
              "Models: update check complete (" + std::to_string(capturedUpdates) + " updates)");
        }
        if (capturedCallbacks.onTaskHistory) {
          capturedCallbacks.onTaskHistory("Model update check completed");
        }
        if (capturedCallbacks.onReport) {
          capturedCallbacks.onReport(capturedReport);
        }
        if (capturedCallbacks.onUpdateCheckComplete) {
          capturedCallbacks.onUpdateCheckComplete(false);
        }
      });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new UpdateCheckJob(std::move(modelIds), registryPath, &cancelFlag, callbacks_, hubOps_), true);
}

void ModelController::verifyIntegrity() {
  const auto registryPath = modelHubRoot_ / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array() || registry.empty()) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: no installed hub models");
    }
    return;
  }

  std::string report = "Model Integrity & Licenses\n";
  report += "Registry: " + registryPath.string() + "\n\n";
  int validCount = 0;
  int missingCount = 0;

  for (const auto& item : registry) {
    if (!item.is_object()) {
      continue;
    }
    const std::string modelId = item.value("modelId", item.value("repoId", ""));
    const std::string source = item.value("source", "huggingface");
    const std::filesystem::path installPath(item.value("installPath", ""));
    const std::filesystem::path primaryFile = installPath / item.value("primaryFile", "");
    std::error_code error;
    const bool present = std::filesystem::is_regular_file(primaryFile, error) && !error;
    if (present) {
      ++validCount;
    } else {
      ++missingCount;
    }
    report += "- " + modelId +
              " source=" + source +
              " integrity=" + std::string(present ? "ok" : "missing") +
              " license=" + item.value("license", "unknown") +
              " source=" + item.value("sourceUrl", "") + "\n";
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: integrity check complete");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Model integrity check completed");
  }
  report += "\nSummary: ok=" + std::to_string(validCount) + " missing=" + std::to_string(missingCount) + "\n";
  if (callbacks_.onReport) {
    callbacks_.onReport(report);
  }
}

} // namespace automix::app
