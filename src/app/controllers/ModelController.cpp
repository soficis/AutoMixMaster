#include "app/controllers/ModelController.h"

#include <fstream>

#include <nlohmann/json.hpp>

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

} // namespace

ModelController::ModelController(ai::ModelManager& modelManager,
                                 juce::ThreadPool& threadPool,
                                 Callbacks callbacks)
    : modelManager_(modelManager),
      threadPool_(threadPool),
      callbacks_(std::move(callbacks)),
      modelHubRoot_(defaultModelHubRoot()) {}

void ModelController::setModelHubRoot(const std::filesystem::path& root) {
  modelHubRoot_ = root;
}

const std::vector<ai::HubModelInfo>& ModelController::discoveredModels() const {
  return discoveredModels_;
}

void ModelController::fetchCatalog() {
  callbacks_.onStatus("Models: fetching Hugging Face catalog...");
  callbacks_.onTaskHistory("Models catalog fetch started");

  struct CatalogJob final : juce::ThreadPoolJob {
    Callbacks callbacks;
    std::vector<ai::HubModelInfo>* modelsOut;

    CatalogJob(Callbacks cb, std::vector<ai::HubModelInfo>* out)
        : juce::ThreadPoolJob("FetchCatalog"), callbacks(std::move(cb)), modelsOut(out) {}

    JobStatus runJob() override {
      ai::HuggingFaceModelHub hub;
      ai::HubModelQueryOptions options;
      options.maxResultsPerQuery = 6;
      auto models = hub.discoverRecommended(options);
      if (models.size() > 20) {
        models.resize(20);
      }

      auto capturedCallbacks = callbacks;
      auto capturedModelsOut = modelsOut;
      auto capturedModels = std::move(models);

      juce::MessageManager::callAsync(
          [capturedCallbacks, capturedModelsOut, models = std::move(capturedModels)]() mutable {
            *capturedModelsOut = std::move(models);

            if (capturedModelsOut->empty()) {
              capturedCallbacks.onStatus("Models: no catalog results");
              capturedCallbacks.onTaskHistory("Models catalog returned no results");
              capturedCallbacks.onReport(
                  "Model browser: no public compatible entries returned by Hugging Face queries.");
              return;
            }

            capturedCallbacks.onStatus("Models: select model to download");
            capturedCallbacks.onTaskHistory(
                "Models catalog loaded (" + std::to_string(capturedModelsOut->size()) + " entries)");
            if (capturedCallbacks.onCatalogReady) {
              capturedCallbacks.onCatalogReady();
            }
          });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new CatalogJob(callbacks_, &discoveredModels_), true);
}

void ModelController::installModel(const std::string& repoId) {
  if (repoId.empty()) {
    return;
  }

  callbacks_.onStatus("Models: installing " + repoId);
  callbacks_.onTaskHistory("Model install started: " + repoId);

  struct InstallJob final : juce::ThreadPoolJob {
    std::string repoId;
    std::filesystem::path hubRoot;
    Callbacks callbacks;

    InstallJob(std::string id, std::filesystem::path root, Callbacks cb)
        : juce::ThreadPoolJob("InstallModel"),
          repoId(std::move(id)),
          hubRoot(std::move(root)),
          callbacks(std::move(cb)) {}

    JobStatus runJob() override {
      ai::HuggingFaceModelHub hub;
      ai::HubInstallOptions installOptions;
      installOptions.destinationRoot = hubRoot;
      const auto install = hub.installModel(repoId, installOptions);

      auto capturedCallbacks = callbacks;
      auto capturedRepoId = repoId;
      auto capturedInstall = install;

      juce::MessageManager::callAsync(
          [capturedCallbacks, capturedRepoId, capturedInstall]() {
            if (capturedInstall.success) {
              capturedCallbacks.onStatus("Model installed: " + capturedRepoId);
              capturedCallbacks.onTaskHistory("Model installed: " + capturedRepoId);
            } else {
              capturedCallbacks.onStatus("Model install failed");
              capturedCallbacks.onTaskHistory("Model install failed: " + capturedRepoId);
            }

            std::string report;
            report += "Model install: " + capturedRepoId + "\n";
            report += "Revision: " + capturedInstall.revision + "\n";
            report += "Result: " + std::string(capturedInstall.success ? "success" : "failed") + "\n";
            report += "Detail: " + capturedInstall.message + "\n";
            if (!capturedInstall.installPath.empty()) {
              report += "Path: " + capturedInstall.installPath.string() + "\n";
            }
            report += "Token usage: env-based token is supported via AUTOMIX_HF_TOKEN/HF_TOKEN/HUGGINGFACE_TOKEN.\n";
            capturedCallbacks.onReport(report);

            if (capturedInstall.success && capturedCallbacks.onModelPacksChanged) {
              capturedCallbacks.onModelPacksChanged();
            }
          });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new InstallJob(repoId, modelHubRoot_, callbacks_), true);
}

void ModelController::showInstalled() {
  const auto registryPath = modelHubRoot_ / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array() || registry.empty()) {
    callbacks_.onStatus("Models: no installed hub models");
    callbacks_.onReport("No installed modelhub entries found at " + registryPath.string());
    return;
  }

  std::string report = "Installed Models\n";
  report += "Registry: " + registryPath.string() + "\n\n";
  for (const auto& item : registry) {
    if (!item.is_object()) {
      continue;
    }
    report += "- " + item.value("repoId", "") +
              " rev=" + item.value("revision", "") +
              " useCase=" + item.value("useCase", "") +
              " license=" + item.value("license", "") + "\n";
  }

  callbacks_.onStatus("Models: installed list loaded");
  callbacks_.onTaskHistory("Loaded installed model list");
  callbacks_.onReport(report);
}

void ModelController::checkUpdates() {
  const auto registryPath = modelHubRoot_ / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array() || registry.empty()) {
    callbacks_.onStatus("Models: no installed hub models");
    return;
  }

  callbacks_.onStatus("Models: checking updates...");
  callbacks_.onTaskHistory("Model update check started");

  std::vector<std::string> repoIds;
  repoIds.reserve(registry.size());
  for (const auto& item : registry) {
    if (item.is_object()) {
      const auto repoId = item.value("repoId", "");
      if (!repoId.empty()) {
        repoIds.push_back(repoId);
      }
    }
  }

  struct UpdateCheckJob final : juce::ThreadPoolJob {
    std::vector<std::string> repoIds;
    std::filesystem::path registryPath;
    Callbacks callbacks;

    UpdateCheckJob(std::vector<std::string> ids, std::filesystem::path regPath, Callbacks cb)
        : juce::ThreadPoolJob("CheckUpdates"),
          repoIds(std::move(ids)),
          registryPath(std::move(regPath)),
          callbacks(std::move(cb)) {}

    JobStatus runJob() override {
      ai::HuggingFaceModelHub hub;
      std::string report = "Model Update Check\n";
      report += "Registry: " + registryPath.string() + "\n\n";
      int updatesAvailable = 0;

      for (const auto& repoId : repoIds) {
        const auto localRegistry = loadJsonIfPresent(registryPath);
        std::string localRevision;
        if (localRegistry.is_array()) {
          for (const auto& item : localRegistry) {
            if (item.is_object() && item.value("repoId", "") == repoId) {
              localRevision = item.value("revision", "");
              break;
            }
          }
        }

        const auto remote = hub.modelInfo(repoId);
        if (!remote.has_value()) {
          report += "- " + repoId + ": unable to fetch remote metadata\n";
          continue;
        }

        const bool changed = !localRevision.empty() && localRevision != remote->revision;
        if (changed) {
          ++updatesAvailable;
        }
        report += "- " + repoId +
                  " local=" + localRevision +
                  " remote=" + remote->revision +
                  " status=" + std::string(changed ? "update-available" : "up-to-date") + "\n";
      }

      auto capturedCallbacks = callbacks;
      auto capturedReport = std::move(report);
      auto capturedUpdates = updatesAvailable;

      juce::MessageManager::callAsync(
          [capturedCallbacks, capturedReport, capturedUpdates]() {
            capturedCallbacks.onStatus(
                "Models: update check complete (" + std::to_string(capturedUpdates) + " updates)");
            capturedCallbacks.onTaskHistory("Model update check completed");
            capturedCallbacks.onReport(capturedReport);
          });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new UpdateCheckJob(std::move(repoIds), registryPath, callbacks_), true);
}

void ModelController::verifyIntegrity() {
  const auto registryPath = modelHubRoot_ / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array() || registry.empty()) {
    callbacks_.onStatus("Models: no installed hub models");
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
    const std::string repoId = item.value("repoId", "");
    const std::filesystem::path installPath(item.value("installPath", ""));
    const std::filesystem::path primaryFile = installPath / item.value("primaryFile", "");
    std::error_code error;
    const bool present = std::filesystem::is_regular_file(primaryFile, error) && !error;
    if (present) {
      ++validCount;
    } else {
      ++missingCount;
    }
    report += "- " + repoId +
              " integrity=" + std::string(present ? "ok" : "missing") +
              " license=" + item.value("license", "unknown") +
              " source=" + item.value("sourceUrl", "") + "\n";
  }

  callbacks_.onStatus("Models: integrity check complete");
  callbacks_.onTaskHistory("Model integrity check completed");
  report += "\nSummary: ok=" + std::to_string(validCount) + " missing=" + std::to_string(missingCount) + "\n";
  callbacks_.onReport(report);
}

} // namespace automix::app
