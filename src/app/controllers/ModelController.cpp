#include "app/controllers/ModelController.h"

#include <fstream>
#include <system_error>

#include <nlohmann/json.hpp>

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

void ModelController::fetchCatalog(std::atomic_bool& cancelFlag) {
  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: fetching Hugging Face catalog...");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Models catalog fetch started");
  }

  struct CatalogJob final : juce::ThreadPoolJob {
    Callbacks callbacks;
    std::vector<ai::HubModelInfo>* modelsOut;
    std::atomic_bool* cancelFlag;

    CatalogJob(Callbacks cb, std::vector<ai::HubModelInfo>* out, std::atomic_bool* cancel)
        : juce::ThreadPoolJob("FetchCatalog"),
          callbacks(std::move(cb)),
          modelsOut(out),
          cancelFlag(cancel) {}

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
        ai::HuggingFaceModelHub hub;
        ai::HubModelQueryOptions options;
        options.maxResultsPerQuery = 6;
        models = hub.discoverRecommended(options);
        if (models.size() > 20) {
          models.resize(20);
        }
      }

      if (isCancellationRequested()) {
        requestCancellation();
        cancelled = true;
      }

      auto capturedCallbacks = callbacks;
      auto capturedModelsOut = modelsOut;
      util::dispatchCallback(
          [capturedCallbacks, capturedModelsOut, models = std::move(models), cancelled]() mutable {
            if (cancelled) {
              if (capturedCallbacks.onStatus) {
                capturedCallbacks.onStatus("Models: catalog fetch cancelled");
              }
              if (capturedCallbacks.onTaskHistory) {
                capturedCallbacks.onTaskHistory("Models catalog fetch cancelled");
              }
              if (capturedCallbacks.onReport) {
                capturedCallbacks.onReport("Models catalog fetch cancelled.");
              }
            } else {
              *capturedModelsOut = std::move(models);

              if (capturedModelsOut->empty()) {
                if (capturedCallbacks.onStatus) {
                  capturedCallbacks.onStatus("Models: no catalog results");
                }
                if (capturedCallbacks.onTaskHistory) {
                  capturedCallbacks.onTaskHistory("Models catalog returned no results");
                }
                if (capturedCallbacks.onReport) {
                  capturedCallbacks.onReport(
                      "Model browser: no public compatible entries returned by Hugging Face queries.");
                }
              } else {
                if (capturedCallbacks.onStatus) {
                  capturedCallbacks.onStatus("Models: select model to download");
                }
                if (capturedCallbacks.onTaskHistory) {
                  capturedCallbacks.onTaskHistory(
                      "Models catalog loaded (" + std::to_string(capturedModelsOut->size()) + " entries)");
                }
                if (capturedCallbacks.onCatalogReady) {
                  capturedCallbacks.onCatalogReady();
                }
              }
            }

            if (capturedCallbacks.onAsyncTaskComplete) {
              capturedCallbacks.onAsyncTaskComplete();
            }
          });

      return jobHasFinished;
    }
  };

  threadPool_.addJob(new CatalogJob(callbacks_, &discoveredModels_, &cancelFlag), true);
}

void ModelController::installModel(const std::string& repoId, std::atomic_bool& cancelFlag) {
  if (repoId.empty()) {
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: installing " + repoId);
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Model install started: " + repoId);
  }

  struct InstallJob final : juce::ThreadPoolJob {
    std::string repoId;
    std::filesystem::path hubRoot;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;

    InstallJob(std::string id, std::filesystem::path root, std::atomic_bool* cancel, Callbacks cb)
        : juce::ThreadPoolJob("InstallModel"),
          repoId(std::move(id)),
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
      ai::HubInstallResult install;

      if (isCancellationRequested()) {
        requestCancellation();
        cancelled = true;
      }

      if (!cancelled) {
        ai::HuggingFaceModelHub hub;
        ai::HubInstallOptions installOptions;
        installOptions.destinationRoot = hubRoot;
        install = hub.installModel(repoId, installOptions);
      }

      if (isCancellationRequested()) {
        requestCancellation();
        cancelled = true;
      }

      auto capturedCallbacks = callbacks;
      auto capturedRepoId = repoId;
      util::dispatchCallback([capturedCallbacks, capturedRepoId, install, cancelled]() {
        if (cancelled) {
          if (capturedCallbacks.onStatus) {
            capturedCallbacks.onStatus("Model install cancelled");
          }
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory("Model install cancelled: " + capturedRepoId);
          }
          if (capturedCallbacks.onReport) {
            capturedCallbacks.onReport("Model install cancelled: " + capturedRepoId);
          }
        } else {
          if (capturedCallbacks.onStatus) {
            capturedCallbacks.onStatus(install.success ? "Model installed: " + capturedRepoId : "Model install failed");
          }
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory(install.success ? "Model installed: " + capturedRepoId
                                                            : "Model install failed: " + capturedRepoId);
          }

          if (capturedCallbacks.onReport) {
            std::string report;
            report += "Model install: " + capturedRepoId + "\n";
            report += "Revision: " + install.revision + "\n";
            report += "Result: " + std::string(install.success ? "success" : "failed") + "\n";
            report += "Detail: " + install.message + "\n";
            if (!install.installPath.empty()) {
              report += "Path: " + install.installPath.string() + "\n";
            }
            report += "Token usage: env-based token is supported via AUTOMIX_HF_TOKEN/HF_TOKEN/HUGGINGFACE_TOKEN.\n";
            capturedCallbacks.onReport(report);
          }

          if (install.success && capturedCallbacks.onModelPacksChanged) {
            capturedCallbacks.onModelPacksChanged();
          }
        }

        if (capturedCallbacks.onAsyncTaskComplete) {
          capturedCallbacks.onAsyncTaskComplete();
        }
      });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new InstallJob(repoId, modelHubRoot_, &cancelFlag, callbacks_), true);
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
    report += "- " + item.value("repoId", "") +
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
  const auto registryPath = modelHubRoot_ / "install_registry.json";
  const auto registry = loadJsonIfPresent(registryPath);
  if (!registry.is_array() || registry.empty()) {
    if (callbacks_.onStatus) {
      callbacks_.onStatus("Models: no installed hub models");
    }
    return;
  }

  if (callbacks_.onStatus) {
    callbacks_.onStatus("Models: checking updates...");
  }
  if (callbacks_.onTaskHistory) {
    callbacks_.onTaskHistory("Model update check started");
  }

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
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;

    UpdateCheckJob(std::vector<std::string> ids,
                   std::filesystem::path regPath,
                   std::atomic_bool* cancel,
                   Callbacks cb)
        : juce::ThreadPoolJob("CheckUpdates"),
          repoIds(std::move(ids)),
          registryPath(std::move(regPath)),
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
      ai::HuggingFaceModelHub hub;
      std::string report = "Model Update Check\n";
      report += "Registry: " + registryPath.string() + "\n\n";
      int updatesAvailable = 0;

      for (const auto& repoId : repoIds) {
        if (isCancellationRequested()) {
          requestCancellation();
          cancelled = true;
          break;
        }

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

        if (isCancellationRequested()) {
          requestCancellation();
          cancelled = true;
          break;
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

      if (cancelled) {
        report += "\nUpdate check cancelled.\n";
      }

      auto capturedCallbacks = callbacks;
      util::dispatchCallback([capturedCallbacks, report = std::move(report), updatesAvailable, cancelled]() mutable {
        if (cancelled) {
          if (capturedCallbacks.onStatus) {
            capturedCallbacks.onStatus("Models: update check cancelled");
          }
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory("Model update check cancelled");
          }
        } else {
          if (capturedCallbacks.onStatus) {
            capturedCallbacks.onStatus("Models: update check complete (" + std::to_string(updatesAvailable) + " updates)");
          }
          if (capturedCallbacks.onTaskHistory) {
            capturedCallbacks.onTaskHistory("Model update check completed");
          }
        }
        if (capturedCallbacks.onReport) {
          capturedCallbacks.onReport(report);
        }
        if (capturedCallbacks.onAsyncTaskComplete) {
          capturedCallbacks.onAsyncTaskComplete();
        }
      });
      return jobHasFinished;
    }
  };

  threadPool_.addJob(new UpdateCheckJob(std::move(repoIds), registryPath, &cancelFlag, callbacks_), true);
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

void ModelController::dispatchMenuAction(const ModelMenuAction action, std::atomic_bool* cancelFlag) {
  static std::atomic_bool fallbackCancelFlag {false};
  auto* resolvedCancelFlag = cancelFlag != nullptr ? cancelFlag : &fallbackCancelFlag;

  switch (action) {
    case ModelMenuAction::BrowseAndDownload:
      fetchCatalog(*resolvedCancelFlag);
      return;
    case ModelMenuAction::ShowInstalled:
      showInstalled();
      return;
    case ModelMenuAction::CheckUpdates:
      checkUpdates(*resolvedCancelFlag);
      return;
    case ModelMenuAction::VerifyIntegrity:
      verifyIntegrity();
      return;
    case ModelMenuAction::OpenHubFolder: {
      std::error_code error;
      std::filesystem::create_directories(modelHubRoot_, error);
      if (callbacks_.onRevealModelHubFolder) {
        callbacks_.onRevealModelHubFolder(modelHubRoot_);
      }
      return;
    }
  }
}

} // namespace automix::app
