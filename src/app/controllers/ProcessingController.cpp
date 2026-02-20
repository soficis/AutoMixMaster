#include "app/controllers/ProcessingController.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "ai/AutoMasterStrategyAI.h"
#include "ai/AutoMixStrategyAI.h"
#include "ai/IModelInference.h"
#include "ai/OnnxModelInference.h"
#include "ai/RtNeuralInference.h"
#include "automaster/OriginalMixReference.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/BatchTypes.h"
#include "engine/AudioFileIO.h"
#include "engine/AudioResampler.h"
#include "engine/BatchQueueRunner.h"
#include "engine/OfflineRenderPipeline.h"
#include "util/StringUtils.h"

namespace automix::app {
namespace {

using ::automix::util::toLower;
using ::automix::util::toJuceText;

std::unique_ptr<ai::IModelInference> createInferenceBackend(const ai::ModelPack* pack,
                                                            const std::string& providerPreference,
                                                            std::string* diagnosticsOut) {
  if (pack == nullptr) {
    return nullptr;
  }

  std::unique_ptr<ai::IModelInference> backend;
  const auto engine = toLower(pack->engine);
  if (engine.find("onnx") != std::string::npos || engine == "unknown") {
    auto onnx = std::make_unique<ai::OnnxModelInference>();

    auto resolvedProvider = providerPreference;
    if ((resolvedProvider.empty() || toLower(resolvedProvider) == "auto") && !pack->providerAffinity.empty()) {
      resolvedProvider = pack->providerAffinity.front();
    }
    onnx->setExecutionProviderPreference(resolvedProvider);
    onnx->setGraphOptimizationEnabled(true);
    onnx->setWarmupEnabled(true);
    onnx->setPreferQuantizedVariants(toLower(pack->preferredPrecision) != "fp32");
    onnx->setPreferredPrecision(pack->preferredPrecision.empty() ? "auto" : pack->preferredPrecision);
    onnx->setThreadConfiguration(pack->defaultIntraOpThreads.value_or(0), pack->defaultInterOpThreads.value_or(0));
    onnx->setProfilingEnabled(pack->enableProfiling);
    backend = std::move(onnx);
  }
  if (!backend && engine.find("rtneural") != std::string::npos) {
    backend = std::make_unique<ai::RtNeuralInference>();
  }
  if (!backend) {
    backend = std::make_unique<ai::RtNeuralInference>();
  }

  const auto modelPath = pack->rootPath / pack->modelFile;
  if (!backend->loadModel(modelPath)) {
    return nullptr;
  }

  if (diagnosticsOut != nullptr) {
    if (const auto* onnx = dynamic_cast<const ai::OnnxModelInference*>(backend.get()); onnx != nullptr) {
      *diagnosticsOut = onnx->backendDiagnostics();
    }
  }

  return backend;
}

} // namespace

ProcessingController::ProcessingController(juce::ThreadPool& threadPool, Callbacks callbacks)
    : threadPool_(threadPool), callbacks_(std::move(callbacks)) {}

void ProcessingController::runAutoMix(const domain::Session& session,
                                      const std::optional<ai::ModelPack>& mixPack,
                                      std::atomic_bool& cancelFlag) {
  struct AutoMixJob final : juce::ThreadPoolJob {
    domain::Session session;
    std::optional<ai::ModelPack> mixPack;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;

    AutoMixJob(domain::Session sess, std::optional<ai::ModelPack> pack,
               std::atomic_bool* cancel, Callbacks cb)
        : juce::ThreadPoolJob("AutoMixJob"),
          session(std::move(sess)),
          mixPack(std::move(pack)),
          cancelFlag(cancel),
          callbacks(std::move(cb)) {}

    JobStatus runJob() override {
      std::vector<analysis::StemAnalysisEntry> analysisEntries;
      std::optional<domain::MixPlan> plan;
      juce::String reportText;
      juce::String errorText;
      bool cancelled = false;

      try {
        if (callbacks.onStatus) {
          callbacks.onStatus("Auto Mix: analyzing...");
        }
        analysis::StemAnalyzer analyzer;
        analysisEntries = analyzer.analyzeSession(session);

        if (cancelFlag != nullptr && cancelFlag->load()) {
          cancelled = true;
        }

        if (!cancelled) {
          if (callbacks.onStatus) {
            callbacks.onStatus("Auto Mix: building plan...");
          }
          automix::HeuristicAutoMixStrategy heuristicMix;
          const auto heuristicPlan = heuristicMix.buildPlan(session, analysisEntries, 1.0);
          plan = heuristicPlan;

          ai::AutoMixStrategyAI aiMix;
          std::string backendDiagnostics;
          std::unique_ptr<ai::IModelInference> inference;
          if (mixPack.has_value()) {
            inference = createInferenceBackend(&mixPack.value(), session.renderSettings.gpuExecutionProvider, &backendDiagnostics);
          }

          if (inference != nullptr) {
            auto aiPlan = aiMix.buildPlan(session, analysisEntries, heuristicPlan, inference.get());
            if (mixPack.has_value()) {
              aiPlan.decisionLog.push_back("AI pack: " + mixPack->id + " license=" + mixPack->licenseId);
            }
            if (!backendDiagnostics.empty()) {
              aiPlan.decisionLog.push_back("Inference backend: " + backendDiagnostics);
            }
            plan = std::move(aiPlan);
          }

          reportText = juce::String("Analysis report JSON:\n") + analyzer.toJsonReport(analysisEntries);
          if (plan.has_value()) {
            reportText += juce::String("\n\nMix decisions:\n") + toJuceText(plan->decisionLog);
          }
        }
      } catch (const std::exception& error) {
        errorText = "Auto Mix failed:\n" + juce::String(error.what());
      } catch (...) {
        errorText = "Auto Mix failed:\nUnknown error";
      }

      AutoMixResult result;
      result.cancelled = cancelled || (cancelFlag != nullptr && cancelFlag->load());
      result.analysisEntries = std::move(analysisEntries);
      result.mixPlan = std::move(plan);
      result.reportText = reportText;
      result.errorText = errorText;

      auto capturedCallbacks = callbacks;
      juce::MessageManager::callAsync([capturedCallbacks, result = std::move(result)]() mutable {
        if (capturedCallbacks.onAutoMixComplete) {
          capturedCallbacks.onAutoMixComplete(std::move(result));
        }
      });

      return jobHasFinished;
    }
  };

  threadPool_.addJob(
      new AutoMixJob(session, mixPack, &cancelFlag, callbacks_),
      true);
}

void ProcessingController::runAutoMaster(const domain::Session& session,
                                         const domain::RenderSettings& settings,
                                         const domain::MasterPreset preset,
                                         const std::optional<ai::ModelPack>& masterPack,
                                         std::atomic_bool& cancelFlag) {
  struct AutoMasterJob final : juce::ThreadPoolJob {
    domain::Session session;
    domain::RenderSettings settings;
    domain::MasterPreset preset;
    std::optional<ai::ModelPack> masterPack;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;

    AutoMasterJob(domain::Session sess, domain::RenderSettings sett,
                  domain::MasterPreset pres, std::optional<ai::ModelPack> pack,
                  std::atomic_bool* cancel, Callbacks cb)
        : juce::ThreadPoolJob("AutoMasterJob"),
          session(std::move(sess)),
          settings(std::move(sett)),
          preset(pres),
          masterPack(std::move(pack)),
          cancelFlag(cancel),
          callbacks(std::move(cb)) {}

    JobStatus runJob() override {
      domain::MasterPlan masterPlan;
      engine::AudioBuffer rawMixBuffer;
      engine::AudioBuffer previewMaster;
      automaster::MasteringReport previewReport;
      juce::String reportAppend;
      juce::String errorText;
      bool cancelled = false;

      try {
        engine::OfflineRenderPipeline pipeline;

        std::mutex progressMutex;
        auto lastProgressEmit = std::chrono::steady_clock::time_point {};
        double lastProgressFraction = -1.0;
        std::string lastProgressStage;
        auto capturedCallbacks = callbacks;

        const auto rawMix = pipeline.renderRawMix(
            session,
            settings,
            [capturedCallbacks, &progressMutex, &lastProgressEmit, &lastProgressFraction, &lastProgressStage](
                const engine::RenderProgress& progress) {
              bool emit = false;
              bool stageChanged = false;
              {
                std::scoped_lock lock(progressMutex);
                const auto now = std::chrono::steady_clock::now();
                stageChanged = progress.stage != lastProgressStage;
                const bool finalProgress = progress.fraction >= 0.999;
                const bool timeGateOpen =
                    lastProgressEmit.time_since_epoch().count() == 0 ||
                    now - lastProgressEmit >= std::chrono::milliseconds(160);
                const bool deltaGateOpen = std::abs(progress.fraction - lastProgressFraction) >= 0.02;
                emit = stageChanged || finalProgress || (timeGateOpen && deltaGateOpen);
                if (emit) {
                  lastProgressEmit = now;
                  lastProgressFraction = progress.fraction;
                  lastProgressStage = progress.stage;
                }
              }
              if (!emit) {
                return;
              }

              if (progress.stage == "Mix render cache hit") {
                if (capturedCallbacks.onStatus) {
                  capturedCallbacks.onStatus("Auto Master: Using cached mix render (fast path)");
                }
                if (capturedCallbacks.onTaskHistory) {
                  capturedCallbacks.onTaskHistory("Auto Master using cached mix render");
                }
                return;
              }
              if (capturedCallbacks.onStatus) {
                capturedCallbacks.onStatus("Auto Master: " + progress.stage + " " +
                                           std::to_string(static_cast<int>(progress.fraction * 100.0)) + "%");
              }
              if (progress.fraction >= 0.999 || progress.stage != "Summing stem buses") {
                if (capturedCallbacks.onTaskHistory) {
                  capturedCallbacks.onTaskHistory("Auto Master " + progress.stage + " " +
                                                   std::to_string(static_cast<int>(progress.fraction * 100.0)) + "%");
                }
              }
            },
            cancelFlag);

        if (rawMix.cancelled) {
          cancelled = true;
        } else {
          rawMixBuffer = rawMix.mixBuffer;
        }

        if (!cancelled) {
          automaster::HeuristicAutoMasterStrategy autoMasterStrategy;
          analysis::StemAnalyzer analyzer;
          masterPlan = autoMasterStrategy.buildPlan(preset, rawMixBuffer);

          if (session.originalMixPath.has_value()) {
            try {
              engine::AudioFileIO fileIO;
              engine::AudioResampler resampler;
              auto originalMix = fileIO.readAudioFile(session.originalMixPath.value());
              if (originalMix.getSampleRate() != rawMixBuffer.getSampleRate()) {
                originalMix = resampler.resampleLinear(originalMix, rawMixBuffer.getSampleRate());
              }

              automaster::OriginalMixReference referenceTarget;
              masterPlan = referenceTarget.applySoftTarget(masterPlan,
                                                           rawMixBuffer,
                                                           originalMix,
                                                           autoMasterStrategy,
                                                           analyzer);
            } catch (const std::exception& error) {
              reportAppend += "\nOriginal mix target skipped: " + juce::String(error.what());
            }
          }

          std::string backendDiagnostics;
          std::unique_ptr<ai::IModelInference> masterInference;
          if (masterPack.has_value()) {
            masterInference = createInferenceBackend(&masterPack.value(), settings.gpuExecutionProvider, &backendDiagnostics);
          }

          ai::AutoMasterStrategyAI aiMaster;
          if (masterInference != nullptr) {
            const auto mixMetrics = analyzer.analyzeBuffer(rawMixBuffer);
            masterPlan = aiMaster.buildPlan(mixMetrics, masterPlan, masterInference.get());
            if (masterPack.has_value()) {
              masterPlan.decisionLog.push_back("AI pack: " + masterPack->id + " license=" + masterPack->licenseId);
            }
            if (!backendDiagnostics.empty()) {
              masterPlan.decisionLog.push_back("Inference backend: " + backendDiagnostics);
            }
          }

          if (masterInference != nullptr) {
            previewMaster = aiMaster.applyPlan(rawMixBuffer, masterPlan, autoMasterStrategy, &previewReport);
          } else {
            previewMaster = autoMasterStrategy.applyPlan(rawMixBuffer, masterPlan, &previewReport);
          }

          reportAppend += "\nMaster decisions:\n" + toJuceText(masterPlan.decisionLog);
        }
      } catch (const std::exception& error) {
        errorText = "Auto Master failed:\n" + juce::String(error.what());
      } catch (...) {
        errorText = "Auto Master failed:\nUnknown error";
      }

      AutoMasterResult result;
      result.cancelled = cancelled;
      result.masterPlan = std::move(masterPlan);
      result.rawMixBuffer = std::move(rawMixBuffer);
      result.previewMaster = std::move(previewMaster);
      result.previewReport = previewReport;
      result.reportAppend = reportAppend;
      result.errorText = errorText;

      auto capturedCb = callbacks;
      juce::MessageManager::callAsync([capturedCb, result = std::move(result)]() mutable {
        if (capturedCb.onAutoMasterComplete) {
          capturedCb.onAutoMasterComplete(std::move(result));
        }
      });

      return jobHasFinished;
    }
  };

  threadPool_.addJob(
      new AutoMasterJob(session, settings, preset, masterPack, &cancelFlag, callbacks_),
      true);
}

void ProcessingController::runBatch(const std::filesystem::path& inputFolder,
                                    const domain::RenderSettings& baseSettings,
                                    std::atomic_bool& cancelFlag) {
  struct BatchJob final : juce::ThreadPoolJob {
    std::filesystem::path inputFolder;
    domain::RenderSettings baseSettings;
    std::atomic_bool* cancelFlag;
    Callbacks callbacks;

    BatchJob(std::filesystem::path folder, domain::RenderSettings sett,
             std::atomic_bool* cancel, Callbacks cb)
        : juce::ThreadPoolJob("BatchJob"),
          inputFolder(std::move(folder)),
          baseSettings(std::move(sett)),
          cancelFlag(cancel),
          callbacks(std::move(cb)) {}

    JobStatus runJob() override {
      const std::filesystem::path outputFolder = inputFolder / "automix_batch_exports";

      std::vector<domain::BatchItem> items;
      juce::String prepError;
      try {
        std::filesystem::create_directories(outputFolder);
        engine::BatchQueueRunner batchQueueRunner;
        items = batchQueueRunner.buildItemsFromFolder(inputFolder, outputFolder);
      } catch (const std::exception& error) {
        prepError = error.what();
      } catch (...) {
        prepError = "Unknown batch preparation error";
      }

      if (!prepError.isEmpty() || items.empty()) {
        BatchResult result;
        if (!prepError.isEmpty()) {
          result.errorText = "Batch preparation error:\n" + prepError;
        } else {
          result.errorText = "Batch folder has no supported audio files";
        }

        auto capturedCallbacks = callbacks;
        juce::MessageManager::callAsync([capturedCallbacks, result = std::move(result)]() mutable {
          if (capturedCallbacks.onBatchComplete) {
            capturedCallbacks.onBatchComplete(std::move(result));
          }
        });
        return jobHasFinished;
      }

      if (callbacks.onStatus) {
        callbacks.onStatus("Batch started");
      }

      domain::BatchJob job;
      job.items = std::move(items);
      job.settings.outputFolder = outputFolder;
      job.settings.parallelAnalysis = true;
      const int hardwareThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
      job.settings.analysisThreads = std::max(1, hardwareThreads / 2);
      job.settings.renderParallelism = std::max(1, hardwareThreads / 2);
      job.settings.renderSettings = baseSettings;

      engine::BatchQueueRunner runner;
      std::mutex progressMutex;
      auto lastProgressEmit = std::chrono::steady_clock::time_point {};
      size_t lastItemIndex = std::numeric_limits<size_t>::max();
      double lastProgress = -1.0;
      std::string lastStage;
      auto capturedCallbacks = callbacks;

      const auto batchResult = runner.process(
          job,
          [capturedCallbacks, &progressMutex, &lastProgressEmit, &lastItemIndex, &lastProgress, &lastStage](
              const size_t itemIndex, const double progress, const std::string& stage) {
            bool emit = false;
            {
              std::scoped_lock lock(progressMutex);
              const auto now = std::chrono::steady_clock::now();
              const bool itemChanged = itemIndex != lastItemIndex;
              const bool stageChanged = stage != lastStage;
              const bool finalProgress = progress >= 0.999;
              const bool timeGateOpen =
                  lastProgressEmit.time_since_epoch().count() == 0 ||
                  now - lastProgressEmit >= std::chrono::milliseconds(220);
              const bool deltaGateOpen = std::abs(progress - lastProgress) >= 0.03;
              emit = itemChanged || stageChanged || finalProgress || (timeGateOpen && deltaGateOpen);
              if (emit) {
                lastProgressEmit = now;
                lastItemIndex = itemIndex;
                lastProgress = progress;
                lastStage = stage;
              }
            }
            if (!emit) {
              return;
            }

            const auto statusMsg = "Batch item " + std::to_string(itemIndex + 1) +
                                   " " + stage + " (" + std::to_string(static_cast<int>(progress * 100.0)) + "%)";
            if (capturedCallbacks.onStatus) {
              capturedCallbacks.onStatus(statusMsg);
            }
            if (capturedCallbacks.onTaskHistory) {
              capturedCallbacks.onTaskHistory("Batch item " + std::to_string(itemIndex + 1) +
                                               " " + stage + " " + std::to_string(static_cast<int>(progress * 100.0)) + "%");
            }
          },
          cancelFlag);

      juce::String summary;
      summary << "Batch completed\n";
      summary << "Completed: " << batchResult.completed << "\n";
      summary << "Failed: " << batchResult.failed << "\n";
      summary << "Cancelled: " << batchResult.cancelled << "\n";

      for (const auto& item : job.items) {
        summary << item.session.sessionName << " -> " << juce::String(item.outputPath.string()) << " ["
                << juce::String(domain::toString(item.status)) << "]";
        if (!item.error.empty()) {
          summary << " error=" << juce::String(item.error);
        }
        summary << "\n";
      }

      BatchResult result;
      result.summary = summary;

      auto finalCallbacks = callbacks;
      juce::MessageManager::callAsync([finalCallbacks, result = std::move(result)]() mutable {
        if (finalCallbacks.onBatchComplete) {
          finalCallbacks.onBatchComplete(std::move(result));
        }
      });

      return jobHasFinished;
    }
  };

  threadPool_.addJob(
      new BatchJob(inputFolder, baseSettings, &cancelFlag, callbacks_),
      true);
}

} // namespace automix::app
