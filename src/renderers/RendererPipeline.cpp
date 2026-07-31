#include "renderers/RendererPipeline.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

#include "renderers/FfmpegDiscovery.h"
#include "renderers/PhaseLimiterDiscovery.h"
#include "renderers/PostRendererFactory.h"
#include "renderers/RendererFactory.h"
#include "renderers/RsgainDiscovery.h"
#include "renderers/SoxDiscovery.h"
#include "util/FileUtils.h"
#include "util/StringUtils.h"

namespace automix::renderers {
namespace {

using ::automix::util::toLower;
using ::automix::util::pathFromUtf8;
using ::automix::util::pathToUtf8;

constexpr const char* kBuiltInRendererId = "BuiltIn";
constexpr const char* kPhaseLimiterRendererId = "PhaseLimiter";
constexpr const char* kFfmpegRendererId = "FFmpeg";
constexpr const char* kSoxRendererId = "SoX";
constexpr const char* kRsgainRendererId = "rsgain";

bool isKnownMasterRenderer(const std::string& rendererId) {
  return rendererId == kBuiltInRendererId ||
         rendererId == kPhaseLimiterRendererId ||
         rendererId == kFfmpegRendererId ||
         rendererId == kSoxRendererId;
}

bool isPostRenderer(const std::string& rendererId) {
  return rendererId == kRsgainRendererId;
}

bool isRendererAvailable(const std::string& rendererId) {
  if (rendererId.empty()) {
    return false;
  }
  if (rendererId == kBuiltInRendererId) {
    return true;
  }
  if (rendererId == kPhaseLimiterRendererId) {
    return PhaseLimiterDiscovery{}.find().has_value();
  }
  if (rendererId == kFfmpegRendererId) {
    return FfmpegDiscovery{}.find().has_value();
  }
  if (rendererId == kSoxRendererId) {
    return SoxDiscovery{}.find().has_value();
  }
  if (rendererId == kRsgainRendererId) {
    return RsgainDiscovery{}.find().has_value();
  }
  return true;
}

std::vector<std::string> deduplicateRenderers(const std::vector<std::string>& rendererIds) {
  std::vector<std::string> deduplicated;
  std::unordered_set<std::string> seenIds;
  deduplicated.reserve(rendererIds.size());

  for (const auto& rendererId : rendererIds) {
    if (rendererId.empty()) {
      continue;
    }
    if (!seenIds.insert(rendererId).second) {
      continue;
    }
    deduplicated.push_back(rendererId);
  }
  return deduplicated;
}

std::string resolvePrimaryRenderer(const domain::RenderSettings& settings) {
  auto rendererId = settings.rendererName.empty() ? std::string(kBuiltInRendererId) : settings.rendererName;
  if (rendererId == kRsgainRendererId) {
    rendererId = kBuiltInRendererId;
  }
  return rendererId;
}

std::vector<std::string> resolveSingleRenderer(const domain::RenderSettings& settings) {
  auto rendererId = settings.rendererName.empty() ? std::string(kBuiltInRendererId) : settings.rendererName;
  return {rendererId};
}

std::vector<std::string> resolveMasterThenRsgainChain(const domain::RenderSettings& settings) {
  std::vector<std::string> chain;
  auto primaryRenderer = resolvePrimaryRenderer(settings);
  if (!isRendererAvailable(primaryRenderer)) {
    primaryRenderer = kBuiltInRendererId;
  }
  chain.push_back(primaryRenderer);
  if (isRendererAvailable(kRsgainRendererId)) {
    chain.push_back(kRsgainRendererId);
  }
  return deduplicateRenderers(chain);
}

std::vector<std::string> resolveLogicalAllChain(const domain::RenderSettings& settings) {
  std::vector<std::string> chain;
  chain.reserve(6);

  const auto primaryRenderer = resolvePrimaryRenderer(settings);
  if (!primaryRenderer.empty() && !isPostRenderer(primaryRenderer) && !isKnownMasterRenderer(primaryRenderer)) {
    chain.push_back(primaryRenderer);
  }

  if (isKnownMasterRenderer(primaryRenderer) && isRendererAvailable(primaryRenderer)) {
    chain.push_back(primaryRenderer);
  }

  const std::vector<std::string> canonicalMasterOrder = {
      kBuiltInRendererId,
      kPhaseLimiterRendererId,
      kFfmpegRendererId,
      kSoxRendererId,
  };

  for (const auto& rendererId : canonicalMasterOrder) {
    if (rendererId == primaryRenderer) {
      continue;
    }
    if (!isRendererAvailable(rendererId)) {
      continue;
    }
    chain.push_back(rendererId);
  }

  if (chain.empty()) {
    chain.push_back(kBuiltInRendererId);
  }

  if (isRendererAvailable(kRsgainRendererId)) {
    chain.push_back(kRsgainRendererId);
  }

  return deduplicateRenderers(chain);
}

std::string chainSummary(const std::vector<std::string>& rendererIds) {
  std::string summary;
  for (size_t i = 0; i < rendererIds.size(); ++i) {
    if (i > 0) {
      summary += " -> ";
    }
    summary += rendererIds[i];
  }
  return summary;
}

std::string sanitizePathToken(std::string value) {
  if (value.empty()) {
    return "renderer";
  }
  for (auto& ch : value) {
    const bool isAlphaNum = (ch >= 'a' && ch <= 'z') ||
                            (ch >= 'A' && ch <= 'Z') ||
                            (ch >= '0' && ch <= '9');
    if (!isAlphaNum && ch != '_' && ch != '-') {
      ch = '_';
    }
  }
  return value;
}

domain::Session stageSessionFromAudioPath(const std::filesystem::path& inputAudioPath,
                                          const domain::Session& sourceSession) {
  domain::Session stageSession;
  stageSession.schemaVersion = sourceSession.schemaVersion;
  stageSession.sessionName = sourceSession.sessionName + "_chain";
  stageSession.originalMixPath.reset();
  stageSession.residualBlend = 0.0;
  stageSession.aiStemsEnabled = false;
  stageSession.batchRecursiveEnabled = sourceSession.batchRecursiveEnabled;
  stageSession.selectedMasterPreset = sourceSession.selectedMasterPreset;
  stageSession.selectedPlatformPreset = sourceSession.selectedPlatformPreset;
  stageSession.renderSettings = sourceSession.renderSettings;
  stageSession.mixPlan.reset();
  stageSession.masterPlan = sourceSession.masterPlan;
  stageSession.timeline = sourceSession.timeline;
  stageSession.projectProfileId = sourceSession.projectProfileId;
  stageSession.safetyPolicyId = sourceSession.safetyPolicyId;
  stageSession.preferredStemCount = sourceSession.preferredStemCount;

  domain::Stem stageStem;
  stageStem.id = "chain_input";
  stageStem.name = pathToUtf8(inputAudioPath.stem());
  stageStem.filePath = pathToUtf8(inputAudioPath);
  stageStem.role = domain::StemRole::Music;
  stageStem.origin = domain::StemOrigin::Recorded;
  stageStem.enabled = true;

  stageSession.stems = {stageStem};
  stageSession.buses.clear();
  return stageSession;
}

RenderResult runSingleRenderer(const std::string& rendererId,
                               const domain::Session& session,
                               const domain::RenderSettings& settings,
                               const IRenderer::ProgressCallback& onProgress,
                               std::atomic_bool* cancelFlag) {
  auto renderer = createRenderer(rendererId);
  return renderer->render(session, settings, onProgress, cancelFlag);
}

} // namespace

int effectiveParallelism(const domain::RenderSettings& settings, const int taskCount) {
  if (taskCount <= 1) {
    return 1;
  }
  int requested = settings.renderParallelism;
  if (requested <= 0) {
    requested = settings.processingThreads;
  }
  if (requested <= 0) {
    requested = static_cast<int>(std::thread::hardware_concurrency());
  }
  return std::clamp(requested, 1, taskCount);
}

void parallelFor(const int taskCount,
                 const int parallelism,
                 const std::function<void(int index)>& func) {
  if (taskCount <= 1 || parallelism <= 1) {
    for (int i = 0; i < taskCount; ++i) {
      func(i);
    }
    return;
  }

  const auto numWorkers = std::min(parallelism, taskCount);

  auto worker = [&](const int start, const int end) {
    for (int i = start; i < end; ++i) {
      func(i);
    }
  };

  const int chunkSize = (taskCount + numWorkers - 1) / numWorkers;
  {
    std::vector<std::jthread> workers;
    workers.reserve(static_cast<size_t>(numWorkers));
    for (int w = 0; w < numWorkers; ++w) {
      const int start = w * chunkSize;
      const int end = std::min(start + chunkSize, taskCount);
      if (start >= end) {
        break;
      }
      workers.emplace_back(worker, start, end);
    }
  }
}

std::vector<std::string> resolveRendererChain(const domain::RenderSettings& settings) {
  if (!settings.rendererChainEnabled) {
    return resolveSingleRenderer(settings);
  }

  if (!settings.rendererChain.empty()) {
    auto chain = deduplicateRenderers(settings.rendererChain);
    if (!chain.empty()) {
      if (!isRendererAvailable(chain.front())) {
        chain.front() = kBuiltInRendererId;
      }
      return deduplicateRenderers(chain);
    }
  }

  const auto mode = toLower(settings.rendererChainMode);
  if (mode == "master_then_rsgain") {
    return resolveMasterThenRsgainChain(settings);
  }

  return resolveLogicalAllChain(settings);
}

RenderResult renderWithPipeline(const domain::Session& session,
                                const domain::RenderSettings& settings,
                                const IRenderer::ProgressCallback& onProgress,
                                std::atomic_bool* cancelFlag) {
  auto chain = resolveRendererChain(settings);
  chain = deduplicateRenderers(chain);
  if (chain.empty()) {
    chain.push_back(kBuiltInRendererId);
  }

  if (chain.size() == 1) {
    return runSingleRenderer(chain.front(), session, settings, onProgress, cancelFlag);
  }

  RenderResult finalResult;
  finalResult.rendererName = "RendererChain";
  finalResult.logs.push_back("Renderer chain enabled: " + chainSummary(chain));

  const auto nonce = std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  const auto tempRoot = std::filesystem::temp_directory_path() / ("automix_renderer_chain_" + nonce);
  std::error_code error;
  std::filesystem::create_directories(tempRoot, error);

  std::filesystem::path previousOutputPath;

  for (size_t stageIndex = 0; stageIndex < chain.size(); ++stageIndex) {
    if (cancelFlag != nullptr && cancelFlag->load()) {
      finalResult.cancelled = true;
      finalResult.success = false;
      finalResult.logs.push_back("Renderer chain cancelled before stage " + std::to_string(stageIndex + 1));
      std::filesystem::remove_all(tempRoot, error);
      return finalResult;
    }

    const auto& stageRendererId = chain[stageIndex];
    const bool isLastStage = stageIndex + 1 == chain.size();

    domain::RenderSettings stageSettings = settings;
    stageSettings.rendererName = stageRendererId;
    stageSettings.rendererChainEnabled = false;
    stageSettings.rendererChainMode = "logical_all";
    stageSettings.rendererChain.clear();
    stageSettings.outputFormat = isLastStage ? settings.outputFormat : "wav";

    std::filesystem::path stageOutputPath;
    if (isLastStage) {
      stageOutputPath = settings.outputPath.empty() ? std::filesystem::path("export_master.wav")
                                                    : pathFromUtf8(settings.outputPath);
    } else {
      stageOutputPath = tempRoot / ("stage_" + std::to_string(stageIndex + 1) + "_" +
                                    sanitizePathToken(stageRendererId) + ".wav");
    }
    stageSettings.outputPath = pathToUtf8(stageOutputPath);

    const domain::Session stageSession =
        stageIndex == 0 ? session : stageSessionFromAudioPath(previousOutputPath, session);

    const auto stageNamePrefix = "Stage " + std::to_string(stageIndex + 1) + "/" + std::to_string(chain.size()) +
                                 " [" + stageRendererId + "] ";
    finalResult.logs.push_back(stageNamePrefix + "started");

    RenderResult stageResult;
    auto postRenderer = createPostRenderer(stageRendererId);
    if (postRenderer != nullptr && stageIndex > 0) {
      // Post-renderers (e.g. rsgain) operate in-place on the previous stage's output.
      // Copy the file to the stage destination first, then apply tagging without re-rendering.
      if (previousOutputPath != stageOutputPath) {
        std::filesystem::copy_file(previousOutputPath, stageOutputPath,
                                   std::filesystem::copy_options::overwrite_existing, error);
        if (error) {
          finalResult.success = false;
          finalResult.logs.push_back(stageNamePrefix + "failed to copy audio for post-processing: " + error.message());
          std::filesystem::remove_all(tempRoot, error);
          return finalResult;
        }
      }
      stageResult = postRenderer->applyPostRender(
          stageOutputPath, finalResult.reportPath,
          [onProgress, stageIndex, stageCount = chain.size(), stageNamePrefix](const double progress,
                                                                               const std::string& stage) {
            if (!onProgress) {
              return;
            }
            const double clampedProgress = std::clamp(progress, 0.0, 1.0);
            const double overallProgress = (static_cast<double>(stageIndex) + clampedProgress) /
                                           static_cast<double>(stageCount);
            onProgress(overallProgress, stageNamePrefix + stage);
          },
          cancelFlag);
    } else {
      stageResult = runSingleRenderer(
          stageRendererId,
          stageSession,
          stageSettings,
          [onProgress, stageIndex, stageCount = chain.size(), stageNamePrefix](const double progress, const std::string& stage) {
            if (!onProgress) {
              return;
            }
            const double clampedProgress = std::clamp(progress, 0.0, 1.0);
            const double overallProgress = (static_cast<double>(stageIndex) + clampedProgress) /
                                           static_cast<double>(stageCount);
            onProgress(overallProgress, stageNamePrefix + stage);
          },
          cancelFlag);
    }

    for (auto& logLine : stageResult.logs) {
      finalResult.logs.push_back(stageNamePrefix + std::move(logLine));
    }

    if (stageResult.cancelled) {
      finalResult.cancelled = true;
      finalResult.success = false;
      finalResult.logs.push_back(stageNamePrefix + "cancelled");
      std::filesystem::remove_all(tempRoot, error);
      return finalResult;
    }
    if (!stageResult.success) {
      finalResult.success = false;
      finalResult.logs.push_back(stageNamePrefix + "failed");
      std::filesystem::remove_all(tempRoot, error);
      return finalResult;
    }

    previousOutputPath = pathFromUtf8(stageResult.outputAudioPath);
    finalResult.outputAudioPath = stageResult.outputAudioPath;
    finalResult.reportPath = stageResult.reportPath;
    finalResult.logs.push_back(stageNamePrefix + "completed");
  }

  finalResult.success = true;
  finalResult.cancelled = false;
  finalResult.rendererName = "Chain(" + chainSummary(chain) + ")";

  if (onProgress) {
    onProgress(1.0, "Renderer chain completed");
  }

  std::filesystem::remove_all(tempRoot, error);
  return finalResult;
}

} // namespace automix::renderers
