#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include "ai/ModelManager.h"
#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "domain/MasterPlan.h"
#include "domain/MixPlan.h"
#include "domain/Session.h"
#include "engine/AudioBuffer.h"

namespace automix::app {

struct AutoMixResult {
  bool cancelled = false;
  std::vector<analysis::StemAnalysisEntry> analysisEntries;
  std::optional<domain::MixPlan> mixPlan;
  juce::String reportText;
  juce::String errorText;
};

struct AutoMasterResult {
  bool cancelled = false;
  domain::MasterPlan masterPlan;
  engine::AudioBuffer rawMixBuffer;
  engine::AudioBuffer previewMaster;
  automaster::MasteringReport previewReport;
  juce::String reportAppend;
  juce::String errorText;
};

struct BatchResult {
  juce::String summary;
  juce::String errorText;
};

class ProcessingController {
 public:
  struct Callbacks {
    std::function<void(const std::string&)> onStatus;
    std::function<void(const std::string&)> onTaskHistory;
    std::function<void(AutoMixResult)> onAutoMixComplete;
    std::function<void(AutoMasterResult)> onAutoMasterComplete;
    std::function<void(BatchResult)> onBatchComplete;
  };

  ProcessingController(juce::ThreadPool& threadPool, Callbacks callbacks);

  void runAutoMix(const domain::Session& session,
                  const std::optional<ai::ModelPack>& mixPack,
                  std::atomic_bool& cancelFlag);

  void runAutoMaster(const domain::Session& session,
                     const domain::RenderSettings& settings,
                     domain::MasterPreset preset,
                     const std::optional<ai::ModelPack>& masterPack,
                     std::atomic_bool& cancelFlag);

  void runBatch(const std::filesystem::path& inputFolder,
                const domain::RenderSettings& baseSettings,
                std::atomic_bool& cancelFlag);

 private:
  juce::ThreadPool& threadPool_;
  Callbacks callbacks_;
};

} // namespace automix::app
