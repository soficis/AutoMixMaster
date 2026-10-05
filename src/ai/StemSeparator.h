#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "domain/Stem.h"

namespace automix::ai {

class StemSeparator final {
 public:
  struct SeparationOptions {
    std::optional<int> targetStemCount;
    std::optional<size_t> gpuMemoryBudgetMb;
    std::optional<int> maxStreams;
    // Set from RenderSettings::tensorSeparationEnabled. When the model root holds
    // a pack with a tensor_contract, separation runs through SeparationRunner;
    // any failure falls back to the existing path. Off: behaviour is unchanged.
    bool useTensorModel = false;
    // Tensor path only, called from the separating thread. JUCE-free so the
    // caller owns any message-thread marshalling.
    std::function<void(int done, int total)> tensorProgress;
    // Tensor path only. Polled between chunks and during inference, possibly
    // from a watcher thread: must be thread-safe. True stops separation with
    // SeparationResult::cancelled and no fallback.
    std::function<bool()> cancelRequested;
    // Tensor path only: "auto" (GPU when the runtime can open one, else CPU),
    // "cpu", or a provider name such as "cuda".
    std::string executionProvider = "auto";
  };

  struct SeparationQaMetrics {
    double energyLeakage = 0.0;
    double residualDistortion = 0.0;
    double transientRetention = 0.0;
  };

  struct SeparationResult {
    bool success = false;
    bool usedModel = false;
    // The user stopped it; success is false and nothing was written.
    bool cancelled = false;
    int stemVariantCount = 0;
    std::vector<domain::Stem> stems;
    std::vector<std::filesystem::path> generatedFiles;
    std::filesystem::path qaReportPath;
    SeparationQaMetrics qaMetrics;
    std::string logMessage;
  };

  explicit StemSeparator(std::filesystem::path modelRoot = "assets/models/stem-separator");

  [[nodiscard]] bool isModelAvailable() const;
  // True when the model root is a pack carrying a tensor_contract whose model
  // file exists. Says nothing about whether ONNX Runtime can open it.
  [[nodiscard]] bool isTensorModelAvailable() const;
  SeparationResult separate(const std::filesystem::path& mixPath,
                            const std::filesystem::path& outputDir) const;
  SeparationResult separate(const std::filesystem::path& mixPath,
                            const std::filesystem::path& outputDir,
                            const SeparationOptions& options) const;

 private:
  [[nodiscard]] std::filesystem::path resolveModelPath() const;
  std::filesystem::path modelRoot_;
};

} // namespace automix::ai
