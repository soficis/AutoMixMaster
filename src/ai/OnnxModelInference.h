#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "ai/IModelInference.h"

namespace automix::ai {

class OnnxModelInference final : public IModelInference {
 public:
  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& modelPath) override;
  InferenceResult run(const InferenceRequest& request) const override;

  std::vector<InferenceResult> runBatch(const std::vector<InferenceRequest>& requests) const;

  void setExecutionProviderPreference(std::string provider);
  void setGraphOptimizationEnabled(bool enabled);
  void setWarmupEnabled(bool enabled);
  void setPreferQuantizedVariants(bool enabled);
  void setThreadConfiguration(int intraOpThreads, int interOpThreads);
  void setProfilingEnabled(bool enabled);
  void setPreferredPrecision(std::string precision);

  [[nodiscard]] std::string activeExecutionProvider() const;
  [[nodiscard]] std::string backendDiagnostics() const;

 private:
  std::string resolveExecutionProvider() const;
  bool supportsExecutionProvider(const std::string& provider) const;
  void warmupIfNeeded();

  bool loaded_ = false;
  bool graphOptimizationEnabled_ = true;
  bool warmupEnabled_ = true;
  bool preferQuantizedVariants_ = true;
  mutable bool warmupRan_ = false;

  std::filesystem::path modelPath_;
  std::optional<size_t> expectedFeatureCount_;
  std::vector<std::string> allowedTasks_;
  std::vector<std::string> availableExecutionProviders_;
  std::string requestedExecutionProvider_ = "auto";
  std::string activeExecutionProvider_ = "cpu";
  std::string preferredPrecision_ = "auto";
  int intraOpThreads_ = 0;
  int interOpThreads_ = 0;
  bool profilingEnabled_ = false;
  std::string diagnostics_;

  mutable std::mutex scratchMutex_;
  mutable std::vector<double> preallocatedNormalized_;
  mutable std::atomic<uint64_t> inferenceCalls_ {0};
  mutable std::atomic<uint64_t> batchCalls_ {0};
  mutable std::atomic<uint64_t> providerFallbacks_ {0};
  mutable std::atomic<uint64_t> cumulativeInferenceMicros_ {0};
  mutable std::atomic<uint64_t> warmupDurationMillis_ {0};
};

} // namespace automix::ai
