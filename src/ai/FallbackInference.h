#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ai/GpuProvider.h"
#include "ai/IModelInference.h"
#include "ai/OnnxModelInference.h"

namespace automix::ai {

class FallbackInference final : public IModelInference {
 public:
  explicit FallbackInference(std::shared_ptr<OnnxModelInference> wrapped);

  ~FallbackInference() noexcept override;

  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& modelPath) override;
  InferenceResult run(const InferenceRequest& request) const override;

  [[nodiscard]] std::vector<std::string> detectAvailableProviders() const;
  [[nodiscard]] std::string activeProvider() const;
  [[nodiscard]] uint64_t providerFallbackCount() const;
  [[nodiscard]] uint64_t totalRetryAttempts() const;

  void setMaxRetriesPerProvider(uint64_t maxRetries);

  // Prevent copy
  FallbackInference(const FallbackInference&) = delete;
  FallbackInference& operator=(const FallbackInference&) = delete;

 private:
  std::shared_ptr<OnnxModelInference> wrapped_;

  mutable std::mutex fallbackMutex_;
  mutable std::vector<std::string> attemptedProviders_;
  mutable std::vector<std::string> exhaustedProviders_;
  mutable std::string activeProvider_;
  mutable uint64_t fallbackCount_{0};
  mutable uint64_t retryAttempts_{0};
  mutable std::atomic<uint64_t> totalRetryAttempts_{0};
  uint64_t maxRetriesPerProvider_{3};

  std::filesystem::path modelPath_;
  bool modelLoaded_{false};
};

} // namespace automix::ai
