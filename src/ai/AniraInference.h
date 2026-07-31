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

namespace automix::ai {

class AniraInference final : public IModelInference {
 public:
  AniraInference();
  ~AniraInference() noexcept override;

  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& modelPath) override;
  InferenceResult run(const InferenceRequest& request) const override;

  [[nodiscard]] std::vector<std::string> detectAvailableProviders() const;
  void setExecutionProviderPreference(std::string provider);
  [[nodiscard]] std::string activeExecutionProvider() const;
  void setMaxInferenceTimeMs(float ms);
  void setNumParallelProcessors(unsigned int n);

 private:
  struct NativeState;

  std::string resolveBestProvider() const;

  bool loaded_ = false;
  bool aniraAvailable_ = false;
  bool warmupRan_ = false;

  std::filesystem::path modelPath_;
  std::string requestedProvider_ = "auto";
  std::string activeProvider_ = gpu::kProviderCpu;
  float maxInferenceTimeMs_ = 10.0f;
  unsigned int numParallelProcessors_ = 0;

  mutable std::mutex nativeMutex_;
  std::shared_ptr<NativeState> nativeState_;

  mutable std::atomic<uint64_t> inferenceCalls_{0};
  mutable std::atomic<uint64_t> providerFallbacks_{0};
  mutable std::atomic<uint64_t> cumulativeInferenceMicros_{0};
};

} // namespace automix::ai
