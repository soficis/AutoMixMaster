#include "ai/FallbackInference.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace automix::ai {

namespace {

// Returns true when the InferenceResult suggests the current execution
// provider failed during inference (as opposed to a configuration rejection
// such as task not allowed or feature-count mismatch).
bool isProviderFailure(const InferenceResult& result) {
  if (!result.usedModel) {
    return true;
  }
  // OnnxModelInference catches ORT exceptions and marks usedModel = true,
  // but the log message carries the fallback indicator.
  const auto& log = result.logMessage;
  if (log.find("deterministic fallback") != std::string::npos ||
      log.find("GPU OOM") != std::string::npos ||
      log.find("device-lost") != std::string::npos ||
      log.find("GPU device-lost") != std::string::npos ||
      log.find("OOM recovery") != std::string::npos) {
    return true;
  }
  return false;
}

} // namespace

FallbackInference::FallbackInference(std::shared_ptr<OnnxModelInference> wrapped)
    : wrapped_(std::move(wrapped)) {
  if (wrapped_) {
    activeProvider_ = wrapped_->activeExecutionProvider();
  } else {
    activeProvider_ = gpu::kProviderCpu;
  }
}

FallbackInference::~FallbackInference() noexcept = default;

bool FallbackInference::isAvailable() const {
  return wrapped_ != nullptr && wrapped_->isAvailable();
}

bool FallbackInference::loadModel(const std::filesystem::path& modelPath) {
  if (!wrapped_) {
    modelLoaded_ = false;
    modelPath_.clear();
    return false;
  }

  // Reset fallback tracking before loading.
  {
    std::scoped_lock lock(fallbackMutex_);
    attemptedProviders_.clear();
    exhaustedProviders_.clear();
    fallbackCount_ = 0;
    retryAttempts_ = 0;
    totalRetryAttempts_ = 0;
  }

  modelPath_ = modelPath;
  modelLoaded_ = wrapped_->loadModel(modelPath);

  if (modelLoaded_) {
    std::scoped_lock lock(fallbackMutex_);
    activeProvider_ = wrapped_->activeExecutionProvider();
  } else {
    modelPath_.clear();
  }

  return modelLoaded_;
}

InferenceResult FallbackInference::run(const InferenceRequest& request) const {
  if (!wrapped_ || !modelLoaded_ || modelPath_.empty()) {
    InferenceResult result;
    result.usedModel = false;
    result.logMessage = "FallbackInference: no model loaded.";
    return result;
  }

  // Fast path — try with the current active provider.
  {
    const auto firstResult = wrapped_->run(request);
    if (!isProviderFailure(firstResult)) {
      return firstResult;
    }

    // Sync any providers that OnnxModelInference has recorded as failed.
    const auto newlyExhausted = wrapped_->failedProviders();
    {
      std::scoped_lock lock(fallbackMutex_);
      for (const auto& ep : newlyExhausted) {
        if (std::find(exhaustedProviders_.begin(), exhaustedProviders_.end(), ep) ==
            exhaustedProviders_.end()) {
          exhaustedProviders_.push_back(ep);
        }
      }
    }
  }

  // Fallback path — walk the priority chain and try each provider.
  const auto& chain = gpu::providerPriorityChain();
  const std::string originalProvider = wrapped_->activeExecutionProvider();
  InferenceResult lastResult;
  bool anyAttemptMade = false;

  for (const auto& candidate : chain) {
    // Skip the provider we just tried.
    if (candidate == originalProvider) {
      continue;
    }

    // Skip exhausted / previously attempted providers.
    {
      std::scoped_lock lock(fallbackMutex_);
      if (std::find(exhaustedProviders_.begin(), exhaustedProviders_.end(), candidate) !=
          exhaustedProviders_.end()) {
        continue;
      }
      if (std::find(attemptedProviders_.begin(), attemptedProviders_.end(), candidate) !=
          attemptedProviders_.end()) {
        continue;
      }
    }

    anyAttemptMade = true;

    // Reload the model with the candidate provider.
    wrapped_->setExecutionProviderPreference(candidate);
    if (!wrapped_->loadModel(modelPath_)) {
      // Provider not available — mark it and move on.
      {
        std::scoped_lock lock(fallbackMutex_);
        attemptedProviders_.push_back(candidate);
        exhaustedProviders_.push_back(candidate);
      }
      continue;
    }

    // Retry loop for this provider.
    for (uint64_t retry = 0; retry <= maxRetriesPerProvider_; ++retry) {
      {
        std::scoped_lock lock(fallbackMutex_);
        ++retryAttempts_;
        ++totalRetryAttempts_;
      }

      lastResult = wrapped_->run(request);
      if (!isProviderFailure(lastResult)) {
        // Success.
        {
          std::scoped_lock lock(fallbackMutex_);
          activeProvider_ = wrapped_->activeExecutionProvider();
          ++fallbackCount_;
        }
        return lastResult;
      }
    }

    // All retries exhausted for this candidate.
    {
      std::scoped_lock lock(fallbackMutex_);
      attemptedProviders_.push_back(candidate);
      exhaustedProviders_.push_back(candidate);
    }
  }

  // If no other provider was attempted fall through to CPU.
  if (!anyAttemptMade) {
    wrapped_->setExecutionProviderPreference(gpu::kProviderCpu);
    wrapped_->loadModel(modelPath_);
  }

  lastResult = wrapped_->run(request);

  {
    std::scoped_lock lock(fallbackMutex_);
    activeProvider_ = wrapped_->activeExecutionProvider();
    ++fallbackCount_;
  }

  std::ostringstream enrich;
  enrich << "FallbackInference: all GPU providers exhausted; using '"
         << activeProvider_ << "'. " << lastResult.logMessage;
  lastResult.logMessage = enrich.str();

  return lastResult;
}

std::vector<std::string> FallbackInference::detectAvailableProviders() const {
  if (!wrapped_) {
    return {gpu::kProviderCpu};
  }
  return wrapped_->detectAvailableProviders();
}

std::string FallbackInference::activeProvider() const {
  std::scoped_lock lock(fallbackMutex_);
  return activeProvider_;
}

uint64_t FallbackInference::providerFallbackCount() const {
  std::scoped_lock lock(fallbackMutex_);
  return fallbackCount_;
}

uint64_t FallbackInference::totalRetryAttempts() const {
  return totalRetryAttempts_.load();
}

void FallbackInference::setMaxRetriesPerProvider(const uint64_t maxRetries) {
  maxRetriesPerProvider_ = maxRetries;
}

} // namespace automix::ai
