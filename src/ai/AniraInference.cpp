#include "ai/AniraInference.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <sstream>
#include <thread>
#include <utility>

#include "util/StringUtils.h"

#ifndef AUTOMIX_HAS_ANIRA
#define AUTOMIX_HAS_ANIRA 0
#endif

#if AUTOMIX_HAS_ANIRA
#include <anira/anira.h>
#include <onnxruntime_cxx_api.h>
#endif

namespace automix::ai {

namespace {

using ::automix::util::toLower;

#if AUTOMIX_HAS_ANIRA
std::string discoverAniraBackend() {
  try {
    const auto runtimeProviders = Ort::GetAvailableProviders();
    for (const auto& p : runtimeProviders) {
      const auto canon = gpu::canonicalProviderName(p);
      if (canon != gpu::kProviderCpu) {
        return canon;
      }
    }
  } catch (...) {
  }
  return gpu::kProviderCpu;
}
#endif

} // namespace

struct AniraInference::NativeState {
#if AUTOMIX_HAS_ANIRA
  anira::InferenceConfig inferenceConfig;
  anira::PrePostProcessor prePostProcessor;
  std::unique_ptr<anira::InferenceHandler> inferenceHandler;
  std::vector<float> audioInput;
  std::vector<std::string> outputKeys;
#endif
  std::string activeProvider = gpu::kProviderCpu;
};

AniraInference::AniraInference() = default;
AniraInference::~AniraInference() noexcept = default;

bool AniraInference::isAvailable() const { return loaded_; }

bool AniraInference::loadModel(const std::filesystem::path& modelPath) {
  loaded_ = false;
  aniraAvailable_ = false;
  modelPath_.clear();

  std::error_code ec;
  if (!std::filesystem::exists(modelPath, ec) || ec) {
    return false;
  }

  modelPath_ = std::filesystem::absolute(modelPath);
  activeProvider_ = resolveBestProvider();

#if AUTOMIX_HAS_ANIRA
  try {
    const auto gpuProvider = discoverAniraBackend();
    if (gpu::isGpuProvider(gpuProvider)) {
      activeProvider_ = gpuProvider;
    }

    auto nativeState = std::make_shared<NativeState>();
    nativeState->activeProvider = activeProvider_;

    constexpr int64_t kDefaultFeatureDim = 27;
    const std::vector<int64_t> inputShape = {1, kDefaultFeatureDim};
    const std::vector<int64_t> outputShape = {1, 16};

    nativeState->inferenceConfig = anira::InferenceConfig(
        {{modelPath_.string(), anira::InferenceBackend::ONNX}},
        {anira::TensorShape({inputShape}, {outputShape})},
        maxInferenceTimeMs_,
        1,
        false,
        0.f,
        numParallelProcessors_ > 0 ? numParallelProcessors_
                                   : std::max(1u, std::thread::hardware_concurrency() / 2u));

    nativeState->prePostProcessor =
        anira::PrePostProcessor(nativeState->inferenceConfig);

    nativeState->inferenceHandler =
        std::make_unique<anira::InferenceHandler>(
            nativeState->prePostProcessor, nativeState->inferenceConfig);

    nativeState->inferenceHandler->set_inference_backend(anira::InferenceBackend::ONNX);

    const anira::HostConfig hostConfig{
        static_cast<unsigned int>(kDefaultFeatureDim),
        44100
    };
    nativeState->inferenceHandler->prepare(hostConfig);

    nativeState->audioInput.resize(static_cast<size_t>(kDefaultFeatureDim), 0.0f);
    nativeState_ = std::move(nativeState);
    aniraAvailable_ = true;
  } catch (const std::exception&) {
    aniraAvailable_ = false;
    nativeState_.reset();
  }
#else
  (void)numParallelProcessors_;
#endif

  loaded_ = true;
  warmupRan_ = false;
  return true;
}

InferenceResult AniraInference::run(const InferenceRequest& request) const {
  const auto started = std::chrono::steady_clock::now();
  InferenceResult result;

  if (!loaded_) {
    result.usedModel = false;
    result.logMessage = "Anira inference: model not loaded.";
    inferenceCalls_.fetch_add(1);
    return result;
  }

#if AUTOMIX_HAS_ANIRA
  if (aniraAvailable_ && nativeState_ != nullptr &&
      nativeState_->inferenceHandler != nullptr) {
    try {
      std::scoped_lock lock(nativeMutex_);

      const size_t featureCount = request.features.size();
      nativeState_->audioInput.resize(
          std::max<size_t>(1, featureCount), 0.0f);
      for (size_t i = 0; i < featureCount; ++i) {
        nativeState_->audioInput[i] =
            static_cast<float>(std::clamp(request.features[i], -100.0, 100.0));
      }

      float* audioData[] = {nativeState_->audioInput.data()};
      nativeState_->inferenceHandler->process(
          audioData, static_cast<int>(nativeState_->audioInput.size()));

      const size_t outputSize = nativeState_->audioInput.size();
      if (outputSize > 0) {
        result.usedModel = true;
        result.outputs["confidence"] =
            std::clamp(0.5 + static_cast<double>(nativeState_->audioInput[0] % 100) / 200.0,
                       0.0, 1.0);

        for (size_t i = 1; i < outputSize && i < 16; ++i) {
          result.outputs["output_" + std::to_string(i)] =
              static_cast<double>(nativeState_->audioInput[i % outputSize]);
        }
      } else {
        providerFallbacks_.fetch_add(1);
        result.usedModel = false;
        result.logMessage = "Anira inference: empty output.";
      }

      result.logMessage =
          "Anira inference for task '" + request.task +
          "' using provider '" + activeProvider_ + "'.";
    } catch (const std::exception& e) {
      providerFallbacks_.fetch_add(1);
      result.usedModel = false;
      result.logMessage = std::string("Anira inference failed: ") + e.what();
    }

    inferenceCalls_.fetch_add(1);
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    cumulativeInferenceMicros_.fetch_add(
        static_cast<uint64_t>(std::max<int64_t>(0, elapsed)));
    return result;
  }
#else
  (void)cumulativeInferenceMicros_;
#endif

  result.usedModel = false;
  result.logMessage = "Anira inference: backend not available (anira not linked).";
  inferenceCalls_.fetch_add(1);
  return result;
}

std::vector<std::string> AniraInference::detectAvailableProviders() const {
  std::vector<std::string> providers = {gpu::kProviderCpu};
#if AUTOMIX_HAS_ANIRA
  try {
    const auto runtimeProviders = Ort::GetAvailableProviders();
    for (const auto& p : runtimeProviders) {
      providers.push_back(gpu::canonicalProviderName(p));
    }
  } catch (...) {
  }
  std::sort(providers.begin(), providers.end());
  providers.erase(std::unique(providers.begin(), providers.end()),
                  providers.end());
#endif
  return providers;
}

void AniraInference::setExecutionProviderPreference(std::string provider) {
  requestedProvider_ = gpu::canonicalProviderName(std::move(provider));
  if (requestedProvider_.empty()) {
    requestedProvider_ = "auto";
  }
  if (loaded_) {
    activeProvider_ = resolveBestProvider();
  }
}

std::string AniraInference::activeExecutionProvider() const {
  if (requestedProvider_ != "auto") {
    return requestedProvider_;
  }
  return activeProvider_;
}

void AniraInference::setMaxInferenceTimeMs(float ms) {
  maxInferenceTimeMs_ = std::max(1.0f, ms);
}

void AniraInference::setNumParallelProcessors(unsigned int n) {
  numParallelProcessors_ = n;
}

std::string AniraInference::resolveBestProvider() const {
  if (requestedProvider_ != "auto") {
    const auto& chain = gpu::providerPriorityChain();
    if (std::find(chain.begin(), chain.end(), requestedProvider_) !=
        chain.end()) {
      return requestedProvider_;
    }
    return gpu::kProviderCpu;
  }

#if AUTOMIX_HAS_ANIRA
  try {
    const auto runtimeProviders = Ort::GetAvailableProviders();
    const auto& chain = gpu::providerPriorityChain();

    for (const auto& preferred : chain) {
      for (const auto& rp : runtimeProviders) {
        if (gpu::canonicalProviderName(rp) == preferred) {
          return preferred;
        }
      }
    }
  } catch (...) {
  }
#endif

  return gpu::kProviderCpu;
}

} // namespace automix::ai
