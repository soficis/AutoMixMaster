#pragma once

// Native-ORT builds only: include after checking AUTOMIX_HAS_NATIVE_ORT.

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "ai/GpuProvider.h"
#include "ai/OrtRuntime.h"

namespace automix::ai {

// Configures an Ort::SessionOptions with provider-specific tuning (threads, memory pattern,
// arena, execution mode) AND appends the execution provider.
// This unifies session configuration for both OnnxModelInference and OnnxTensorInference.
inline void configureSessionForProvider(Ort::SessionOptions& options,
                                        const std::string& canonical,
                                        int hardwareThreads = 0) {
  const auto canon = gpu::canonicalProviderName(canonical);
  const auto plan = gpu::sessionConfigPlan(canon, hardwareThreads);
  if (plan.intraOpThreads) options.SetIntraOpNumThreads(*plan.intraOpThreads);
  if (plan.interOpThreads) options.SetInterOpNumThreads(*plan.interOpThreads);
  if (plan.sequentialExecution)
    options.SetExecutionMode(*plan.sequentialExecution ? ExecutionMode::ORT_SEQUENTIAL
                                                       : ExecutionMode::ORT_PARALLEL);
  if (plan.memPattern) { if (*plan.memPattern) options.EnableMemPattern(); else options.DisableMemPattern(); }
  if (plan.cpuArena)   { if (*plan.cpuArena) options.EnableCpuMemArena(); else options.DisableCpuMemArena(); }

  if (canon == gpu::kProviderCpu || canon == "auto" || canon.empty()) {
    return;
  }

  // CUDA and TensorRT are not accepted by the generic string-keyed
  // AppendExecutionProvider(); they need their dedicated V2 option objects.
  // In a build without them, Create*ProviderOptions fails and throws here.
  if (canon == gpu::kProviderCuda) {
    const auto& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
    const std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> owned(
        cuda, api.ReleaseCUDAProviderOptions);
    // kSameAsRequested: the default power-of-two arena growth over-reserves
    // VRAM on large graphs until the driver spills into shared system memory,
    // which on BS-RoFormer made CUDA slower than the CPU.
    const char* keys[] = {"device_id", "cudnn_conv_algo_search", "arena_extend_strategy"};
    const char* values[] = {"0", "DEFAULT", "kSameAsRequested"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda, keys, values, 3));
    options.AppendExecutionProvider_CUDA_V2(*cuda);
    return;
  }
  if (canon == "tensorrt") {
    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt = nullptr;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt));
    const std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> owned(
        tensorrt, api.ReleaseTensorRTProviderOptions);
    options.AppendExecutionProvider_TensorRT_V2(*tensorrt);
    return;
  }

  const auto providerOptions = gpu::providerOptionMap(canon);
  if (canon == gpu::kProviderWebGpu) {
#if defined(__APPLE__)
    options.AppendExecutionProvider("WebGPU", providerOptions);
#elif AUTOMIX_HAS_EP_PLUGIN
    auto& runtime = OrtRuntime::instance();
    const auto& devices = runtime.webGpuDevices();
    if (devices.empty()) {
      throw std::runtime_error("WebGPU execution provider requested but no WebGPU devices are available");
    }
    std::vector<Ort::ConstEpDevice> selectedDevice = {devices[0]};
    options.AppendExecutionProvider_V2(runtime.env(), selectedDevice, providerOptions);
#else
    throw std::runtime_error("WebGPU plugin EP not compiled in");
#endif
    return;
  }
  if (canon == gpu::kProviderDirectMl) {
    options.AppendExecutionProvider("DML", providerOptions);
    return;
  }
  if (canon == gpu::kProviderCoreMl || canon == gpu::kProviderAne) {
    options.AppendExecutionProvider("CoreML", providerOptions);
    return;
  }
  if (canon == gpu::kProviderOpenVino) {
    // OpenVINO provider for Intel NPU / GPU: device_type=CPU_FP32 is a CPU device, and OpenVINO is not shipped
    options.AppendExecutionProvider("OpenVINO", providerOptions);
    return;
  }
}

// Deprecated: prefer configureSessionForProvider() to ensure session tuning is applied.
inline void appendOrtExecutionProvider(Ort::SessionOptions& options, const std::string& canonical) {
  configureSessionForProvider(options, canonical, 0);
}

} // namespace automix::ai
