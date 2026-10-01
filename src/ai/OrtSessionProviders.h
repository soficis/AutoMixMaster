#pragma once

// Native-ORT builds only: include after checking AUTOMIX_HAS_NATIVE_ORT.

#include <memory>
#include <string>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>

#include "ai/GpuProvider.h"

namespace automix::ai {

// Appends the ONNX Runtime execution provider for a canonical provider name
// (see gpu::canonicalProviderName). "cpu", "auto" and "" append nothing: the
// CPU provider is always present. Throws Ort::Exception when this runtime
// build does not contain the provider; a provider whose own dependencies
// (CUDA, cuDNN) are missing may instead only fail at session creation.
inline void appendOrtExecutionProvider(Ort::SessionOptions& options, const std::string& canonical) {
  if (canonical == gpu::kProviderCpu || canonical == "auto" || canonical.empty()) {
    return;
  }

  std::unordered_map<std::string, std::string> providerOptions;
  // CUDA and TensorRT are not accepted by the generic string-keyed
  // AppendExecutionProvider(); they need their dedicated V2 option objects.
  // In a build without them, Create*ProviderOptions fails and throws here.
  if (canonical == gpu::kProviderCuda) {
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
  if (canonical == "tensorrt") {
    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt = nullptr;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt));
    const std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> owned(
        tensorrt, api.ReleaseTensorRTProviderOptions);
    options.AppendExecutionProvider_TensorRT_V2(*tensorrt);
    return;
  }
  if (canonical == gpu::kProviderDirectMl) {
    providerOptions["device_id"] = "0";
    options.AppendExecutionProvider("DML", providerOptions);
    return;
  }
  if (canonical == gpu::kProviderCoreMl) {
    providerOptions["ModelFormat"] = "MLProgram";
    options.AppendExecutionProvider("CoreML", providerOptions);
    return;
  }
  if (canonical == gpu::kProviderAne) {
    // Apple Neural Engine via CoreML with ANE override
    providerOptions["ModelFormat"] = "MLProgram";
    providerOptions["ANEUnits"] = "256";
    options.AppendExecutionProvider("CoreML", providerOptions);
    return;
  }
  if (canonical == gpu::kProviderOpenVino) {
    // OpenVINO provider for Intel NPU / GPU
    providerOptions["device_type"] = "CPU_FP32";
    options.AppendExecutionProvider("OpenVINO", providerOptions);
    return;
  }
}

} // namespace automix::ai
