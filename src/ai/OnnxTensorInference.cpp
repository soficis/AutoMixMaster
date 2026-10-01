#include "ai/OnnxTensorInference.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifndef AUTOMIX_HAS_NATIVE_ORT
#define AUTOMIX_HAS_NATIVE_ORT 0
#endif

#include "ai/GpuProvider.h"
#include "ai/GpuRuntimePack.h"

#if AUTOMIX_HAS_NATIVE_ORT
#include <onnxruntime_cxx_api.h>

#include "ai/OrtSessionProviders.h"
#endif

namespace automix::ai {
namespace {

#if AUTOMIX_HAS_NATIVE_ORT

std::string describeOnnxElementType(const ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "complex64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "complex128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
    default: return "onnx element type " + std::to_string(static_cast<int>(type));
  }
}

// Reads one graph input or output back as a TensorSpec. Anything that is not a
// float32 tensor is refused here, naming the tensor and what it actually is.
bool probeSpec(const std::string& role,
               const std::string& name,
               const Ort::TypeInfo& info,
               std::vector<TensorSpec>& specs,
               std::string& errorOut) {
  if (info.GetONNXType() != ONNX_TYPE_TENSOR) {
    errorOut = "graph " + role + " '" + name + "' is not a tensor (ONNX value type " +
               std::to_string(static_cast<int>(info.GetONNXType())) + "); only float32 tensors are supported.";
    return false;
  }
  const auto tensorInfo = info.GetTensorTypeAndShapeInfo();
  const auto elementType = tensorInfo.GetElementType();
  if (elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    errorOut = "graph " + role + " '" + name + "' has element type " + describeOnnxElementType(elementType) +
               "; only float32 is supported.";
    return false;
  }
  specs.push_back(TensorSpec{name, TensorElementType::Float32, tensorInfo.GetShape()});
  return true;
}

bool isConcrete(const std::vector<int64_t>& dims) {
  for (const auto dim : dims) {
    if (dim <= 0) {
      return false;
    }
  }
  return true;
}

#endif

} // namespace

std::vector<std::string> tensorProviderCandidates(const std::string& requested,
                                                  const std::vector<std::string>& runtimeProviders) {
  std::vector<std::string> reported;
  for (const auto& provider : runtimeProviders) {
    reported.push_back(gpu::canonicalProviderName(provider));
  }
  const auto isReported = [&reported](const std::string& provider) {
    return std::find(reported.begin(), reported.end(), provider) != reported.end();
  };

  std::vector<std::string> candidates;
  const auto wanted = gpu::canonicalProviderName(requested.empty() ? std::string("auto") : requested);
  if (wanted == gpu::kProviderCpu) {
    return {gpu::kProviderCpu};
  }
  // A named GPU provider is tried first; if this runtime lacks it, the request
  // still means "a GPU" (e.g. a DirectML preference on a CUDA build), so the
  // remaining reported GPU providers follow before CPU.
  if (wanted != "auto" && isReported(wanted)) {
    candidates.push_back(wanted);
  }
  for (const auto& provider : gpu::providerPriorityChain()) {
    if (provider != gpu::kProviderCpu && provider != wanted && isReported(provider)) {
      candidates.push_back(provider);
    }
  }
  candidates.emplace_back(gpu::kProviderCpu);
  return candidates;
}

namespace {

#if AUTOMIX_HAS_NATIVE_ORT
void putVarint(std::string& out, std::uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}
std::string varintField(std::uint32_t number, std::uint64_t value) {
  std::string out;
  putVarint(out, static_cast<std::uint64_t>(number) << 3);
  putVarint(out, value);
  return out;
}
std::string bytesField(std::uint32_t number, const std::string& payload) {
  std::string out;
  putVarint(out, (static_cast<std::uint64_t>(number) << 3) | 2);
  putVarint(out, payload.size());
  return out + payload;
}

// y = Identity(x), x and y float[1], IR 8 / opset 17: the smallest graph that
// makes a session initialise its execution provider and device.
std::string identityProbeModel() {
  const auto valueInfo = [](const std::string& name) {
    const auto dim = bytesField(1, varintField(1, 1));             // Dimension.dim_value = 1
    const auto tensorType = varintField(1, 1) + bytesField(2, dim);  // elem_type FLOAT, shape
    return bytesField(1, name) + bytesField(2, bytesField(1, tensorType));
  };
  const auto node = bytesField(1, "x") + bytesField(2, "y") + bytesField(4, "Identity");
  const auto graph = bytesField(1, node) + bytesField(2, "gpu_probe") + bytesField(11, valueInfo("x")) +
                     bytesField(12, valueInfo("y"));
  return varintField(1, 8) + bytesField(8, bytesField(1, "") + varintField(2, 17)) + bytesField(7, graph);
}
#endif

} // namespace

bool runtimeReportsProvider(const std::string& provider) {
#if AUTOMIX_HAS_NATIVE_ORT
  try {
    const auto wanted = gpu::canonicalProviderName(provider);
    for (const auto& reported : Ort::GetAvailableProviders()) {
      if (gpu::canonicalProviderName(reported) == wanted) {
        return true;
      }
    }
  } catch (...) {
  }
#else
  (void)provider;
#endif
  return false;
}
bool gpuTensorSessionAvailable(std::string* providerOut) {
#if AUTOMIX_HAS_NATIVE_ORT
  // Only success is cached: a GPU runtime pack installed later in this
  // process must be able to turn a failed probe into a working one.
  static std::mutex mutex;
  static std::string cached;
  const std::scoped_lock lock(mutex);
  GpuRuntimePack::preload();
  const std::string provider = cached.empty() ? [] {
    std::vector<std::string> runtimeProviders;
    try {
      runtimeProviders = Ort::GetAvailableProviders();
    } catch (...) {
    }
    const auto model = identityProbeModel();
    for (const auto& candidate : tensorProviderCandidates("auto", runtimeProviders)) {
      if (candidate == gpu::kProviderCpu) {
        break;
      }
      try {
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "AutoMixMasterGpuProbe");
        Ort::SessionOptions options;
        appendOrtExecutionProvider(options, candidate);
        Ort::Session session(env, model.data(), model.size(), options);
        return candidate;
      } catch (...) {
      }
    }
    return std::string();
  }() : cached;
  cached = provider;
  if (providerOut != nullptr) {
    *providerOut = provider;
  }
  return !provider.empty();
#else
  if (providerOut != nullptr) {
    providerOut->clear();
  }
  return false;
#endif
}
struct OnnxTensorInference::NativeState {
#if AUTOMIX_HAS_NATIVE_ORT
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
#endif
};

OnnxTensorInference::OnnxTensorInference() = default;
OnnxTensorInference::~OnnxTensorInference() noexcept = default;

void OnnxTensorInference::setTensorContract(std::optional<TensorContract> contract) { contract_ = std::move(contract); }

void OnnxTensorInference::setExecutionProvider(std::string provider) { requestedProvider_ = std::move(provider); }

std::string OnnxTensorInference::activeExecutionProvider() const { return activeProvider_; }

bool OnnxTensorInference::isAvailable() const { return nativeState_ != nullptr; }

bool OnnxTensorInference::usingNativeSession() const { return nativeState_ != nullptr; }

std::string OnnxTensorInference::backendDiagnostics() const { return diagnostics_; }

std::vector<TensorSpec> OnnxTensorInference::inputSpecs() const { return inputs_; }

std::vector<TensorSpec> OnnxTensorInference::outputSpecs() const { return outputs_; }

void OnnxTensorInference::unload(std::string diagnostics) {
  nativeState_.reset();
  activeProvider_.clear();
  inputs_.clear();
  outputs_.clear();
  diagnostics_ = std::move(diagnostics);
}

bool OnnxTensorInference::loadModel(const std::filesystem::path& modelPath) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(modelPath, error) || error) {
    unload("ONNX tensor load failed: missing model file '" + modelPath.string() + "'.");
    return false;
  }

#if AUTOMIX_HAS_NATIVE_ORT
  std::vector<std::string> runtimeProviders;
  try {
    runtimeProviders = Ort::GetAvailableProviders();
  } catch (...) {
  }
  if (gpu::canonicalProviderName(requestedProvider_) != gpu::kProviderCpu) {
    GpuRuntimePack::preload();  // CUDA libraries installed per user, if any
  }
  const auto candidates = tensorProviderCandidates(requestedProvider_, runtimeProviders);

  std::unique_ptr<NativeState> state;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  std::string provider;
  std::string attempts;  // why each provider before the winner was passed over
  for (const auto& candidate : candidates) {
    auto attempt = std::make_unique<NativeState>();
    try {
      attempt->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AutoMixMasterTensor");
      Ort::SessionOptions options;
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      appendOrtExecutionProvider(options, candidate);
#if defined(_WIN32)
      attempt->session = std::make_unique<Ort::Session>(*attempt->env, modelPath.wstring().c_str(), options);
#else
      attempt->session = std::make_unique<Ort::Session>(*attempt->env, modelPath.string().c_str(), options);
#endif
    } catch (const std::exception& exception) {
      if (candidate == gpu::kProviderCpu) {
        // Session creation is where a missing external-data sidecar surfaces; ORT's
        // message names the file it could not open.
        unload("ONNX tensor load failed for '" + modelPath.string() + "': " + attempts + exception.what());
        return false;
      }
      attempts += candidate + " unavailable (" + exception.what() + "); ";
      continue;
    }
    state = std::move(attempt);
    provider = candidate;
    break;
  }

  try {
    Ort::AllocatorWithDefaultOptions allocator;
    std::string probeError;
    for (std::size_t i = 0; i < state->session->GetInputCount(); ++i) {
      const auto name = std::string(state->session->GetInputNameAllocated(i, allocator).get());
      if (!probeSpec("input", name, state->session->GetInputTypeInfo(i), inputs, probeError)) {
        unload("ONNX tensor load failed for '" + modelPath.string() + "': " + probeError);
        return false;
      }
    }
    for (std::size_t i = 0; i < state->session->GetOutputCount(); ++i) {
      const auto name = std::string(state->session->GetOutputNameAllocated(i, allocator).get());
      if (!probeSpec("output", name, state->session->GetOutputTypeInfo(i), outputs, probeError)) {
        unload("ONNX tensor load failed for '" + modelPath.string() + "': " + probeError);
        return false;
      }
    }
  } catch (const std::exception& exception) {
    unload("ONNX tensor load failed for '" + modelPath.string() + "': " + exception.what());
    return false;
  }
  if (contract_.has_value()) {
    std::string contractError;
    if (!checkTensorContract(*contract_, inputs, outputs, contractError)) {
      unload("ONNX tensor load failed for '" + modelPath.string() + "': tensor contract mismatch: " + contractError);
      return false;
    }
  }

  inputs_ = std::move(inputs);
  outputs_ = std::move(outputs);
  nativeState_ = std::move(state);
  activeProvider_ = provider;
  diagnostics_ = "backend=native_onnxruntime; provider=" + provider + "; model=" + modelPath.filename().string() +
                 "; inputs=" + std::to_string(inputs_.size()) + "; outputs=" + std::to_string(outputs_.size()) +
                 (attempts.empty() ? std::string() : "; fallback: " + attempts);
  return true;
#else
  unload("ONNX tensor load failed for '" + modelPath.string() +
         "': this build has no native ONNX Runtime (AUTOMIX_HAS_NATIVE_ORT is off).");
  return false;
#endif
}

TensorInferenceResult OnnxTensorInference::run(const std::vector<TensorBinding>& inputs) const {
  return runImpl(inputs, nullptr);
}

TensorInferenceResult OnnxTensorInference::runCancellable(const std::vector<TensorBinding>& inputs,
                                                          const std::function<bool()>& cancelRequested) const {
  return runImpl(inputs, cancelRequested ? &cancelRequested : nullptr);
}

TensorInferenceResult OnnxTensorInference::runImpl(const std::vector<TensorBinding>& inputs,
                                                   const std::function<bool()>* cancelRequested) const {
  TensorInferenceResult result;
  if (nativeState_ == nullptr) {
    result.logMessage = "OnnxTensorInference: no model loaded (" + diagnostics_ + ")";
    return result;
  }

#if AUTOMIX_HAS_NATIVE_ORT
  if (inputs.size() != inputs_.size()) {
    result.logMessage = "OnnxTensorInference: graph declares " + std::to_string(inputs_.size()) +
                        " input(s) but " + std::to_string(inputs.size()) + " binding(s) were supplied.";
    return result;
  }

  // ORT takes a mutable pointer; copying keeps the caller's bindings const
  // without casting it away. The copy is small next to the inference itself.
  std::vector<std::vector<float>> buffers;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<const char*> inputNames;
  buffers.reserve(inputs_.size());
  shapes.reserve(inputs_.size());
  inputNames.reserve(inputs_.size());
  for (const auto& spec : inputs_) {
    const TensorBinding* binding = nullptr;
    for (const auto& candidate : inputs) {
      if (candidate.expected.name == spec.name) {
        binding = &candidate;
        break;
      }
    }
    if (binding == nullptr) {
      result.logMessage = "OnnxTensorInference: no binding for graph input '" + spec.name + "'.";
      return result;
    }
    const auto& dims = binding->expected.dims;
    if (!isConcrete(dims) || !shapesMatch(spec, binding->expected)) {
      result.logMessage = "OnnxTensorInference: binding for '" + spec.name + "' has shape " + describeShape(dims) +
                          " but the graph declares " + describeShape(spec.dims) + ".";
      return result;
    }
    const auto expectedCount = elementCount(binding->expected);
    if (!expectedCount.has_value() || *expectedCount != binding->data.size()) {
      result.logMessage = "OnnxTensorInference: binding for '" + spec.name + "' carries " +
                          std::to_string(binding->data.size()) + " value(s) for shape " + describeShape(dims) + ".";
      return result;
    }
    buffers.push_back(binding->data);
    shapes.push_back(dims);
    inputNames.push_back(spec.name.c_str());
  }

  std::vector<const char*> outputNames;
  outputNames.reserve(outputs_.size());
  for (const auto& spec : outputs_) {
    outputNames.push_back(spec.name.c_str());
  }

  try {
    const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> values;
    values.reserve(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      values.push_back(Ort::Value::CreateTensor<float>(
          memoryInfo, buffers[i].data(), buffers[i].size(), shapes[i].data(), shapes[i].size()));
    }

    // A watcher polls the cancel check while the native run executes and
    // terminates it; ORT then throws, which the catch below reports.
    Ort::RunOptions runOptions;
    if (activeProvider_ == gpu::kProviderCuda) {
      // Return the run's unused arena memory to the device afterwards. Without
      // it, the second BS-RoFormer chunk grows the arena past a 16 GB card and
      // the driver spills into shared system memory (~5x slower per chunk).
      // Requires arena_extend_strategy=kSameAsRequested (OrtSessionProviders.h).
      runOptions.AddConfigEntry("memory.enable_memory_arena_shrinkage", "gpu:0");
    }
    std::atomic<bool> runFinished{false};
    std::thread watcher;
    if (cancelRequested != nullptr) {
      watcher = std::thread([&runOptions, &runFinished, cancelRequested] {
        while (!runFinished.load()) {
          if ((*cancelRequested)()) {
            runOptions.SetTerminate();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
      });
    }
    struct WatcherJoin {
      std::atomic<bool>& finished;
      std::thread& thread;
      ~WatcherJoin() {
        finished.store(true);
        if (thread.joinable()) {
          thread.join();
        }
      }
    } watcherJoin{runFinished, watcher};

    auto produced = nativeState_->session->Run(runOptions,
                                               inputNames.data(),
                                               values.data(),
                                               values.size(),
                                               outputNames.data(),
                                               outputNames.size());

    for (std::size_t i = 0; i < produced.size(); ++i) {
      const auto& value = produced[i];
      if (!value.IsTensor()) {
        result.logMessage = "OnnxTensorInference: graph output '" + outputs_[i].name + "' is not a tensor.";
        return result;
      }
      const auto info = value.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        result.logMessage = "OnnxTensorInference: graph output '" + outputs_[i].name + "' has element type " +
                            describeOnnxElementType(info.GetElementType()) + "; only float32 is supported.";
        return result;
      }
      const float* data = value.GetTensorData<float>();
      Tensor tensor;
      tensor.spec = TensorSpec{outputs_[i].name, TensorElementType::Float32, info.GetShape()};
      tensor.data.assign(data, data + info.GetElementCount());
      result.outputs.push_back(std::move(tensor));
    }
  } catch (const std::exception& exception) {
    result.outputs.clear();
    result.logMessage = std::string("OnnxTensorInference: run failed: ") + exception.what();
    return result;
  }

  result.usedModel = true;
  result.logMessage = "OnnxTensorInference: ran " + std::to_string(inputs_.size()) + " input(s) -> " +
                      std::to_string(result.outputs.size()) + " output(s).";
  return result;
#else
  static_cast<void>(inputs);
  static_cast<void>(cancelRequested);
  result.logMessage = "OnnxTensorInference: this build has no native ONNX Runtime.";
  return result;
#endif
}

} // namespace automix::ai
