#include "ai/OnnxTensorInference.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef AUTOMIX_HAS_NATIVE_ORT
#define AUTOMIX_HAS_NATIVE_ORT 0
#endif

#if AUTOMIX_HAS_NATIVE_ORT
#include <onnxruntime_cxx_api.h>
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

struct OnnxTensorInference::NativeState {
#if AUTOMIX_HAS_NATIVE_ORT
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
#endif
};

OnnxTensorInference::OnnxTensorInference() = default;
OnnxTensorInference::~OnnxTensorInference() noexcept = default;

void OnnxTensorInference::setTensorContract(std::optional<TensorContract> contract) { contract_ = std::move(contract); }

bool OnnxTensorInference::isAvailable() const { return nativeState_ != nullptr; }

bool OnnxTensorInference::usingNativeSession() const { return nativeState_ != nullptr; }

std::string OnnxTensorInference::backendDiagnostics() const { return diagnostics_; }

std::vector<TensorSpec> OnnxTensorInference::inputSpecs() const { return inputs_; }

std::vector<TensorSpec> OnnxTensorInference::outputSpecs() const { return outputs_; }

void OnnxTensorInference::unload(std::string diagnostics) {
  nativeState_.reset();
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
  auto state = std::make_unique<NativeState>();
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  try {
    state->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AutoMixMasterTensor");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(_WIN32)
    state->session = std::make_unique<Ort::Session>(*state->env, modelPath.wstring().c_str(), options);
#else
    state->session = std::make_unique<Ort::Session>(*state->env, modelPath.string().c_str(), options);
#endif

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
    // Session creation is where a missing external-data sidecar surfaces; ORT's
    // message names the file it could not open.
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
  diagnostics_ = "backend=native_onnxruntime; model=" + modelPath.filename().string() +
                 "; inputs=" + std::to_string(inputs_.size()) + "; outputs=" + std::to_string(outputs_.size());
  return true;
#else
  unload("ONNX tensor load failed for '" + modelPath.string() +
         "': this build has no native ONNX Runtime (AUTOMIX_HAS_NATIVE_ORT is off).");
  return false;
#endif
}

TensorInferenceResult OnnxTensorInference::run(const std::vector<TensorBinding>& inputs) const {
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

    auto produced = nativeState_->session->Run(Ort::RunOptions{nullptr},
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
  result.logMessage = "OnnxTensorInference: this build has no native ONNX Runtime.";
  return result;
#endif
}

} // namespace automix::ai
