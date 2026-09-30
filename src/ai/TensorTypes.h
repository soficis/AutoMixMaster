#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace automix::ai {

// Deliberately a single member. ONNX Runtime has no complex dtype, so complex
// spectra travel as float32 with an explicit real/imag axis; any other graph
// dtype is rejected at load rather than widened silently.
enum class TensorElementType { Float32 };

struct TensorSpec {
  std::string name;
  TensorElementType elementType = TensorElementType::Float32;
  std::vector<int64_t> dims;  // -1 == dynamic axis
};

struct TensorBinding {
  TensorSpec expected;
  std::vector<float> data;
};

struct Tensor {
  TensorSpec spec;
  std::vector<float> data;
};

struct TensorInferenceResult {
  bool usedModel = false;
  std::vector<Tensor> outputs;  // named, never positional
  std::string logMessage;
};

// Product of the static dims, treating a single dynamic axis as 1. nullopt when
// more than one axis is dynamic, because the volume is then undetermined.
std::optional<std::size_t> elementCount(const TensorSpec& spec);

// Equal rank, and every non-dynamic expected dim equals the actual dim.
bool shapesMatch(const TensorSpec& expected, const TensorSpec& actual);

std::string describeShape(const std::vector<int64_t>& dims);
std::string describeDtype(TensorElementType type);

} // namespace automix::ai
