#include "ai/TensorTypes.h"

namespace automix::ai {

std::optional<std::size_t> elementCount(const TensorSpec& spec) {
  std::size_t count = 1;
  int dynamicAxes = 0;
  for (const auto dim : spec.dims) {
    if (dim == -1) {
      ++dynamicAxes;
      continue;
    }
    if (dim < 0) {
      return std::nullopt;
    }
    count *= static_cast<std::size_t>(dim);
  }
  if (dynamicAxes > 1) {
    return std::nullopt;
  }
  return count;
}

bool shapesMatch(const TensorSpec& expected, const TensorSpec& actual) {
  if (expected.dims.size() != actual.dims.size()) {
    return false;
  }
  for (std::size_t axis = 0; axis < expected.dims.size(); ++axis) {
    if (expected.dims[axis] != -1 && expected.dims[axis] != actual.dims[axis]) {
      return false;
    }
  }
  return true;
}

std::string describeShape(const std::vector<int64_t>& dims) {
  std::string text = "[";
  for (std::size_t axis = 0; axis < dims.size(); ++axis) {
    if (axis > 0) {
      text += ", ";
    }
    text += std::to_string(dims[axis]);
  }
  text += "]";
  return text;
}

std::string describeDtype(const TensorElementType type) {
  switch (type) {
    case TensorElementType::Float32:
      return "float32";
  }
  return "unknown";
}

} // namespace automix::ai
