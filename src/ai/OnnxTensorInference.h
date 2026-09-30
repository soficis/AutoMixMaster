#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ai/ITensorInference.h"
#include "ai/ModelPackLoader.h"

namespace automix::ai {

// ONNX Runtime backend for tensor graphs. Without the native SDK
// (AUTOMIX_HAS_NATIVE_ORT undefined) it is a deterministic no-op: every load
// fails with a diagnostic and nothing ever reports usedModel == true. There is
// no approximate fallback, because a tensor graph has none that is honest.
class OnnxTensorInference final : public ITensorInference {
 public:
  OnnxTensorInference();
  ~OnnxTensorInference() noexcept override;

  // Checked against the probed graph on the next loadModel(); a mismatch fails
  // the load. Without a contract the graph only has to be float32 throughout.
  void setTensorContract(std::optional<TensorContract> contract);

  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& modelPath) override;
  std::vector<TensorSpec> inputSpecs() const override;
  std::vector<TensorSpec> outputSpecs() const override;
  TensorInferenceResult run(const std::vector<TensorBinding>& inputs) const override;

  [[nodiscard]] bool usingNativeSession() const;
  [[nodiscard]] std::string backendDiagnostics() const;

 private:
  struct NativeState;

  void unload(std::string diagnostics);

  std::optional<TensorContract> contract_;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  std::string diagnostics_;
  std::unique_ptr<NativeState> nativeState_;
};

} // namespace automix::ai
