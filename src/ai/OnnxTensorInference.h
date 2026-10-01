#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ai/ITensorInference.h"
#include "ai/ModelPackLoader.h"

namespace automix::ai {

// Ordered execution providers a tensor session tries; always ends with "cpu".
// "auto" walks gpu::providerPriorityChain() over the providers this runtime
// build reports; a named provider goes first if reported, and is otherwise
// read as "any GPU" so the reported ones are still tried; "cpu" is CPU only.
// Reported is not the same as usable (CUDA may lack its DLLs), which is why
// loadModel() treats every non-CPU entry as an attempt that may fail.
std::vector<std::string> tensorProviderCandidates(const std::string& requested,
                                                  const std::vector<std::string>& runtimeProviders);

// True when a GPU tensor session can actually be opened here, proven by
// opening one on a tiny in-memory graph (a reported provider can still lack
// its DLLs or a device). Probed once per process; `providerOut` receives the
// provider that opened. Always false without native ONNX Runtime.
bool gpuTensorSessionAvailable(std::string* providerOut = nullptr);

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
  // "auto" (default), "cpu", or a provider name such as "cuda". Applies to
  // the next loadModel().
  void setExecutionProvider(std::string provider);
  // Provider the loaded session actually runs on ("cpu", "cuda", ...), or
  // empty when nothing is loaded.
  [[nodiscard]] std::string activeExecutionProvider() const;

  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& modelPath) override;
  std::vector<TensorSpec> inputSpecs() const override;
  std::vector<TensorSpec> outputSpecs() const override;
  TensorInferenceResult run(const std::vector<TensorBinding>& inputs) const override;
  // Aborts an in-flight native run within ~25 ms of cancelRequested() turning true.
  TensorInferenceResult runCancellable(const std::vector<TensorBinding>& inputs,
                                       const std::function<bool()>& cancelRequested) const override;

  [[nodiscard]] bool usingNativeSession() const;
  [[nodiscard]] std::string backendDiagnostics() const;

 private:
  struct NativeState;

  void unload(std::string diagnostics);
  TensorInferenceResult runImpl(const std::vector<TensorBinding>& inputs,
                                const std::function<bool()>* cancelRequested) const;

  std::optional<TensorContract> contract_;
  std::string requestedProvider_ = "auto";
  std::string activeProvider_;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  std::string diagnostics_;
  std::unique_ptr<NativeState> nativeState_;
};

} // namespace automix::ai
