#include <chrono>

#include <catch2/catch_test_macros.hpp>

#include "ai/OnnxModelInference.h"

namespace {

double measureInferenceTime(automix::ai::OnnxModelInference& inference,
                            const automix::ai::InferenceRequest& request,
                            int iterations = 5) {
  double totalMs = 0.0;
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    inference.run(request);
    const auto end = std::chrono::high_resolution_clock::now();
    totalMs += std::chrono::duration<double, std::milli>(end - start).count();
  }
  return totalMs / static_cast<double>(iterations);
}

} // namespace

TEST_CASE("ONNX inference reports availability status", "[benchmark][onnx]") {
  automix::ai::OnnxModelInference inference;
  // Without a loaded model, isAvailable should be false
  REQUIRE_FALSE(inference.isAvailable());
}

TEST_CASE("ONNX inference returns error on unloaded model", "[benchmark][onnx]") {
  automix::ai::OnnxModelInference inference;
  automix::ai::InferenceRequest request;
  request.task = "mix_parameters";
  request.features = {0.0, 1.0, 2.0, 3.0, 0.5};

  const auto result = inference.run(request);
  REQUIRE_FALSE(result.usedModel);
  REQUIRE_FALSE(result.logMessage.empty());
}

TEST_CASE("ONNX model load failure is graceful", "[benchmark][onnx]") {
  automix::ai::OnnxModelInference inference;
  const bool loaded = inference.loadModel("/nonexistent/path/model.onnx");
  REQUIRE_FALSE(loaded);
  REQUIRE_FALSE(inference.isAvailable());
}

#include <juce_core/juce_core.h>

TEST_CASE("Model benchmark on demo-mix-v1 stub exits with code 3", "[benchmark][model]") {
  const juce::File exe = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  const auto devTools = exe.getSiblingFile("automix_dev_tools"
#if defined(_WIN32)
                                          ".exe"
#endif
  );
  if (!devTools.existsAsFile()) {
    SKIP("automix_dev_tools executable not found in test directory");
  }

  juce::StringArray args;
  args.add(devTools.getFullPathName());
  args.add("model");
  args.add("bench");
  args.add("--pack");
  args.add("assets/models/demo-mix-v1");

  juce::ChildProcess process;
  REQUIRE(process.start(args));
  REQUIRE(process.waitForProcessToFinish(30000));
  REQUIRE(process.getExitCode() == 3);
}

