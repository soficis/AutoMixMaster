#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef ENABLE_ONNX
#include "ai/FeatureSchema.h"
#include "ai/OnnxModelInference.h"
#else
#include "ai/IModelInference.h"
#endif

TEST_CASE("ONNX inference backend validates model load and schema", "[ai][onnx]") {
#ifndef ENABLE_ONNX
  SUCCEED("ENABLE_ONNX is off; ONNX backend tests skipped.");
#else
  automix::ai::OnnxModelInference inference;
  REQUIRE_FALSE(inference.loadModel("missing_model.onnx"));

  const auto tempDir = std::filesystem::temp_directory_path() / "automix_onnx_inference_test";
  std::filesystem::remove_all(tempDir);
  std::filesystem::create_directories(tempDir);
  const auto modelPath = tempDir / "model.onnx";

  {
    std::ofstream model(modelPath, std::ios::binary);
    model << "dummy_onnx";
  }
  {
    std::ofstream meta(modelPath.string() + ".meta.json");
    meta << "{\"input_feature_count\":" << automix::ai::FeatureSchemaV1::featureCount()
         << ",\"allowed_tasks\":[\"master_parameters\",\"mix_parameters\"]}";
  }

  REQUIRE(inference.loadModel(modelPath));
  REQUIRE(inference.isAvailable());

  automix::ai::InferenceRequest mismatch{
      .task = "master_parameters",
      .features = {1.0, 2.0},
  };
  const auto mismatchResult = inference.run(mismatch);
  REQUIRE_FALSE(mismatchResult.usedModel);
  REQUIRE(mismatchResult.logMessage.find("mismatch") != std::string::npos);

  automix::ai::InferenceRequest valid{
      .task = "master_parameters",
      .features = std::vector<double>(automix::ai::FeatureSchemaV1::featureCount(), 0.25),
  };
  const auto resultA = inference.run(valid);
  const auto resultB = inference.run(valid);
  REQUIRE(resultA.usedModel);
  REQUIRE(resultB.usedModel);
  REQUIRE(resultA.outputs == resultB.outputs);
  REQUIRE(inference.backendDiagnostics().find("calls=2") != std::string::npos);

  std::filesystem::remove_all(tempDir);
#endif
}
