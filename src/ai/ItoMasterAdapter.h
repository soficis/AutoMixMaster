#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "ai/InferenceTypes.h"
#include "ai/OnnxModelInference.h"
#include "dsp/ItoMasterFxChain.h"
#include "engine/AudioBuffer.h"

namespace automix::ai {

// ─────────────────────────────────────────────────────────────────────────────
// ITO-Master mastering-adapter contract.
//
// Parses the ITO-Master pack's config.json (the 46-param schema) into a typed
// contract, denormalizes the model's normalized parameter vector onto the
// white-box FX chain settings (dsp::ItoMasterFxChain), and runs both ONNX
// artifacts (fxencoder.onnx, mastering_tcn.onnx) through OnnxModelInference.
//
// Native ONNX Runtime may be absent from a build (AUTOMIX_HAS_NATIVE_ORT=0,
// deterministic fallback). In that case tensor-shape validation is performed
// against the STATIC contract parsed from config.json rather than an introspected
// session, so shape integrity is still asserted without a false "pass" on the
// deterministic fallback (which never emits 2048-dim embeddings or 46 params).
// ─────────────────────────────────────────────────────────────────────────────

inline constexpr const char* kItoMasterRepoId = "kramp/ito-master-onnx";
inline constexpr const char* kItoMasterModelId = "huggingface:kramp/ito-master-onnx";
inline constexpr const char* kItoMasterEncoderFile = "fxencoder.onnx";
inline constexpr const char* kItoMasterPredictorFile = "mastering_tcn.onnx";
inline constexpr const char* kItoMasterConfigFile = "config.json";
inline constexpr const char* kItoMasterLicense = "CC BY-NC 4.0";
inline constexpr const char* kItoMasterAttribution =
    "ITO-Master, Koo et al., Sony Research (github.com/SonyResearch/ITO-Master); "
    "ONNX export by kramp (huggingface.co/kramp/ito-master-onnx)";

inline constexpr int kItoMasterEmbeddingSize = 2048;
inline constexpr int kItoMasterParamCount = 46;
inline constexpr int kItoMasterDefaultChunkSamples = 44100 * 10; // 10 s reference window

struct ItoParamSpec {
  std::string fx;
  std::string name;
  double min = 0.0;
  double max = 1.0;
};

// A tensor dimension; -1 represents the dynamic length "N" from config.json.
struct ItoTensorShape {
  std::vector<int64_t> dims;
};

struct ItoGraphContract {
  std::string file;
  std::vector<ItoTensorShape> inputShapes;
  std::vector<ItoTensorShape> outputShapes;
};

struct ItoMasterConfig {
  std::string model;
  double sampleRate = 44100.0;
  std::vector<std::string> fxOrder;
  int numParams = 0;
  bool normalized = true;
  std::vector<ItoParamSpec> params;
  ItoGraphContract encoder;    // fxencoder.onnx
  ItoGraphContract predictor;  // mastering_tcn.onnx
};

class ItoMasterAdapter {
 public:
  // Parse config.json into the typed 46-param contract (nullopt on any schema
  // violation). This is the authoritative static shape source when native ORT
  // is absent.
  static std::optional<ItoMasterConfig> loadConfig(const std::filesystem::path& configPath);

  // Structural checks: 46 params, sane ranges, expected graph files/shapes.
  static bool validateConfig(const ItoMasterConfig& config);

  // Static tensor-shape contract validation, independent of any ONNX session:
  //   fxencoder: audio[1,2,N]  -> embedding[1,2048]
  //   mastering_tcn: audio[1,2,N] + embedding[1,2048] -> params[1,46]
  static bool validateTensorContract(const ItoMasterConfig& config, int numChannels, int numSamples);

  // value = norm * (max - min) + min, with norm clamped to [0,1] first.
  static std::vector<double> denormalize(const ItoMasterConfig& config,
                                         const std::vector<double>& normalized);

  // Inverse mapping for round-trip testing: norm = (value - min) / (max - min).
  static std::vector<double> normalize(const ItoMasterConfig& config,
                                       const std::vector<double>& physical);

  // Map 46 physical parameter values onto the white-box chain settings.
  // Out-of-range values are clamped to their config bounds.
  static dsp::ItoMasterChainSettings toChainSettings(const ItoMasterConfig& config,
                                                     const std::vector<double>& physical);

  // Convenience contract: denormalize + map straight onto the chain settings.
  static dsp::ItoMasterChainSettings apply(const ItoMasterConfig& config,
                                           const std::vector<double>& normalized);

  // Extract a fixed-size parameter/embedding vector from flattened model
  // outputs (output_0..output_N keys). Returns nullopt when the expected count
  // of values is not present (e.g. deterministic fallback output).
  static std::optional<std::vector<double>> extractParams(const std::vector<double>& flattenedOutputs);
  static std::optional<std::vector<double>> extractEmbedding(const std::vector<double>& flattenedOutputs);
};

// Runs both ONNX artifacts through OnnxModelInference (in-process, sharing the
// execution-provider chain + recovery). Produces the 46 normalized params only
// when the native path actually yields them; otherwise returns nullopt.
class ItoMasterModelRunner {
 public:
  // Loads fxencoder.onnx + mastering_tcn.onnx from the pack directory.
  bool load(const std::filesystem::path& packDir, const ItoMasterConfig& config, std::string* errorOut = nullptr);

  bool loaded() const { return loaded_; }
  bool usesNativeSession() const;

  // Reference audio -> 46 normalized params in [0,1].
  std::optional<std::vector<double>> predict(const engine::AudioBuffer& referenceAudio) const;

 private:
  std::filesystem::path packDir_;
  ItoMasterConfig config_;
  OnnxModelInference encoder_;
  OnnxModelInference predictor_;
  bool loaded_ = false;
};

} // namespace automix::ai
