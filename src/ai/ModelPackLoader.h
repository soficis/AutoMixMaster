#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "ai/SeparationRunner.h"
#include "ai/TensorTypes.h"

namespace automix::ai {

// Declared audio-tensor interface of a pack (spec section 6). Strings are kept
// verbatim from the manifest so that checkTensorContract() can name an
// unrecognised value instead of the parser silently mapping it to a default.
struct TensorContract {
  struct Stft {
    int nFft = 0;
    int hopLength = 0;
    int winLength = 0;
    std::string window;
    bool periodicWindow = true;
    bool center = true;
    std::string padMode;
    bool normalized = false;
    bool zeroDc = true;
  };
  struct Io {
    std::string name;  // empty in the catalog form; the install-time probe fills it
    std::vector<int64_t> shape;
    std::string dtype;
  };
  struct Stem {
    std::string name;
    std::string residualOf;
  };

  std::string engine;
  int sampleRate = 0;
  bool stereo = true;
  int chunkSamples = 0;
  int overlapSamples = 0;
  Stft stft;
  std::string inputLayout;
  std::vector<Io> inputs;
  std::vector<Io> outputs;
  std::string outputMode;
  std::string targetStem;
  std::vector<Stem> stems;
};

// Structural parse of a `tensor_contract` block. Fails (nullopt + reason) only
// on missing or mistyped fields; value checks belong to checkTensorContract().
std::optional<TensorContract> parseTensorContract(const nlohmann::json& block, std::string& errorOut);
nlohmann::json tensorContractToJson(const TensorContract& contract);

// The spec 5.3 load-time cross-check: the declared contract must be internally
// consistent (STFT geometry vs declared shapes, layout, mode, stems) and must
// agree with the graph's probed inputs/outputs. Every failure names the tensor
// or field and prints both shapes where shapes are involved.
bool checkTensorContract(const TensorContract& contract,
                         const std::vector<TensorSpec>& probedInputs,
                         const std::vector<TensorSpec>& probedOutputs,
                         std::string& errorOut);

// The runner configuration a (checked) contract describes.
std::optional<RunnerConfig> runnerConfigFromContract(const TensorContract& contract, std::string& errorOut);

struct ModelPack {
  int schemaVersion = 1;
  std::string id;
  std::string name;
  std::string type;
  std::string taskScope;
  std::string engine;
  std::string minAppVersion;
  std::string version;
  std::string licenseId;
  std::string source;
  std::string intendedUse;
  std::string featureSchemaVersion;
  std::string modelFile;
  // Additional artifacts carried by the pack alongside the primary model file,
  // consumed as one model pack (e.g. ITO-Master: fxencoder.onnx primary +
  // mastering_tcn.onnx + config.json).
  std::vector<std::string> auxiliaryFiles;
  std::string checksum;
  std::optional<size_t> inputFeatureCount;
  std::string preferredPrecision;
  std::vector<std::string> providerAffinity;
  std::optional<int> defaultIntraOpThreads;
  std::optional<int> defaultInterOpThreads;
  bool enableProfiling = false;
  std::vector<std::string> expectedOutputKeys;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  // Absent for scalar-only packs; existing packs load unchanged.
  std::optional<TensorContract> tensorContract;
  // Device memory the model needs while running on a GPU (MiB). When free
  // memory is below it, GPU execution would spill into system memory, so
  // callers run on CPU instead. Absent: no known requirement.
  std::optional<std::uint64_t> gpuMemoryMb;
  // Per-pack GPU provider allow-list (e.g. {"cuda"}). When non-empty, GPU
  // candidates outside this list are ignored and fallback to CPU. Empty means
  // any runtime-supported GPU provider in the priority chain is allowed.
  std::vector<std::string> gpuProviders;
  std::filesystem::path rootPath;
};

class ModelPackLoader {
 public:
  std::optional<ModelPack> load(const std::filesystem::path& directory) const;

 private:
  std::string computeChecksum(const std::filesystem::path& filePath) const;
};

} // namespace automix::ai
