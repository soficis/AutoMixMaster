#include "ai/ModelPackLoader.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <fstream>

#include <nlohmann/json.hpp>

#include "ai/FeatureSchema.h"
#include "analysis/SpectrogramFrontEnd.h"
#include "util/HashUtils.h"
#include "util/Sha256.h"
#include "util/StringUtils.h"

namespace automix::ai {
namespace {

// Retained only to validate manifests written before the SHA-256 switch. Never
// used to produce a new checksum: FNV-1a has no collision resistance, so a
// tampered model could be made to match a forged manifest digest.
std::string computeLegacyChecksum(const std::filesystem::path& filePath) {
  std::ifstream in(filePath, std::ios::binary);
  if (!in.is_open()) {
    return "";
  }

  uint64_t hash = util::kFnv1a64OffsetBasis;
  char buffer[4096];
  while (in.good()) {
    in.read(buffer, static_cast<std::streamsize>(sizeof(buffer)));
    const auto readCount = static_cast<size_t>(in.gcount());
    if (readCount > 0) {
      hash = util::fnv1a64Update(hash, buffer, readCount);
    }
  }

  return util::toHex(hash);
}

std::vector<std::string> readStringArray(const nlohmann::json& json, const char* keyA, const char* keyB = nullptr) {
  if (json.contains(keyA) && json.at(keyA).is_array()) {
    return json.at(keyA).get<std::vector<std::string>>();
  }
  if (keyB != nullptr && json.contains(keyB) && json.at(keyB).is_array()) {
    return json.at(keyB).get<std::vector<std::string>>();
  }
  return {};
}

bool hasRequiredMetadata(const ModelPack& pack) {
  return !pack.licenseId.empty() && !pack.source.empty() && !pack.featureSchemaVersion.empty();
}

std::string normalizeTaskScope(std::string scope) {
  scope = util::toLower(util::trim(std::move(scope)));
  if (scope == "mix" || scope == "master" || scope == "analysis" || scope == "separation") {
    return scope;
  }
  if (scope == "stem-separation" || scope == "source-separation") {
    return "separation";
  }
  if (scope == "role" || scope == "classifier" || scope == "metadata") {
    return "analysis";
  }
  return "";
}

std::string inferTaskScopeFromType(const std::string& type) {
  const auto normalized = util::toLower(type);
  if (normalized == "mix_parameters" || normalized == "mix_model") {
    return "mix";
  }
  if (normalized == "master_parameters" || normalized == "master_model") {
    return "master";
  }
  if (normalized == "separation_model" || normalized == "source_separation" || normalized == "stem_separation") {
    return "separation";
  }
  if (normalized == "analysis_model" ||
      normalized == "role_classifier" ||
      normalized == "tag_classifier" ||
      normalized == "embedding_model" ||
      normalized == "analysis") {
    return "analysis";
  }
  return "";
}

// Always true: createInferenceBackend only ever builds OnnxModelInference, so a
// non-ONNX pack can never load and would silently no-op. Do NOT narrow this to
// mix/master again - that let a PyTorch `.pt` analysis pack install unnoticed.
bool requiresOnnxModelForScope(const std::string& scope) {
  static_cast<void>(scope);
  return true;
}

bool hasRequiredOutputKeysForScope(const std::string& scope, const std::vector<std::string>& keys) {
  if (scope != "mix" && scope != "master") {
    return !keys.empty();
  }

  const auto hasKey = [&](const char* key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
  };

  if (scope == "mix") {
    return hasKey("confidence") && hasKey("global_gain_db") && hasKey("global_pan_bias");
  }

  return hasKey("confidence") &&
         hasKey("target_lufs") &&
         hasKey("pre_gain_db") &&
         hasKey("limiter_ceiling_db") &&
         hasKey("glue_ratio");
}

template <typename T>
bool readRequired(const nlohmann::json& object, const char* key, T& out, const std::string& where, std::string& errorOut) {
  if (!object.contains(key)) {
    errorOut = where + " is missing required field '" + key + "'";
    return false;
  }
  try {
    out = object.at(key).get<T>();
  } catch (const nlohmann::json::exception&) {
    errorOut = where + " field '" + key + "' has the wrong type";
    return false;
  }
  return true;
}

bool parseIoList(const nlohmann::json& block,
                 const char* key,
                 std::vector<TensorContract::Io>& out,
                 std::string& errorOut) {
  const std::string where = std::string("tensor_contract.") + key;
  if (!block.contains(key) || !block.at(key).is_array()) {
    errorOut = where + " must be an array";
    return false;
  }
  for (const auto& entry : block.at(key)) {
    TensorContract::Io io;
    if (!entry.is_object()) {
      errorOut = where + " entries must be objects";
      return false;
    }
    if (entry.contains("name")) {
      if (!entry.at("name").is_string()) {
        errorOut = where + " name must be a string";
        return false;
      }
      io.name = entry.at("name").get<std::string>();
    }
    if (!readRequired(entry, "shape", io.shape, where, errorOut) ||
        !readRequired(entry, "dtype", io.dtype, where, errorOut)) {
      return false;
    }
    out.push_back(std::move(io));
  }
  return true;
}

// Can a graph whose probed dims are `probed` accept/produce `declared`? A
// dynamic probed axis accepts any declared extent.
bool graphAccepts(const std::vector<int64_t>& probed, const std::vector<int64_t>& declared) {
  if (probed.size() != declared.size()) {
    return false;
  }
  for (std::size_t axis = 0; axis < probed.size(); ++axis) {
    if (probed[axis] != -1 && declared[axis] != -1 && probed[axis] != declared[axis]) {
      return false;
    }
  }
  return true;
}

bool checkProbedSide(const char* side,
                     const std::vector<TensorContract::Io>& declared,
                     const std::vector<TensorSpec>& probed,
                     std::string& errorOut) {
  const bool namesDeclared = std::any_of(declared.begin(), declared.end(), [](const auto& io) { return !io.name.empty(); });
  if (namesDeclared) {
    std::set<std::string> declaredNames;
    std::set<std::string> probedNames;
    for (const auto& io : declared) {
      declaredNames.insert(io.name);
    }
    for (const auto& spec : probed) {
      probedNames.insert(spec.name);
    }
    for (const auto& name : declaredNames) {
      if (probedNames.count(name) == 0) {
        errorOut = std::string("declared ") + side + " '" + name + "' does not exist in the graph";
        return false;
      }
    }
    for (const auto& name : probedNames) {
      if (declaredNames.count(name) == 0) {
        errorOut = std::string("graph ") + side + " '" + name + "' is not declared in tensor_contract";
        return false;
      }
    }
  }
  if (declared.size() != probed.size()) {
    errorOut = std::string("tensor_contract declares ") + std::to_string(declared.size()) + " " + side +
               "(s) but the graph has " + std::to_string(probed.size());
    return false;
  }
  for (std::size_t i = 0; i < declared.size(); ++i) {
    const auto& io = declared[i];
    const auto match = namesDeclared ? std::find_if(probed.begin(), probed.end(),
                                                    [&](const TensorSpec& spec) { return spec.name == io.name; })
                                     : probed.begin() + static_cast<std::ptrdiff_t>(i);
    if (!graphAccepts(match->dims, io.shape)) {
      errorOut = std::string(side) + " '" + match->name + "' is " + describeShape(match->dims) +
                 " in the graph but declared " + describeShape(io.shape);
      return false;
    }
  }
  return true;
}

} // namespace

std::optional<TensorContract> parseTensorContract(const nlohmann::json& block, std::string& errorOut) {
  if (!block.is_object()) {
    errorOut = "tensor_contract must be an object";
    return std::nullopt;
  }
  TensorContract contract;
  const std::string where = "tensor_contract";
  if (!readRequired(block, "engine", contract.engine, where, errorOut) ||
      !readRequired(block, "sample_rate", contract.sampleRate, where, errorOut) ||
      !readRequired(block, "stereo", contract.stereo, where, errorOut) ||
      !readRequired(block, "chunk_samples", contract.chunkSamples, where, errorOut) ||
      !readRequired(block, "overlap_samples", contract.overlapSamples, where, errorOut) ||
      !readRequired(block, "input_layout", contract.inputLayout, where, errorOut) ||
      !readRequired(block, "output_mode", contract.outputMode, where, errorOut)) {
    return std::nullopt;
  }
  if (!block.contains("stft") || !block.at("stft").is_object()) {
    errorOut = "tensor_contract.stft must be an object";
    return std::nullopt;
  }
  const auto& stft = block.at("stft");
  const std::string stftWhere = "tensor_contract.stft";
  if (!readRequired(stft, "n_fft", contract.stft.nFft, stftWhere, errorOut) ||
      !readRequired(stft, "hop_length", contract.stft.hopLength, stftWhere, errorOut) ||
      !readRequired(stft, "win_length", contract.stft.winLength, stftWhere, errorOut) ||
      !readRequired(stft, "window", contract.stft.window, stftWhere, errorOut) ||
      !readRequired(stft, "periodic_window", contract.stft.periodicWindow, stftWhere, errorOut) ||
      !readRequired(stft, "center", contract.stft.center, stftWhere, errorOut) ||
      !readRequired(stft, "pad_mode", contract.stft.padMode, stftWhere, errorOut) ||
      !readRequired(stft, "normalized", contract.stft.normalized, stftWhere, errorOut) ||
      !readRequired(stft, "zero_dc", contract.stft.zeroDc, stftWhere, errorOut)) {
    return std::nullopt;
  }
  if (!parseIoList(block, "inputs", contract.inputs, errorOut) ||
      !parseIoList(block, "outputs", contract.outputs, errorOut)) {
    return std::nullopt;
  }
  if (block.contains("target_stem")) {
    if (!readRequired(block, "target_stem", contract.targetStem, where, errorOut)) {
      return std::nullopt;
    }
  }
  if (!block.contains("stems") || !block.at("stems").is_array()) {
    errorOut = "tensor_contract.stems must be an array";
    return std::nullopt;
  }
  for (const auto& entry : block.at("stems")) {
    TensorContract::Stem stem;
    if (!entry.is_object() || !readRequired(entry, "name", stem.name, "tensor_contract.stems", errorOut)) {
      if (errorOut.empty()) {
        errorOut = "tensor_contract.stems entries must be objects";
      }
      return std::nullopt;
    }
    if (entry.contains("residual_of") &&
        !readRequired(entry, "residual_of", stem.residualOf, "tensor_contract.stems", errorOut)) {
      return std::nullopt;
    }
    contract.stems.push_back(std::move(stem));
  }
  return contract;
}

nlohmann::json tensorContractToJson(const TensorContract& contract) {
  const auto ioList = [](const std::vector<TensorContract::Io>& list) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& io : list) {
      nlohmann::json entry = {{"shape", io.shape}, {"dtype", io.dtype}};
      if (!io.name.empty()) {
        entry["name"] = io.name;
      }
      out.push_back(entry);
    }
    return out;
  };
  nlohmann::json stems = nlohmann::json::array();
  for (const auto& stem : contract.stems) {
    nlohmann::json entry = {{"name", stem.name}};
    if (!stem.residualOf.empty()) {
      entry["residual_of"] = stem.residualOf;
    }
    stems.push_back(entry);
  }
  nlohmann::json block = {
      {"engine", contract.engine},
      {"sample_rate", contract.sampleRate},
      {"stereo", contract.stereo},
      {"chunk_samples", contract.chunkSamples},
      {"overlap_samples", contract.overlapSamples},
      {"stft",
       {{"n_fft", contract.stft.nFft},
        {"hop_length", contract.stft.hopLength},
        {"win_length", contract.stft.winLength},
        {"window", contract.stft.window},
        {"periodic_window", contract.stft.periodicWindow},
        {"center", contract.stft.center},
        {"pad_mode", contract.stft.padMode},
        {"normalized", contract.stft.normalized},
        {"zero_dc", contract.stft.zeroDc}}},
      {"input_layout", contract.inputLayout},
      {"inputs", ioList(contract.inputs)},
      {"outputs", ioList(contract.outputs)},
      {"output_mode", contract.outputMode},
      {"stems", stems},
  };
  if (!contract.targetStem.empty()) {
    block["target_stem"] = contract.targetStem;
  }
  return block;
}

std::optional<RunnerConfig> runnerConfigFromContract(const TensorContract& contract, std::string& errorOut) {
  const auto layout = inputLayoutFromString(contract.inputLayout);
  if (!layout.has_value()) {
    errorOut = "tensor_contract input_layout '" + contract.inputLayout +
               "' is not one of folded_stereo, split_channels";
    return std::nullopt;
  }
  const auto mode = outputModeFromString(contract.outputMode);
  if (!mode.has_value()) {
    errorOut = "tensor_contract output_mode '" + contract.outputMode + "' is not one of direct, mask";
    return std::nullopt;
  }
  if (contract.stft.padMode != "reflect") {
    errorOut = "tensor_contract.stft pad_mode '" + contract.stft.padMode + "' is not implemented (only reflect)";
    return std::nullopt;
  }
  if (contract.stft.window != "hann" || !contract.stft.periodicWindow) {
    errorOut = "tensor_contract.stft window '" + contract.stft.window +
               (contract.stft.periodicWindow ? "" : " (symmetric)") + "' is not implemented (only periodic hann)";
    return std::nullopt;
  }
  if (contract.sampleRate <= 0) {
    errorOut = "tensor_contract sample_rate must be positive";
    return std::nullopt;
  }

  RunnerConfig config;
  config.chunkSamples = contract.chunkSamples;
  config.overlapSamples = contract.overlapSamples;
  config.sampleRate = static_cast<double>(contract.sampleRate);
  config.channels = contract.stereo ? 2 : 1;
  config.stft.nFft = contract.stft.nFft;
  config.stft.hopLength = contract.stft.hopLength;
  config.stft.winLength = contract.stft.winLength;
  config.stft.center = contract.stft.center;
  config.stft.normalized = contract.stft.normalized;
  config.stft.zeroDc = contract.stft.zeroDc;
  config.inputLayout = *layout;
  config.outputMode = *mode;
  config.targetStem = contract.targetStem;
  config.stems.clear();
  for (const auto& stem : contract.stems) {
    config.stems.push_back({stem.name, stem.residualOf});
  }
  if (const auto error = validateRunnerConfig(config); !error.empty()) {
    errorOut = "tensor_contract: " + error;
    return std::nullopt;
  }
  if (config.stft.nFft <= 0 || (config.stft.nFft & (config.stft.nFft - 1)) != 0 || config.stft.hopLength <= 0 ||
      config.stft.winLength <= 0 || config.stft.winLength > config.stft.nFft) {
    errorOut = "tensor_contract.stft n_fft " + std::to_string(config.stft.nFft) + " / hop_length " +
               std::to_string(config.stft.hopLength) + " / win_length " + std::to_string(config.stft.winLength) +
               " is not a supported STFT";
    return std::nullopt;
  }
  return config;
}

bool checkTensorContract(const TensorContract& contract,
                         const std::vector<TensorSpec>& probedInputs,
                         const std::vector<TensorSpec>& probedOutputs,
                         std::string& errorOut) {
  const auto config = runnerConfigFromContract(contract, errorOut);
  if (!config.has_value()) {
    return false;
  }
  if (contract.inputs.size() != 1 || contract.outputs.size() != 1) {
    errorOut = "tensor_contract must declare exactly one input and one output, got " +
               std::to_string(contract.inputs.size()) + " and " + std::to_string(contract.outputs.size());
    return false;
  }

  const int freqBins = config->stft.nFft / 2 + 1;
  const int frames = analysis::stftFrameCount(config->chunkSamples, config->stft);
  const int graphStems = static_cast<int>(std::count_if(contract.stems.begin(), contract.stems.end(),
                                                        [](const auto& stem) { return stem.residualOf.empty(); }));
  const int stemAxis = config->outputMode == OutputMode::Mask ? 1 : graphStems;
  const auto impliedInput = tensorInputDims(config->inputLayout, config->channels, freqBins, frames);
  const auto impliedOutput = tensorOutputDims(config->inputLayout, stemAxis, config->channels, freqBins, frames);
  const auto stftSummary = "n_fft " + std::to_string(config->stft.nFft) + " (" + std::to_string(freqBins) +
                           " bins), " + std::to_string(frames) + " frames per " +
                           std::to_string(config->chunkSamples) + "-sample chunk, " +
                           std::to_string(config->channels) + " channel(s), " + contract.inputLayout;

  const auto label = [](const TensorContract::Io& io, const char* fallback) {
    return io.name.empty() ? std::string(fallback) : io.name;
  };
  const auto& input = contract.inputs.front();
  const auto& output = contract.outputs.front();
  for (const auto* io : {&input, &output}) {
    if (io->dtype != "float32") {
      errorOut = "tensor '" + label(*io, io == &input ? "input" : "output") + "' declares dtype '" + io->dtype +
                 "'; only float32 is supported";
      return false;
    }
  }
  const TensorSpec declaredInput{input.name, TensorElementType::Float32, input.shape};
  if (!shapesMatch(declaredInput, TensorSpec{input.name, TensorElementType::Float32, impliedInput})) {
    errorOut = "input '" + label(input, "input") + "' declares " + describeShape(input.shape) + " but " + stftSummary +
               " implies " + describeShape(impliedInput);
    return false;
  }
  const TensorSpec declaredOutput{output.name, TensorElementType::Float32, output.shape};
  if (!shapesMatch(declaredOutput, TensorSpec{output.name, TensorElementType::Float32, impliedOutput})) {
    errorOut = "output '" + label(output, "output") + "' declares " + describeShape(output.shape) + " but " +
               stftSummary + " implies " + describeShape(impliedOutput);
    return false;
  }

  return checkProbedSide("input", contract.inputs, probedInputs, errorOut) &&
         checkProbedSide("output", contract.outputs, probedOutputs, errorOut);
}

std::optional<ModelPack> ModelPackLoader::load(const std::filesystem::path& directory) const {
  const auto metadataPath = directory / "model.json";
  if (!std::filesystem::exists(metadataPath)) {
    return std::nullopt;
  }

  std::ifstream in(metadataPath);
  nlohmann::json json;
  in >> json;

  ModelPack pack;
  pack.schemaVersion = json.value("schemaVersion", json.value("schema_version", 1));
  pack.id = json.value("id", directory.filename().string());
  pack.name = json.value("name", pack.id);
  pack.type = json.value("type", "unknown");
  pack.taskScope = normalizeTaskScope(json.value("task_scope", json.value("taskScope", "")));
  if (pack.taskScope.empty()) {
    pack.taskScope = inferTaskScopeFromType(pack.type);
  }
  pack.engine = json.value("engine", "unknown");
  pack.minAppVersion = json.value("min_app_version", json.value("minAppVersion", "0.0.0"));
  pack.version = json.value("version", "0.0.0");
  pack.licenseId = json.value("license", json.value("licenseId", ""));
  pack.source = json.value("source", "");
  pack.intendedUse = json.value("intended_use", json.value("intendedUse", ""));
  pack.featureSchemaVersion = json.value("feature_schema_version", json.value("featureSchemaVersion", ""));
  pack.modelFile = json.value("modelFile", json.value("model_file", "model.onnx"));
  pack.auxiliaryFiles = readStringArray(json, "auxiliaryFiles", "auxiliary_files");
  pack.checksum = json.value("checksum", "");
  if (json.contains("inputFeatureCount")) {
    pack.inputFeatureCount = json.at("inputFeatureCount").get<size_t>();
  } else if (json.contains("input_feature_count")) {
    pack.inputFeatureCount = json.at("input_feature_count").get<size_t>();
  } else {
    pack.inputFeatureCount.reset();
  }

  pack.expectedOutputKeys = readStringArray(json, "expectedOutputKeys", "output_keys");
  pack.inputNames = readStringArray(json, "inputNames", "input_names");
  pack.outputNames = readStringArray(json, "outputNames", "output_names");

  if (pack.expectedOutputKeys.empty() && json.contains("output_schema") && json.at("output_schema").is_object()) {
    for (const auto& entry : json.at("output_schema").items()) {
      pack.expectedOutputKeys.push_back(entry.key());
    }
  }

  pack.preferredPrecision = json.value("preferredPrecision", json.value("preferred_precision", "auto"));
  pack.providerAffinity = readStringArray(json, "providerAffinity", "provider_affinity");
  if (json.contains("defaultIntraOpThreads")) {
    pack.defaultIntraOpThreads = json.at("defaultIntraOpThreads").get<int>();
  } else if (json.contains("default_intra_op_threads")) {
    pack.defaultIntraOpThreads = json.at("default_intra_op_threads").get<int>();
  } else {
    pack.defaultIntraOpThreads.reset();
  }
  if (json.contains("defaultInterOpThreads")) {
    pack.defaultInterOpThreads = json.at("defaultInterOpThreads").get<int>();
  } else if (json.contains("default_inter_op_threads")) {
    pack.defaultInterOpThreads = json.at("default_inter_op_threads").get<int>();
  } else {
    pack.defaultInterOpThreads.reset();
  }
  pack.enableProfiling = json.value("enableProfiling", json.value("enable_profiling", false));

  if (pack.featureSchemaVersion.empty() && json.contains("feature_schema") && json.at("feature_schema").is_object()) {
    pack.featureSchemaVersion = json.at("feature_schema").value("version", "");
  }

  if (json.contains("tensor_contract")) {
    std::string contractError;
    auto contract = parseTensorContract(json.at("tensor_contract"), contractError);
    if (!contract.has_value()) {
      return std::nullopt;
    }
    pack.tensorContract = std::move(contract);
  }

  pack.rootPath = directory;

  if (!hasRequiredMetadata(pack)) {
    return std::nullopt;
  }
  if (pack.taskScope.empty()) {
    return std::nullopt;
  }
  const auto inferredScope = inferTaskScopeFromType(pack.type);
  if (!inferredScope.empty() && inferredScope != pack.taskScope) {
    return std::nullopt;
  }
  if (!FeatureSchemaV1::isCompatible(pack.featureSchemaVersion)) {
    return std::nullopt;
  }
  if (pack.inputFeatureCount.has_value() && pack.inputFeatureCount.value() == 0) {
    return std::nullopt;
  }

  const auto modelPath = directory / pack.modelFile;
  if (!std::filesystem::exists(modelPath)) {
    return std::nullopt;
  }
  if (requiresOnnxModelForScope(pack.taskScope) &&
      util::toLower(modelPath.extension().string()) != ".onnx") {
    return std::nullopt;
  }
  for (const auto& auxiliaryFile : pack.auxiliaryFiles) {
    if (auxiliaryFile.empty() || !std::filesystem::exists(directory / auxiliaryFile)) {
      return std::nullopt;
    }
  }

  // SHA-256 is the pack trust anchor. Manifests written before this change carry a
  // 16-hex FNV-1a-64 digest instead, so that legacy form stays accepted for
  // back-compat; a checksum that is neither is a corrupt or hand-edited manifest
  // and the pack is rejected rather than trusted.
  const std::string computedChecksum = computeChecksum(modelPath);
  if (computedChecksum.empty()) {
    return std::nullopt;
  }

  if (pack.checksum.empty()) {
    pack.checksum = computedChecksum;
  } else if (util::isSha256Hex(pack.checksum)) {
    if (pack.checksum != computedChecksum) {
      return std::nullopt;
    }
  } else if (pack.checksum != computeLegacyChecksum(modelPath)) {
    return std::nullopt;
  }
  // A tensor pack's outputs are the named tensors of its contract, not scalar
  // keys, so the scalar output-key requirement does not apply to it.
  if (!pack.tensorContract.has_value() &&
      !hasRequiredOutputKeysForScope(pack.taskScope, pack.expectedOutputKeys)) {
    return std::nullopt;
  }

  return pack;
}

std::string ModelPackLoader::computeChecksum(const std::filesystem::path& filePath) const {
  return util::fileSha256(filePath);
}

} // namespace automix::ai
