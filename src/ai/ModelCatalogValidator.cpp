#include "ai/ModelCatalogValidator.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "ai/ItoMasterAdapter.h"
#include "util/StringUtils.h"

namespace automix::ai {
namespace {

using ::automix::util::toLower;

bool containsToken(const std::string& haystack, const std::string& token) {
  return haystack.find(token) != std::string::npos;
}

std::string extensionLower(const std::string& fileName) {
  const auto ext = std::filesystem::path(fileName).extension().string();
  return toLower(ext);
}

bool isKnownTaskScope(const std::string& taskScope) {
  return taskScope == "mix" || taskScope == "master" || taskScope == "analysis" || taskScope == "separation";
}

std::string inferTaskScopeFromJoined(const std::string& joinedLower) {
  if (containsToken(joinedLower, "master") || containsToken(joinedLower, "loudness")) {
    return "master";
  }
  if (containsToken(joinedLower, "mix") || containsToken(joinedLower, "stereo-balance")) {
    return "mix";
  }
  if (containsToken(joinedLower, "demucs") ||
      containsToken(joinedLower, "htdemucs") ||
      containsToken(joinedLower, "mdx") ||
      containsToken(joinedLower, "roformer") ||
      containsToken(joinedLower, "unmix") ||
      containsToken(joinedLower, "source-separation") ||
      containsToken(joinedLower, "separator") ||
      containsToken(joinedLower, "denois") ||
      containsToken(joinedLower, "dereverb")) {
    return "separation";
  }
  if (containsToken(joinedLower, "clap") ||
      containsToken(joinedLower, "panns") ||
      containsToken(joinedLower, "tagging") ||
      containsToken(joinedLower, "embedding") ||
      containsToken(joinedLower, "pitch") ||
      containsToken(joinedLower, "midi") ||
      containsToken(joinedLower, "classif")) {
    return "analysis";
  }
  return "analysis";
}

std::string packTypeForScope(const std::string& taskScope) {
  if (taskScope == "mix") {
    return "mix_parameters";
  }
  if (taskScope == "master") {
    return "master_parameters";
  }
  if (taskScope == "separation") {
    return "separation_model";
  }
  return "analysis_model";
}

std::vector<std::string> outputKeysForScope(const std::string& taskScope) {
  if (taskScope == "mix") {
    return {"confidence", "global_gain_db", "global_pan_bias"};
  }
  if (taskScope == "master") {
    return {"confidence", "target_lufs", "pre_gain_db", "limiter_ceiling_db", "glue_ratio"};
  }
  if (taskScope == "separation") {
    return {"confidence", "separation_quality"};
  }
  return {"confidence"};
}

std::string sanitizePackId(std::string value) {
  if (value.empty()) {
    return "model-pack";
  }

  for (char& c : value) {
    const bool alphaNum = std::isalnum(static_cast<unsigned char>(c)) != 0;
    if (!alphaNum && c != '-' && c != '_') {
      c = '-';
    }
  }

  while (!value.empty() && (value.front() == '-' || value.front() == '_')) {
    value.erase(value.begin());
  }
  while (!value.empty() && (value.back() == '-' || value.back() == '_')) {
    value.pop_back();
  }

  if (value.empty()) {
    return "model-pack";
  }

  if (value.size() > 80) {
    value.resize(80);
    while (!value.empty() && (value.back() == '-' || value.back() == '_')) {
      value.pop_back();
    }
  }

  return value.empty() ? "model-pack" : value;
}

} // namespace

std::string inferTaskScope(const HubModelInfo& model) {
  if (isKnownTaskScope(model.taskScope)) {
    return model.taskScope;
  }

  std::string joined = toLower(model.useCase);
  joined += "|" + toLower(model.repoId);
  joined += "|" + toLower(model.displayName);
  for (const auto& tag : model.tags) {
    joined += "|" + toLower(tag);
  }
  return inferTaskScopeFromJoined(joined);
}

ModelCompatibilityResult validateCatalogModel(const HubModelInfo& model) {
  ModelCompatibilityResult result;
  result.taskScope = inferTaskScope(model);
  result.packType = packTypeForScope(result.taskScope);
  result.expectedOutputKeys = outputKeysForScope(result.taskScope);

  if (model.primaryFile.empty()) {
    result.reason = "No primary model file was detected";
    return result;
  }

  const auto ext = extensionLower(model.primaryFile);
  static const std::unordered_set<std::string> supportedExt = {
      ".onnx", ".safetensors", ".bin", ".pt", ".pth", ".th", ".ckpt"};
  if (supportedExt.find(ext) == supportedExt.end()) {
    result.reason = "Unsupported primary model file extension: " + ext;
    return result;
  }

  result.engine = ext == ".onnx" ? "onnxruntime" : "unknown";

  if ((result.taskScope == "mix" || result.taskScope == "master") && ext != ".onnx") {
    result.reason = "Mix/Master task packs must provide an ONNX model for deterministic runtime support";
    return result;
  }

  result.compatible = true;
  result.reason = "compatible";
  return result;
}

std::string normalizeModelIdForPack(const std::string& modelId) {
  return sanitizePackId(toLower(modelId));
}

bool writeTurnkeyModelPackManifest(const std::filesystem::path& installPath,
                                   const HubModelInfo& model,
                                   const HubInstallResult& installResult,
                                   const ModelCompatibilityResult& compatibility,
                                   std::string* errorOut) {
  if (!compatibility.compatible) {
    if (errorOut != nullptr) {
      *errorOut = compatibility.reason;
    }
    return false;
  }

  std::error_code error;
  std::filesystem::create_directories(installPath, error);
  if (error) {
    if (errorOut != nullptr) {
      *errorOut = "Failed to create model pack directory: " + installPath.string();
    }
    return false;
  }

  const auto modelFileName = installResult.primaryFilePath.filename().string();
  if (modelFileName.empty()) {
    if (errorOut != nullptr) {
      *errorOut = "Model pack install missing primary model file name";
    }
    return false;
  }

  const auto packIdSource = !model.modelId.empty() ? model.modelId : model.repoId;
  const auto packId = normalizeModelIdForPack(packIdSource);

  nlohmann::json outputSchema = nlohmann::json::object();
  for (const auto& key : compatibility.expectedOutputKeys) {
    outputSchema[key] = "float";
  }

  nlohmann::json manifest = {
      {"schema_version", 1},
      {"id", packId},
      {"name", model.displayName.empty() ? model.repoId : model.displayName},
      {"type", compatibility.packType},
      {"task_scope", compatibility.taskScope},
      {"engine", compatibility.engine.empty() ? "unknown" : compatibility.engine},
      {"version", installResult.revision.empty() ? "0.0.0" : installResult.revision},
      {"model_file", modelFileName},
      {"auxiliary_files", installResult.auxiliaryFiles},
      {"license", model.license.empty() ? "unknown" : model.license},
      {"source", model.sourceUrl.empty() ? model.source : model.sourceUrl},
      {"feature_schema_version", "1.0.0"},
      {"output_schema", outputSchema},
      {"input_names", nlohmann::json::array()},
      {"output_names", nlohmann::json::array()},
  };

  if (model.repoId == kItoMasterRepoId) {
    manifest["intended_use"] =
        std::string("Experimental ITO-Master AI mastering route (non-default, off by default). ") +
        "License: " + kItoMasterLicense + ". Attribution: " + kItoMasterAttribution;
  }

  const auto manifestPath = installPath / "model.json";
  std::ofstream out(manifestPath);
  if (!out.is_open()) {
    if (errorOut != nullptr) {
      *errorOut = "Failed to open model manifest for writing: " + manifestPath.string();
    }
    return false;
  }
  out << manifest.dump(2);
  return true;
}

} // namespace automix::ai
