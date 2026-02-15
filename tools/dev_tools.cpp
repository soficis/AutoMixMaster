#include <array>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "ai/IModelInference.h"
#include "ai/FeatureSchema.h"
#include "ai/AutoMasterStrategyAI.h"
#include "ai/AutoMixStrategyAI.h"
#include "ai/ModelPackLoader.h"
#include "ai/OnnxModelInference.h"
#include "ai/RtNeuralInference.h"
#include "analysis/StemHealthAssistant.h"
#include "analysis/StemAnalyzer.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automix/HeuristicAutoMixStrategy.h"
#include "domain/MasterPlan.h"
#include "domain/ProjectProfile.h"
#include "domain/Session.h"
#include "domain/JsonSerialization.h"
#include "domain/Stem.h"
#include "domain/StemOrigin.h"
#include "domain/StemRole.h"
#include "engine/AudioFileIO.h"
#include "engine/BatchQueueRunner.h"
#include "engine/OfflineRenderPipeline.h"
#include "engine/SessionRepository.h"
#include "renderers/BuiltInRenderer.h"
#include "renderers/RendererFactory.h"
#include "renderers/RendererRegistry.h"
#include "util/LameDownloader.h"
#include "util/WavWriter.h"
#include "renderers/ExternalLimiterRenderer.h"
#include "RegressionHarness.h"

namespace {

std::string sanitizeFileName(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    if (std::isalnum(c) || c == '_' || c == '-') {
      return static_cast<char>(c);
    }
    return '_';
  });
  if (value.empty()) {
    return "segment";
  }
  return value;
}

std::optional<std::string> argValue(const std::vector<std::string>& args, const std::string& key) {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      return args[i + 1];
    }
  }
  return std::nullopt;
}

bool hasFlag(const std::vector<std::string>& args, const std::string& key) {
  return std::find(args.begin(), args.end(), key) != args.end();
}

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::vector<std::string> splitCommaSeparated(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      items.push_back(token);
    }
  }
  return items;
}

uint64_t fnv1a64(const std::string& input) {
  uint64_t hash = 14695981039346656037ull;
  constexpr uint64_t prime = 1099511628211ull;
  for (const auto c : input) {
    hash ^= static_cast<uint8_t>(c);
    hash *= prime;
  }
  return hash;
}

std::string toHex(const uint64_t value) {
  std::ostringstream out;
  out << std::hex << value;
  return out.str();
}

std::string iso8601NowUtc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc {};
#if defined(_WIN32)
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  std::ostringstream out;
  out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

std::optional<int> parseIntArg(const std::vector<std::string>& args, const std::string& key) {
  const auto value = argValue(args, key);
  if (!value.has_value()) {
    return std::nullopt;
  }
  try {
    return std::stoi(*value);
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<double> parseDoubleArg(const std::vector<std::string>& args, const std::string& key) {
  const auto value = argValue(args, key);
  if (!value.has_value()) {
    return std::nullopt;
  }
  try {
    return std::stod(*value);
  } catch (...) {
    return std::nullopt;
  }
}

std::string extensionForFormat(const std::string& format) {
  const auto normalized = toLower(format);
  if (normalized == "wav") {
    return ".wav";
  }
  if (normalized == "aif" || normalized == "aiff") {
    return ".aiff";
  }
  if (normalized == "flac") {
    return ".flac";
  }
  if (normalized == "mp3") {
    return ".mp3";
  }
  if (normalized == "ogg" || normalized == "vorbis") {
    return ".ogg";
  }
  return ".wav";
}

std::string csvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (const auto c : value) {
    if (c == '"') {
      escaped += "\"\"";
    } else {
      escaped += c;
    }
  }
  escaped += "\"";
  return escaped;
}

std::optional<nlohmann::json> loadJsonFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return std::nullopt;
  }
  try {
    nlohmann::json json;
    in >> json;
    return json;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::string> readTextFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return std::nullopt;
  }
  std::ostringstream content;
  content << in.rdbuf();
  return content.str();
}

void writeJsonFile(const std::filesystem::path& path, const nlohmann::json& payload) {
  std::error_code error;
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), error);
  }
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open output file: " + path.string());
  }
  out << payload.dump(2);
}

automix::engine::AudioBuffer sliceBuffer(const automix::engine::AudioBuffer& input,
                                         const int maxSamples) {
  const int outputSamples = std::max(0, std::min(input.getNumSamples(), maxSamples));
  automix::engine::AudioBuffer output(input.getNumChannels(), outputSamples, input.getSampleRate());

  for (int ch = 0; ch < input.getNumChannels(); ++ch) {
    for (int i = 0; i < outputSamples; ++i) {
      output.setSample(ch, i, input.getSample(ch, i));
    }
  }

  return output;
}

std::vector<double> deterministicFeatures(const size_t count) {
  std::vector<double> features;
  features.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    features.push_back(static_cast<double>(i + 1) / static_cast<double>(count + 1));
  }
  return features;
}

std::string taskFromModelType(const std::string& type) {
  if (type == "role_classifier") {
    return "role_classifier";
  }
  if (type == "master_parameters") {
    return "master_parameters";
  }
  return "mix_parameters";
}

struct SupportedModelPack {
  std::string id;
  std::string type;
  std::string description;
};

struct SupportedLimiter {
  std::string id;
  std::string name;
  std::string description;
  std::string licenseId;
};

const std::vector<SupportedModelPack>& supportedModelPacks() {
  static const std::vector<SupportedModelPack> packs = {
      {"demo-role-v1", "role_classifier", "Deterministic demo role-classifier pack."},
      {"demo-mix-v1", "mix_parameters", "Deterministic demo mix-parameter pack."},
      {"demo-master-v1", "master_parameters", "Deterministic demo mastering pack."},
  };
  return packs;
}

const std::vector<SupportedLimiter>& supportedLimiters() {
  static const std::vector<SupportedLimiter> limiters = {
      {"phaselimiter", "PhaseLimiter", "Phase limiter external renderer package.", "See assets/phaselimiter/licenses"},
      {"external-template", "ExternalLimiterTemplate", "Template external limiter descriptor for custom tools.", "User-supplied"},
  };
  return limiters;
}

std::optional<std::filesystem::path> findRepoPath(const std::filesystem::path& relativePath) {
  std::error_code error;
  auto current = std::filesystem::absolute(std::filesystem::current_path(error), error);
  if (error) {
    return std::nullopt;
  }
  for (int depth = 0; depth < 6; ++depth) {
    const auto candidate = current / relativePath;
    if (std::filesystem::exists(candidate, error) && !error) {
      return candidate;
    }
    if (!current.has_parent_path()) {
      break;
    }
    current = current.parent_path();
  }
  return std::nullopt;
}

void copyDirectory(const std::filesystem::path& source, const std::filesystem::path& destination) {
  std::error_code error;
  std::filesystem::create_directories(destination, error);
  if (error) {
    throw std::runtime_error("Failed to create destination directory: " + destination.string());
  }
  std::filesystem::copy(source,
                        destination,
                        std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing,
                        error);
  if (error) {
    throw std::runtime_error("Failed to copy directory: " + source.string() + " -> " + destination.string());
  }
}

struct ReportMetrics {
  bool loaded = false;
  std::string renderer;
  double integratedLufs = -120.0;
  double truePeakDbtp = 0.0;
  double targetLufs = -14.0;
  double targetTruePeakDbtp = -1.0;
  double monoCorrelation = 1.0;
  double stereoCorrelation = 1.0;
  double artifactRisk = 0.0;
};

ReportMetrics readReportMetrics(const std::filesystem::path& reportPath) {
  ReportMetrics metrics;
  const auto report = loadJsonFile(reportPath);
  if (!report.has_value() || !report->is_object()) {
    return metrics;
  }

  metrics.loaded = true;
  metrics.renderer = report->value("renderer", "");
  metrics.integratedLufs = report->value("integratedLufs", metrics.integratedLufs);
  metrics.truePeakDbtp = report->value("truePeakDbtp", metrics.truePeakDbtp);
  metrics.targetLufs = report->value("targetLufs", metrics.targetLufs);
  metrics.targetTruePeakDbtp = report->value("targetTruePeakDbtp", metrics.targetTruePeakDbtp);
  metrics.monoCorrelation = report->value("monoCorrelation", metrics.monoCorrelation);
  metrics.stereoCorrelation = report->value("stereoCorrelation", metrics.stereoCorrelation);

  if (report->contains("artifactRisk")) {
    metrics.artifactRisk = std::clamp(report->value("artifactRisk", 0.0), 0.0, 1.0);
  } else {
    const double high = report->value("spectrumHigh", 0.0);
    const double mid = report->value("spectrumMid", 0.0);
    metrics.artifactRisk = std::clamp((high - mid) * 0.8 + (1.0 - metrics.monoCorrelation) * 0.3, 0.0, 1.0);
  }

  return metrics;
}

double computeComparatorScore(const ReportMetrics& metrics) {
  const double loudnessError = std::abs(metrics.integratedLufs - metrics.targetLufs);
  const double truePeakOverflow = std::max(0.0, metrics.truePeakDbtp - metrics.targetTruePeakDbtp);
  const double monoPenalty = std::max(0.0, 0.95 - metrics.monoCorrelation) * 40.0;
  const double stereoPenalty = std::max(0.0, 0.80 - metrics.stereoCorrelation) * 20.0;
  const double artifactPenalty = metrics.artifactRisk * 15.0;
  const double score = 100.0 - loudnessError * 18.0 - truePeakOverflow * 30.0 - monoPenalty - stereoPenalty - artifactPenalty;
  return std::clamp(score, 0.0, 100.0);
}

automix::domain::BatchItemStatus batchStatusFromString(const std::string& value) {
  const auto normalized = toLower(value);
  if (normalized == "analyzing") {
    return automix::domain::BatchItemStatus::Analyzing;
  }
  if (normalized == "rendering") {
    return automix::domain::BatchItemStatus::Rendering;
  }
  if (normalized == "completed") {
    return automix::domain::BatchItemStatus::Completed;
  }
  if (normalized == "failed") {
    return automix::domain::BatchItemStatus::Failed;
  }
  if (normalized == "cancelled") {
    return automix::domain::BatchItemStatus::Cancelled;
  }
  return automix::domain::BatchItemStatus::Pending;
}

class DeterministicPlanDiffInference final : public automix::ai::IModelInference {
 public:
  bool isAvailable() const override { return true; }

  bool loadModel(const std::filesystem::path&) override {
    loaded_ = true;
    return true;
  }

  automix::ai::InferenceResult run(const automix::ai::InferenceRequest& request) const override {
    automix::ai::InferenceResult result;
    result.usedModel = loaded_;
    if (!loaded_) {
      result.logMessage = "deterministic inference not loaded";
      return result;
    }

    if (request.task == "mix_parameters") {
      result.outputs = {
          {"confidence", 0.72},
          {"global_gain_db", -0.8},
          {"global_pan_bias", 0.04},
          {"stem0_gain_db", -1.4},
          {"stem0_pan", 0.05},
      };
      result.logMessage = "deterministic mix diff inference";
      return result;
    }

    if (request.task == "master_parameters") {
      result.outputs = {
          {"confidence", 0.68},
          {"target_lufs", -13.2},
          {"pre_gain_db", 0.6},
          {"limiter_ceiling_db", -1.2},
          {"glue_ratio", 2.8},
          {"glue_threshold_db", -19.0},
      };
      result.logMessage = "deterministic master diff inference";
      return result;
    }

    result.outputs = {
        {"confidence", 0.60},
    };
    result.logMessage = "deterministic generic inference";
    return result;
  }

 private:
  bool loaded_ = true;
};

std::unique_ptr<automix::ai::IModelInference> buildPlanDiffInference(const std::optional<std::string>& modelPathArg,
                                                                     const std::string& taskLabel,
                                                                     std::vector<std::string>* notes) {
  if (modelPathArg.has_value() && !modelPathArg->empty()) {
    auto onnx = std::make_unique<automix::ai::OnnxModelInference>();
    onnx->setExecutionProviderPreference("cpu");
    onnx->setWarmupEnabled(false);
    if (onnx->loadModel(*modelPathArg)) {
      notes->push_back("Loaded " + taskLabel + " model: " + *modelPathArg);
      return onnx;
    }
    notes->push_back("Failed to load " + taskLabel + " model (" + *modelPathArg +
                     "). Falling back to deterministic model-diff adapter.");
  } else {
    notes->push_back("No " + taskLabel + " model path provided; using deterministic model-diff adapter.");
  }

  return std::make_unique<DeterministicPlanDiffInference>();
}

struct JsonMergeTelemetry {
  bool preferRight = true;
  size_t conflictCount = 0;
  std::vector<std::string> conflictPaths;
};

void recordMergeConflict(JsonMergeTelemetry* telemetry, const std::string& path) {
  if (telemetry == nullptr) {
    return;
  }
  ++telemetry->conflictCount;
  if (telemetry->conflictPaths.size() < 200) {
    telemetry->conflictPaths.push_back(path.empty() ? "/" : path);
  }
}

bool isDecisionLogPath(const std::string& path) {
  return path == "/mixPlan/decisionLog" || path == "/masterPlan/decisionLog";
}

bool keyedMergePath(const std::string& path, std::string* keyField) {
  if (path == "/stems") {
    *keyField = "id";
    return true;
  }
  if (path == "/buses") {
    *keyField = "id";
    return true;
  }
  if (path == "/mixPlan/stemDecisions") {
    *keyField = "stemId";
    return true;
  }
  return false;
}

std::optional<nlohmann::json> mergeJsonNode(const std::optional<nlohmann::json>& base,
                                            const std::optional<nlohmann::json>& left,
                                            const std::optional<nlohmann::json>& right,
                                            const std::string& path,
                                            JsonMergeTelemetry* telemetry);

std::optional<nlohmann::json> mergeStringArrayUnion(const std::optional<nlohmann::json>& left,
                                                    const std::optional<nlohmann::json>& right) {
  if (!left.has_value() && !right.has_value()) {
    return std::nullopt;
  }

  nlohmann::json merged = nlohmann::json::array();
  std::set<std::string> seen;

  const auto append = [&](const std::optional<nlohmann::json>& arrayJson) {
    if (!arrayJson.has_value() || !arrayJson->is_array()) {
      return;
    }
    for (const auto& item : *arrayJson) {
      if (!item.is_string()) {
        continue;
      }
      const auto value = item.get<std::string>();
      if (seen.insert(value).second) {
        merged.push_back(value);
      }
    }
  };

  append(left);
  append(right);
  return merged;
}

bool mapArrayByKey(const std::optional<nlohmann::json>& value,
                   const std::string& keyField,
                   std::map<std::string, nlohmann::json>* out) {
  if (!value.has_value()) {
    return true;
  }
  if (!value->is_array()) {
    return false;
  }
  for (const auto& item : *value) {
    if (!item.is_object() || !item.contains(keyField)) {
      return false;
    }
    std::string key;
    if (item.at(keyField).is_string()) {
      key = item.at(keyField).get<std::string>();
    } else {
      key = item.at(keyField).dump();
    }
    (*out)[key] = item;
  }
  return true;
}

std::optional<nlohmann::json> mergeKeyedArray(const std::optional<nlohmann::json>& base,
                                              const std::optional<nlohmann::json>& left,
                                              const std::optional<nlohmann::json>& right,
                                              const std::string& path,
                                              const std::string& keyField,
                                              JsonMergeTelemetry* telemetry) {
  std::map<std::string, nlohmann::json> baseMap;
  std::map<std::string, nlohmann::json> leftMap;
  std::map<std::string, nlohmann::json> rightMap;
  if (!mapArrayByKey(base, keyField, &baseMap) ||
      !mapArrayByKey(left, keyField, &leftMap) ||
      !mapArrayByKey(right, keyField, &rightMap)) {
    return std::nullopt;
  }

  std::set<std::string> keys;
  for (const auto& [key, _] : baseMap) {
    keys.insert(key);
  }
  for (const auto& [key, _] : leftMap) {
    keys.insert(key);
  }
  for (const auto& [key, _] : rightMap) {
    keys.insert(key);
  }

  nlohmann::json merged = nlohmann::json::array();
  for (const auto& key : keys) {
    const auto baseIt = baseMap.find(key);
    const auto leftIt = leftMap.find(key);
    const auto rightIt = rightMap.find(key);

    const std::optional<nlohmann::json> baseItem =
        baseIt == baseMap.end() ? std::nullopt : std::optional<nlohmann::json>(baseIt->second);
    const std::optional<nlohmann::json> leftItem =
        leftIt == leftMap.end() ? std::nullopt : std::optional<nlohmann::json>(leftIt->second);
    const std::optional<nlohmann::json> rightItem =
        rightIt == rightMap.end() ? std::nullopt : std::optional<nlohmann::json>(rightIt->second);

    const auto mergedItem = mergeJsonNode(baseItem,
                                          leftItem,
                                          rightItem,
                                          path + "/" + keyField + "=" + key,
                                          telemetry);
    if (mergedItem.has_value()) {
      merged.push_back(mergedItem.value());
    }
  }

  return merged;
}

std::optional<nlohmann::json> mergeJsonNode(const std::optional<nlohmann::json>& base,
                                            const std::optional<nlohmann::json>& left,
                                            const std::optional<nlohmann::json>& right,
                                            const std::string& path,
                                            JsonMergeTelemetry* telemetry) {
  if (left == right) {
    return left;
  }

  if (left == base) {
    return right;
  }

  if (right == base) {
    return left;
  }

  if (!left.has_value() && !right.has_value()) {
    return std::nullopt;
  }

  if (left.has_value() && right.has_value() &&
      left->is_object() && right->is_object()) {
    nlohmann::json merged = nlohmann::json::object();
    std::set<std::string> keys;
    if (base.has_value() && base->is_object()) {
      for (const auto& [key, _] : base->items()) {
        keys.insert(key);
      }
    }
    for (const auto& [key, _] : left->items()) {
      keys.insert(key);
    }
    for (const auto& [key, _] : right->items()) {
      keys.insert(key);
    }

    for (const auto& key : keys) {
      const auto nextPath = path.empty() ? ("/" + key) : (path + "/" + key);

      const std::optional<nlohmann::json> baseChild =
          (base.has_value() && base->is_object() && base->contains(key))
              ? std::optional<nlohmann::json>(base->at(key))
              : std::nullopt;
      const std::optional<nlohmann::json> leftChild =
          left->contains(key) ? std::optional<nlohmann::json>(left->at(key)) : std::nullopt;
      const std::optional<nlohmann::json> rightChild =
          right->contains(key) ? std::optional<nlohmann::json>(right->at(key)) : std::nullopt;

      const auto mergedChild = mergeJsonNode(baseChild, leftChild, rightChild, nextPath, telemetry);
      if (mergedChild.has_value()) {
        merged[key] = mergedChild.value();
      }
    }

    return merged;
  }

  if (left.has_value() && right.has_value() &&
      left->is_array() && right->is_array()) {
    if (isDecisionLogPath(path)) {
      return mergeStringArrayUnion(left, right);
    }

    std::string keyField;
    if (keyedMergePath(path, &keyField)) {
      if (const auto merged = mergeKeyedArray(base, left, right, path, keyField, telemetry); merged.has_value()) {
        return merged;
      }
    }

    recordMergeConflict(telemetry, path);
    return telemetry->preferRight ? right : left;
  }

  recordMergeConflict(telemetry, path);
  return telemetry->preferRight ? right : left;
}

int commandListSupportedModels() {
  std::cout << "Supported model packs:\n";
  for (const auto& pack : supportedModelPacks()) {
    std::cout << "  - " << pack.id << " [" << pack.type << "] " << pack.description << "\n";
  }
  return 0;
}

int commandInstallSupportedModel(const std::vector<std::string>& args) {
  const auto idArg = argValue(args, "--id");
  if (!idArg.has_value()) {
    std::cerr << "install-supported-model requires --id <model_id>\n";
    return 2;
  }
  const std::filesystem::path destinationRoot = argValue(args, "--dest").value_or("assets/models");

  const auto it = std::find_if(supportedModelPacks().begin(), supportedModelPacks().end(),
                               [&](const SupportedModelPack& pack) { return pack.id == *idArg; });
  if (it == supportedModelPacks().end()) {
    std::cerr << "Unknown supported model id: " << *idArg << "\n";
    return 2;
  }

  const auto source = findRepoPath(std::filesystem::path("tools/catalog/modelpacks") / it->id);
  if (!source.has_value()) {
    std::cerr << "Catalog source not found for model: " << it->id << "\n";
    return 1;
  }

  const auto destination = destinationRoot / it->id;
  copyDirectory(source.value(), destination);
  std::cout << "Installed model pack '" << it->id << "' to " << destination.string() << "\n";
  return 0;
}

int commandListSupportedLimiters() {
  std::cout << "Supported limiters:\n";
  for (const auto& limiter : supportedLimiters()) {
    std::cout << "  - " << limiter.id << " [" << limiter.name << "] " << limiter.description << "\n";
  }
  return 0;
}

int commandInstallSupportedLimiter(const std::vector<std::string>& args) {
  const auto idArg = argValue(args, "--id");
  if (!idArg.has_value()) {
    std::cerr << "install-supported-limiter requires --id <limiter_id>\n";
    return 2;
  }
  const std::filesystem::path destinationRoot = argValue(args, "--dest").value_or("assets/limiters");
  std::filesystem::create_directories(destinationRoot);

  if (*idArg == "phaselimiter") {
    const auto source = findRepoPath("assets/phaselimiter");
    if (!source.has_value()) {
      std::cerr << "PhaseLimiter source package not found under assets/phaselimiter\n";
      return 1;
    }
    const auto destination = destinationRoot / "phaselimiter";
    std::filesystem::create_directories(destination);
    copyDirectory(source.value(), destination / "runtime");

    nlohmann::json descriptor = {
        {"id", "PhaseLimiterPack"},
        {"name", "PhaseLimiter (pack)"},
        {"version", "external"},
        {"licenseId", "See assets/phaselimiter/licenses"},
        {"binaryPath", "runtime/phase_limiter"},
        {"bundledByDefault", false},
    };
    std::ofstream out(destination / "renderer.json");
    out << descriptor.dump(2);
    std::cout << "Installed limiter '" << *idArg << "' to " << destination.string() << "\n";
    return 0;
  }

  if (*idArg == "external-template") {
    const auto destination = destinationRoot / "external-template";
    std::filesystem::create_directories(destination);
    nlohmann::json descriptor = {
        {"id", "ExternalTemplate"},
        {"name", "External Limiter Template"},
        {"version", "1.0"},
        {"licenseId", "User-supplied"},
        {"binaryPath", "your_limiter_binary_here"},
        {"bundledByDefault", false},
    };
    std::ofstream out(destination / "renderer.json");
    out << descriptor.dump(2);
    std::cout << "Installed limiter template to " << destination.string() << "\n";
    return 0;
  }

  std::cerr << "Unknown supported limiter id: " << *idArg << "\n";
  return 2;
}

int commandInstallLameFallback(const std::vector<std::string>& args) {
  const bool force = hasFlag(args, "--force");
  const bool jsonOutput = hasFlag(args, "--json");

  const auto result = automix::util::LameDownloader::ensureAvailable(force);
  const auto cachePath = automix::util::LameDownloader::cacheBinaryPath();

  if (jsonOutput) {
    nlohmann::json payload = {
        {"success", result.success},
        {"attempted", result.attempted},
        {"path", result.success ? result.executablePath.string() : cachePath.string()},
        {"detail", result.detail},
    };
    std::cout << payload.dump(2) << "\n";
    return result.success ? 0 : 1;
  }

  std::cout << "LAME fallback installation:\n";
  std::cout << "  Success: " << (result.success ? "yes" : "no") << "\n";
  std::cout << "  Attempted download: " << (result.attempted ? "yes" : "no") << "\n";
  std::cout << "  Binary path: " << (result.success ? result.executablePath.string() : cachePath.string()) << "\n";
  if (!result.detail.empty()) {
    std::cout << "  Detail: " << result.detail << "\n";
  }

  return result.success ? 0 : 1;
}

int commandExportFeatures(const std::vector<std::string>& args) {
  const auto sessionPathArg = argValue(args, "--session");
  const auto outPathArg = argValue(args, "--out");
  if (!sessionPathArg.has_value() || !outPathArg.has_value()) {
    std::cerr << "export-features requires --session <session.json> and --out <features.jsonl>\n";
    return 2;
  }
  const auto manifestPathArg = argValue(args, "--manifest");
  const auto datasetIdArg = argValue(args, "--dataset-id");
  const auto sourceTagArg = argValue(args, "--source-tag");
  const auto lineageParentsArg = argValue(args, "--lineage-parents");

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  std::unordered_map<std::string, automix::domain::Stem> stemsById;
  for (const auto& stem : session.stems) {
    stemsById.emplace(stem.id, stem);
  }

  automix::analysis::StemAnalyzer analyzer;
  const auto entries = analyzer.analyzeSession(session);

  std::ofstream out(*outPathArg);
  if (!out.is_open()) {
    std::cerr << "Failed to open output file: " << *outPathArg << "\n";
    return 1;
  }

  for (const auto& entry : entries) {
    nlohmann::json line;
    line["sessionName"] = session.sessionName;
    line["stemId"] = entry.stemId;
    line["stemName"] = entry.stemName;
    line["metrics"] = {
        {"peakDb", entry.metrics.peakDb},
        {"rmsDb", entry.metrics.rmsDb},
        {"crestDb", entry.metrics.crestDb},
        {"crestFactor", entry.metrics.crestFactor},
        {"lowEnergy", entry.metrics.lowEnergy},
        {"midEnergy", entry.metrics.midEnergy},
        {"highEnergy", entry.metrics.highEnergy},
        {"silenceRatio", entry.metrics.silenceRatio},
        {"stereoCorrelation", entry.metrics.stereoCorrelation},
        {"stereoWidth", entry.metrics.stereoWidth},
        {"dcOffset", entry.metrics.dcOffset},
        {"subEnergy", entry.metrics.subEnergy},
        {"bassEnergy", entry.metrics.bassEnergy},
        {"lowMidEnergy", entry.metrics.lowMidEnergy},
        {"highMidEnergy", entry.metrics.highMidEnergy},
        {"presenceEnergy", entry.metrics.presenceEnergy},
        {"airEnergy", entry.metrics.airEnergy},
        {"spectralCentroidHz", entry.metrics.spectralCentroidHz},
        {"spectralSpreadHz", entry.metrics.spectralSpreadHz},
        {"spectralFlatness", entry.metrics.spectralFlatness},
        {"spectralFlux", entry.metrics.spectralFlux},
        {"onsetStrength", entry.metrics.onsetStrength},
        {"mfccCoefficients", entry.metrics.mfccCoefficients},
        {"constantQBins", entry.metrics.constantQBins},
        {"channelBalanceDb", entry.metrics.channelBalanceDb},
        {"artifactRisk", entry.metrics.artifactRisk},
        {"artifactSwirlRisk", entry.metrics.artifactProfile.swirlRisk},
        {"artifactSmearRisk", entry.metrics.artifactProfile.smearRisk},
        {"artifactNoiseDominance", entry.metrics.artifactProfile.noiseDominance},
        {"artifactHarmonicity", entry.metrics.artifactProfile.harmonicity},
        {"artifactPhaseInstability", entry.metrics.artifactProfile.phaseInstability},
    };

    const auto it = stemsById.find(entry.stemId);
    if (it != stemsById.end()) {
      line["origin"] = automix::domain::toString(it->second.origin);
      line["role"] = automix::domain::toString(it->second.role);
    }

    out << line.dump() << "\n";
  }

  std::cout << "Exported " << entries.size() << " feature rows to " << *outPathArg << "\n";

  if (manifestPathArg.has_value()) {
    const auto sessionText = readTextFile(*sessionPathArg).value_or("");
    const auto featureText = readTextFile(*outPathArg).value_or("");
    const auto lineageParents =
        lineageParentsArg.has_value() ? splitCommaSeparated(*lineageParentsArg) : std::vector<std::string>{};
    const auto datasetId = datasetIdArg.has_value() && !datasetIdArg->empty()
                               ? *datasetIdArg
                               : (sanitizeFileName(session.sessionName) + "_" + toHex(fnv1a64(iso8601NowUtc())));

    nlohmann::json manifest = {
        {"schemaVersion", 1},
        {"generatedAtUtc", iso8601NowUtc()},
        {"datasetId", datasetId},
        {"sourceTag", sourceTagArg.value_or("manual")},
        {"sourceSessionPath", *sessionPathArg},
        {"sessionName", session.sessionName},
        {"rowCount", entries.size()},
        {"featureSchemaVersion", "v1"},
        {"featureCountPerStem", automix::ai::FeatureSchemaV1::featureCount()},
        {"featureFilePath", *outPathArg},
        {"lineageParents", lineageParents},
        {"sessionHashFnv1a64", toHex(fnv1a64(sessionText))},
        {"featureFileHashFnv1a64", toHex(fnv1a64(featureText))},
        {"columns", nlohmann::json::array({
                        "peakDb", "rmsDb", "crestDb", "crestFactor", "lowEnergy", "midEnergy",
                        "highEnergy", "silenceRatio", "stereoCorrelation", "stereoWidth", "dcOffset",
                        "subEnergy", "bassEnergy", "lowMidEnergy", "highMidEnergy", "presenceEnergy",
                        "airEnergy", "spectralCentroidHz", "spectralSpreadHz", "spectralFlatness",
                        "spectralFlux", "onsetStrength", "mfccCoefficients", "constantQBins",
                        "channelBalanceDb", "artifactRisk", "artifactSwirlRisk", "artifactSmearRisk",
                        "artifactNoiseDominance", "artifactHarmonicity", "artifactPhaseInstability",
                    })},
    };

    writeJsonFile(*manifestPathArg, manifest);
    std::cout << "Wrote feature lineage manifest to " << *manifestPathArg << "\n";
  }

  return 0;
}

int commandExportSegments(const std::vector<std::string>& args) {
  const auto sessionPathArg = argValue(args, "--session");
  const auto outDirArg = argValue(args, "--out-dir");
  if (!sessionPathArg.has_value() || !outDirArg.has_value()) {
    std::cerr << "export-segments requires --session <session.json> and --out-dir <directory>\n";
    return 2;
  }

  double segmentSeconds = 5.0;
  if (const auto secondsArg = argValue(args, "--segment-seconds"); secondsArg.has_value()) {
    segmentSeconds = std::max(0.5, std::stod(*secondsArg));
  }

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  const std::filesystem::path outDir(*outDirArg);
  const std::filesystem::path stemDir = outDir / "stems";
  std::filesystem::create_directories(stemDir);

  automix::engine::AudioFileIO fileIO;
  automix::util::WavWriter writer;
  for (const auto& stem : session.stems) {
    if (!stem.enabled) {
      continue;
    }

    const auto buffer = fileIO.readAudioFile(stem.filePath);
    const int maxSamples = static_cast<int>(segmentSeconds * buffer.getSampleRate());
    const auto segment = sliceBuffer(buffer, maxSamples);
    const auto outPath = stemDir / (sanitizeFileName(stem.id + "_" + stem.name) + ".wav");
    writer.write(outPath, segment, 24);
  }

  automix::domain::RenderSettings settings = session.renderSettings;
  settings.outputSampleRate = settings.outputSampleRate > 0 ? settings.outputSampleRate : 44100;
  settings.blockSize = settings.blockSize > 0 ? settings.blockSize : 1024;
  settings.outputBitDepth = 24;
  settings.outputPath = (outDir / "mix_full.wav").string();

  automix::engine::OfflineRenderPipeline pipeline;
  const auto rawMix = pipeline.renderRawMix(session, settings, {}, nullptr);
  if (rawMix.cancelled) {
    std::cerr << "Raw mix render cancelled while exporting segments.\n";
    return 1;
  }

  const int maxMixSamples = static_cast<int>(segmentSeconds * rawMix.mixBuffer.getSampleRate());
  const auto mixSegment = sliceBuffer(rawMix.mixBuffer, maxMixSamples);
  writer.write(outDir / "mix_segment.wav", mixSegment, 24);

  nlohmann::json manifest = {
      {"sessionName", session.sessionName},
      {"segmentSeconds", segmentSeconds},
      {"stemCount", session.stems.size()},
      {"mixSegmentPath", (outDir / "mix_segment.wav").string()},
  };
  std::ofstream manifestOut(outDir / "manifest.json");
  manifestOut << manifest.dump(2);

  std::cout << "Exported stem and mix segments to " << outDir.string() << "\n";
  return 0;
}

int commandValidateModelPack(const std::vector<std::string>& args) {
  const auto packArg = argValue(args, "--pack");
  if (!packArg.has_value()) {
    std::cerr << "validate-modelpack requires --pack <directory>\n";
    return 2;
  }

  const std::filesystem::path packDir(*packArg);
  automix::ai::ModelPackLoader loader;
  const auto maybePack = loader.load(packDir);
  if (!maybePack.has_value()) {
    std::cerr << "Model pack validation failed: could not load model.json or model file.\n";
    return 1;
  }
  const auto& pack = maybePack.value();

  std::cout << "Model pack loaded: " << pack.id << " engine=" << pack.engine
            << " type=" << pack.type << " version=" << pack.version << "\n";

  std::unique_ptr<automix::ai::IModelInference> inference = std::make_unique<automix::ai::NullModelInference>();
  if (pack.engine == "onnxruntime") {
    inference = std::make_unique<automix::ai::OnnxModelInference>();
  } else if (pack.engine == "rtneural") {
    inference = std::make_unique<automix::ai::RtNeuralInference>();
    if (!inference->isAvailable()) {
      std::cout << "Warning: RTNeural backend not enabled in this build. Schema-only validation performed.\n";
    }
  }

  const auto modelPath = pack.rootPath / pack.modelFile;
  if (!inference->loadModel(modelPath)) {
    if (pack.engine == "unknown") {
      std::cout << "Warning: unknown model engine; skipped runtime inference validation.\n";
      return 0;
    }
    if (!inference->isAvailable()) {
      return 0;
    }
    std::cerr << "Model pack validation failed: backend refused to load model file.\n";
    return 1;
  }

  const size_t featureCount = pack.inputFeatureCount.value_or(automix::ai::FeatureSchemaV1::featureCount());
  const automix::ai::InferenceRequest request{
      .task = taskFromModelType(pack.type),
      .features = deterministicFeatures(featureCount),
  };
  const auto result = inference->run(request);
  if (!result.usedModel) {
    std::cerr << "Model pack validation failed: sample inference did not use model (" << result.logMessage << ")\n";
    return 1;
  }

  for (const auto& key : pack.expectedOutputKeys) {
    if (!result.outputs.contains(key)) {
      std::cerr << "Model pack validation failed: missing expected output key '" << key << "'\n";
      return 1;
    }
  }

  std::cout << "Model pack validation passed.\n";
  return 0;
}

int commandValidateExternalLimiter(const std::vector<std::string>& args) {
  const auto binaryArg = argValue(args, "--binary");
  if (!binaryArg.has_value()) {
    std::cerr << "validate-external-limiter requires --binary <path>\n";
    return 2;
  }

  const bool jsonOutput = hasFlag(args, "--json");
  const std::filesystem::path binaryPath(*binaryArg);
  const auto validation = automix::renderers::ExternalLimiterRenderer::validateBinary(binaryPath);

  if (jsonOutput) {
    nlohmann::json payload = {
        {"binary", binaryPath.string()},
        {"valid", validation.valid},
        {"version", validation.version},
        {"errorCode", validation.errorCode},
        {"diagnostics", validation.diagnostics},
        {"supportedFeatures", validation.supportedFeatures},
    };
    std::cout << payload.dump(2) << "\n";
    return validation.valid ? 0 : 1;
  }

  std::cout << "External limiter validation summary:\n";
  std::cout << "  Binary: " << binaryPath.string() << "\n";
  std::cout << "  Valid: " << (validation.valid ? "yes" : "no") << "\n";
  std::cout << "  Version: " << (validation.version.empty() ? "(none)" : validation.version) << "\n";
  std::cout << "  Error code: " << (validation.errorCode.empty() ? "(none)" : validation.errorCode) << "\n";
  std::cout << "  Diagnostics: " << validation.diagnostics << "\n";

  if (!validation.supportedFeatures.empty()) {
    std::cout << "  Supported features:\n";
    for (const auto& feature : validation.supportedFeatures) {
      std::cout << "    - " << feature << "\n";
    }
  } else {
    std::cout << "  Supported features: (none reported)\n";
  }

  return validation.valid ? 0 : 1;
}

int commandStemHealth(const std::vector<std::string>& args) {
  const auto sessionPathArg = argValue(args, "--session");
  if (!sessionPathArg.has_value()) {
    std::cerr << "stem-health requires --session <session.json>\n";
    return 2;
  }

  const auto outPathArg = argValue(args, "--out");
  const bool jsonOutput = hasFlag(args, "--json");

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  automix::analysis::StemAnalyzer analyzer;
  const auto analysisEntries = analyzer.analyzeSession(session);
  automix::analysis::StemHealthAssistant assistant;
  const auto report = assistant.analyze(session, analysisEntries);
  const auto reportJson = assistant.toJson(report);
  const auto reportText = assistant.toText(report);

  if (outPathArg.has_value()) {
    std::filesystem::path outPath(*outPathArg);
    std::filesystem::path jsonPath = outPath;
    std::filesystem::path textPath = outPath;

    if (outPath.extension() == ".json") {
      textPath.replace_extension(".txt");
    } else if (outPath.extension() == ".txt") {
      jsonPath.replace_extension(".json");
    } else {
      jsonPath += ".json";
      textPath += ".txt";
    }

    writeJsonFile(jsonPath, reportJson);
    std::ofstream textOut(textPath);
    if (textOut.is_open()) {
      textOut << reportText << "\n";
    }
    std::cout << "Wrote stem health report JSON: " << jsonPath.string() << "\n";
    std::cout << "Wrote stem health report text: " << textPath.string() << "\n";
  }

  if (jsonOutput) {
    std::cout << reportJson.dump(2) << "\n";
  } else {
    std::cout << reportText << "\n";
  }

  return 0;
}

int commandCompareRenders(const std::vector<std::string>& args) {
  const auto sessionPathArg = argValue(args, "--session");
  if (!sessionPathArg.has_value()) {
    std::cerr << "compare-renders requires --session <session.json>\n";
    return 2;
  }

  const auto outDirArg = argValue(args, "--out-dir").value_or("comparison_out");
  const auto renderersArg = argValue(args, "--renderers").value_or("BuiltIn,PhaseLimiter");
  auto rendererIds = splitCommaSeparated(renderersArg);
  if (rendererIds.empty()) {
    rendererIds = {"BuiltIn"};
  }

  const auto formatOverride = argValue(args, "--format");
  const auto externalBinaryArg = argValue(args, "--external-binary");
  const bool jsonOutput = hasFlag(args, "--json");

  struct ComparatorRow {
    std::string rendererId;
    bool success = false;
    bool cancelled = false;
    std::string outputPath;
    std::string reportPath;
    std::string message;
    ReportMetrics metrics;
    double score = 0.0;
  };

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  const std::filesystem::path outDir(outDirArg);
  std::filesystem::create_directories(outDir);
  const auto stem = sanitizeFileName(session.sessionName.empty() ? "session" : session.sessionName);

  std::vector<ComparatorRow> rows;
  rows.reserve(rendererIds.size());

  for (const auto& rendererId : rendererIds) {
    auto settings = session.renderSettings;
    settings.rendererName = rendererId;
    if (formatOverride.has_value()) {
      settings.outputFormat = *formatOverride;
    }
    if (settings.outputFormat.empty() || settings.outputFormat == "auto") {
      settings.outputFormat = "wav";
    }
    if (externalBinaryArg.has_value()) {
      settings.externalRendererPath = *externalBinaryArg;
    }

    const auto outputName = stem + "_" + sanitizeFileName(rendererId) + extensionForFormat(settings.outputFormat);
    settings.outputPath = (outDir / outputName).string();

    ComparatorRow row;
    row.rendererId = rendererId;
    try {
      auto renderer = automix::renderers::createRenderer(rendererId);
      const auto result = renderer->render(session, settings, {}, nullptr);
      row.success = result.success;
      row.cancelled = result.cancelled;
      row.outputPath = result.outputAudioPath;
      row.reportPath = result.reportPath;
      row.message = result.logs.empty() ? "" : result.logs.back();

      std::filesystem::path reportCandidate(result.reportPath);
      if (reportCandidate.empty()) {
        reportCandidate = std::filesystem::path(settings.outputPath + ".report.json");
      }
      row.metrics = readReportMetrics(reportCandidate);
      row.score = row.success && row.metrics.loaded ? computeComparatorScore(row.metrics)
                                                    : (row.success ? 50.0 : 0.0);
    } catch (const std::exception& error) {
      row.success = false;
      row.message = error.what();
    }

    rows.push_back(row);
  }

  std::sort(rows.begin(), rows.end(), [](const ComparatorRow& a, const ComparatorRow& b) {
    if (a.success != b.success) {
      return a.success > b.success;
    }
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.rendererId < b.rendererId;
  });

  nlohmann::json ranking = nlohmann::json::array();
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows[i];
    ranking.push_back({
        {"rank", i + 1},
        {"rendererId", row.rendererId},
        {"success", row.success},
        {"cancelled", row.cancelled},
        {"score", row.score},
        {"outputPath", row.outputPath},
        {"reportPath", row.reportPath},
        {"message", row.message},
        {"metrics",
         {
             {"integratedLufs", row.metrics.integratedLufs},
             {"targetLufs", row.metrics.targetLufs},
             {"truePeakDbtp", row.metrics.truePeakDbtp},
             {"targetTruePeakDbtp", row.metrics.targetTruePeakDbtp},
             {"monoCorrelation", row.metrics.monoCorrelation},
             {"stereoCorrelation", row.metrics.stereoCorrelation},
             {"artifactRisk", row.metrics.artifactRisk},
         }},
    });
  }

  nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"sessionPath", *sessionPathArg},
      {"outDir", outDir.string()},
      {"renderers", rendererIds},
      {"ranking", ranking},
  };
  const auto jsonReportPath = outDir / "comparison_report.json";
  writeJsonFile(jsonReportPath, payload);

  const auto csvPath = outDir / "comparison_report.csv";
  std::ofstream csv(csvPath);
  if (csv.is_open()) {
    csv << "rank,renderer,success,cancelled,score,integrated_lufs,target_lufs,true_peak_dbtp,target_true_peak_dbtp,mono_corr,stereo_corr,artifact_risk,output_path,report_path,message\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto& row = rows[i];
      csv << (i + 1) << ","
          << csvEscape(row.rendererId) << ","
          << (row.success ? "true" : "false") << ","
          << (row.cancelled ? "true" : "false") << ","
          << row.score << ","
          << row.metrics.integratedLufs << ","
          << row.metrics.targetLufs << ","
          << row.metrics.truePeakDbtp << ","
          << row.metrics.targetTruePeakDbtp << ","
          << row.metrics.monoCorrelation << ","
          << row.metrics.stereoCorrelation << ","
          << row.metrics.artifactRisk << ","
          << csvEscape(row.outputPath) << ","
          << csvEscape(row.reportPath) << ","
          << csvEscape(row.message) << "\n";
    }
  }

  if (jsonOutput) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Multi-render comparison complete. Ranking:\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto& row = rows[i];
      std::cout << "  " << (i + 1) << ". " << row.rendererId
                << " success=" << (row.success ? "yes" : "no")
                << " score=" << std::fixed << std::setprecision(2) << row.score
                << " LUFS=" << row.metrics.integratedLufs
                << " TP=" << row.metrics.truePeakDbtp << "\n";
    }
    std::cout << "JSON report: " << jsonReportPath.string() << "\n";
    std::cout << "CSV report: " << csvPath.string() << "\n";
  }

  const auto successes = std::count_if(rows.begin(), rows.end(), [](const ComparatorRow& row) { return row.success; });
  return successes > 0 ? 0 : 1;
}

int commandCatalogProcess(const std::vector<std::string>& args) {
  const auto inputArg = argValue(args, "--input");
  const auto outputArg = argValue(args, "--output");
  if (!inputArg.has_value() || !outputArg.has_value()) {
    std::cerr << "catalog-process requires --input <folder> --output <folder>\n";
    return 2;
  }

  const std::filesystem::path inputDir(*inputArg);
  const std::filesystem::path outputDir(*outputArg);
  std::filesystem::create_directories(outputDir);

  const auto checkpointPath =
      std::filesystem::path(argValue(args, "--checkpoint").value_or((outputDir / "catalog_checkpoint.json").string()));
  const bool resume = hasFlag(args, "--resume");
  const auto csvPath =
      std::filesystem::path(argValue(args, "--csv").value_or((outputDir / "catalog_deliverables.csv").string()));
  const auto jsonPath =
      std::filesystem::path(argValue(args, "--json").value_or((outputDir / "catalog_deliverables.json").string()));

  automix::engine::BatchQueueRunner runner;
  auto discoveredItems = runner.buildItemsFromFolder(inputDir, outputDir);
  if (discoveredItems.empty()) {
    std::cout << "No audio items found in " << inputDir.string() << "\n";
    nlohmann::json emptyPayload = {
        {"generatedAtUtc", iso8601NowUtc()},
        {"inputDir", inputDir.string()},
        {"outputDir", outputDir.string()},
        {"summary", {{"total", 0}, {"completed", 0}, {"failed", 0}, {"cancelled", 0}}},
        {"items", nlohmann::json::array()},
    };
    writeJsonFile(jsonPath, emptyPayload);
    return 0;
  }

  std::unordered_map<std::string, nlohmann::json> checkpointBySession;
  if (resume) {
    if (const auto checkpoint = loadJsonFile(checkpointPath); checkpoint.has_value() && checkpoint->contains("items")) {
      for (const auto& item : checkpoint->at("items")) {
        if (!item.is_object()) {
          continue;
        }
        const auto sessionName = item.value("sessionName", "");
        if (!sessionName.empty()) {
          checkpointBySession[sessionName] = item;
        }
      }
    }
  }

  std::vector<automix::domain::BatchItem> completedFromCheckpoint;
  std::vector<automix::domain::BatchItem> pendingItems;
  completedFromCheckpoint.reserve(discoveredItems.size());
  pendingItems.reserve(discoveredItems.size());

  for (auto& item : discoveredItems) {
    const auto it = checkpointBySession.find(item.session.sessionName);
    if (it != checkpointBySession.end()) {
      const auto& checkpointItem = it->second;
      item.status = batchStatusFromString(checkpointItem.value("status", "pending"));
      item.error = checkpointItem.value("error", "");
      item.reportPath = checkpointItem.value("reportPath", "");
      if (checkpointItem.contains("outputPath")) {
        item.outputPath = checkpointItem.value("outputPath", item.outputPath.string());
      }
    }

    const bool checkpointCompleted =
        item.status == automix::domain::BatchItemStatus::Completed &&
        !item.reportPath.empty() &&
        std::filesystem::exists(item.reportPath);

    if (checkpointCompleted) {
      completedFromCheckpoint.push_back(item);
      continue;
    }

    item.status = automix::domain::BatchItemStatus::Pending;
    pendingItems.push_back(item);
  }

  automix::domain::BatchJob job;
  job.items = std::move(pendingItems);
  job.settings.outputFolder = outputDir;
  job.settings.renderSettings.rendererName = argValue(args, "--renderer").value_or("BuiltIn");
  job.settings.renderSettings.outputFormat = argValue(args, "--format").value_or("wav");
  job.settings.analysisThreads = parseIntArg(args, "--analysis-threads").value_or(1);
  job.settings.renderParallelism = parseIntArg(args, "--render-parallelism").value_or(1);
  job.settings.parallelAnalysis = !hasFlag(args, "--serial-analysis");

  std::atomic_bool cancelFlag {false};
  const auto processResult = runner.process(job, {}, &cancelFlag);

  std::vector<automix::domain::BatchItem> allItems = completedFromCheckpoint;
  allItems.insert(allItems.end(), job.items.begin(), job.items.end());
  std::sort(allItems.begin(), allItems.end(), [](const automix::domain::BatchItem& a, const automix::domain::BatchItem& b) {
    return a.session.sessionName < b.session.sessionName;
  });

  int completed = 0;
  int failed = 0;
  int cancelled = 0;
  nlohmann::json itemPayload = nlohmann::json::array();
  for (const auto& item : allItems) {
    completed += item.status == automix::domain::BatchItemStatus::Completed ? 1 : 0;
    failed += item.status == automix::domain::BatchItemStatus::Failed ? 1 : 0;
    cancelled += item.status == automix::domain::BatchItemStatus::Cancelled ? 1 : 0;

    const auto metrics = item.reportPath.empty() ? ReportMetrics{} : readReportMetrics(item.reportPath);
    itemPayload.push_back({
        {"sessionName", item.session.sessionName},
        {"status", automix::domain::toString(item.status)},
        {"outputPath", item.outputPath.string()},
        {"reportPath", item.reportPath},
        {"error", item.error},
        {"integratedLufs", metrics.integratedLufs},
        {"truePeakDbtp", metrics.truePeakDbtp},
        {"artifactRisk", metrics.artifactRisk},
    });
  }

  const nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"inputDir", inputDir.string()},
      {"outputDir", outputDir.string()},
      {"checkpoint", checkpointPath.string()},
      {"resumeEnabled", resume},
      {"processedInThisRun",
       {
           {"completed", processResult.completed},
           {"failed", processResult.failed},
           {"cancelled", processResult.cancelled},
       }},
      {"summary",
       {
           {"total", static_cast<int>(allItems.size())},
           {"completed", completed},
           {"failed", failed},
           {"cancelled", cancelled},
       }},
      {"items", itemPayload},
  };

  writeJsonFile(jsonPath, payload);
  writeJsonFile(checkpointPath, payload);

  std::ofstream csv(csvPath);
  if (csv.is_open()) {
    csv << "session_name,status,output_path,report_path,integrated_lufs,true_peak_dbtp,artifact_risk,error\n";
    for (const auto& item : allItems) {
      const auto metrics = item.reportPath.empty() ? ReportMetrics{} : readReportMetrics(item.reportPath);
      csv << csvEscape(item.session.sessionName) << ","
          << csvEscape(automix::domain::toString(item.status)) << ","
          << csvEscape(item.outputPath.string()) << ","
          << csvEscape(item.reportPath) << ","
          << metrics.integratedLufs << ","
          << metrics.truePeakDbtp << ","
          << metrics.artifactRisk << ","
          << csvEscape(item.error) << "\n";
    }
  }

  std::cout << "Catalog processing complete. total=" << allItems.size()
            << " completed=" << completed
            << " failed=" << failed
            << " cancelled=" << cancelled << "\n";
  std::cout << "Deliverables JSON: " << jsonPath.string() << "\n";
  std::cout << "Deliverables CSV: " << csvPath.string() << "\n";
  std::cout << "Checkpoint: " << checkpointPath.string() << "\n";

  return failed == 0 && cancelled == 0 ? 0 : 1;
}

int commandSessionDiff(const std::vector<std::string>& args) {
  const auto baseArg = argValue(args, "--base");
  const auto headArg = argValue(args, "--head");
  if (!baseArg.has_value() || !headArg.has_value()) {
    std::cerr << "session-diff requires --base <session.json> --head <session.json>\n";
    return 2;
  }

  const auto baseJson = loadJsonFile(*baseArg);
  const auto headJson = loadJsonFile(*headArg);
  if (!baseJson.has_value() || !headJson.has_value()) {
    std::cerr << "Failed to load session JSON for diff.\n";
    return 1;
  }

  const auto patch = nlohmann::json::diff(*baseJson, *headJson);
  std::map<std::string, int> opCounts;
  for (const auto& op : patch) {
    if (op.is_object()) {
      opCounts[op.value("op", "unknown")] += 1;
    }
  }

  if (const auto outPathArg = argValue(args, "--out"); outPathArg.has_value()) {
    writeJsonFile(*outPathArg, patch);
    std::cout << "Wrote JSON patch to " << *outPathArg << "\n";
  } else {
    std::cout << patch.dump(2) << "\n";
  }

  if (hasFlag(args, "--summary")) {
    std::cout << "Patch operations: total=" << patch.size();
    for (const auto& [op, count] : opCounts) {
      std::cout << " " << op << "=" << count;
    }
    std::cout << "\n";
  }

  return 0;
}

int commandSessionMerge(const std::vector<std::string>& args) {
  const auto baseArg = argValue(args, "--base");
  const auto leftArg = argValue(args, "--left");
  const auto rightArg = argValue(args, "--right");
  const auto outArg = argValue(args, "--out");
  if (!baseArg.has_value() || !leftArg.has_value() || !rightArg.has_value() || !outArg.has_value()) {
    std::cerr << "session-merge requires --base <session.json> --left <session.json> --right <session.json> --out <session.json>\n";
    return 2;
  }

  const auto baseJson = loadJsonFile(*baseArg);
  const auto leftJson = loadJsonFile(*leftArg);
  const auto rightJson = loadJsonFile(*rightArg);
  if (!baseJson.has_value() || !leftJson.has_value() || !rightJson.has_value()) {
    std::cerr << "Failed to load session JSON for merge.\n";
    return 1;
  }

  JsonMergeTelemetry telemetry;
  telemetry.preferRight = toLower(argValue(args, "--prefer").value_or("right")) != "left";

  const auto merged = mergeJsonNode(baseJson, leftJson, rightJson, "", &telemetry);
  if (!merged.has_value()) {
    std::cerr << "Merged session resolved to null unexpectedly.\n";
    return 1;
  }

  automix::domain::Session mergedSession;
  try {
    mergedSession = merged->get<automix::domain::Session>();
  } catch (const std::exception& error) {
    std::cerr << "Merged JSON is not a valid Session schema: " << error.what() << "\n";
    return 1;
  }

  automix::engine::SessionRepository repository;
  repository.save(*outArg, mergedSession);

  nlohmann::json report = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"base", *baseArg},
      {"left", *leftArg},
      {"right", *rightArg},
      {"out", *outArg},
      {"preferredSide", telemetry.preferRight ? "right" : "left"},
      {"conflictCount", telemetry.conflictCount},
      {"conflictPaths", telemetry.conflictPaths},
  };

  if (const auto reportArg = argValue(args, "--report"); reportArg.has_value()) {
    writeJsonFile(*reportArg, report);
    std::cout << "Merge report: " << *reportArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << report.dump(2) << "\n";
  } else {
    std::cout << "Merged session written to " << *outArg
              << " (conflicts resolved=" << telemetry.conflictCount
              << ", preferred=" << (telemetry.preferRight ? "right" : "left") << ")\n";
  }

  return 0;
}

int commandExternalLimiterCompat(const std::vector<std::string>& args) {
  const auto binaryArg = argValue(args, "--binary");
  if (!binaryArg.has_value()) {
    std::cerr << "external-limiter-compat requires --binary <path>\n";
    return 2;
  }

  const auto timeoutMs = parseIntArg(args, "--timeout-ms").value_or(5000);
  const auto requiredFeatures =
      argValue(args, "--required-features").has_value()
          ? splitCommaSeparated(argValue(args, "--required-features").value())
          : std::vector<std::string>{};

  const std::filesystem::path binaryPath(*binaryArg);
  const auto validation = automix::renderers::ExternalLimiterRenderer::validateBinary(binaryPath, timeoutMs);

  std::unordered_set<std::string> supported;
  for (const auto& feature : validation.supportedFeatures) {
    supported.insert(toLower(feature));
  }
  std::vector<std::string> missingRequired;
  for (const auto& required : requiredFeatures) {
    if (supported.find(toLower(required)) == supported.end()) {
      missingRequired.push_back(required);
    }
  }

  const bool strictFeatureCheck = !requiredFeatures.empty();
  const bool featureCompatible = missingRequired.empty();
  const bool compatible = validation.valid && (!strictFeatureCheck || featureCompatible);
  const std::string tier = !validation.valid ? "incompatible"
                                             : (featureCompatible ? "compatible" : "partial");

  nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"binary", binaryPath.string()},
      {"timeoutMs", timeoutMs},
      {"valid", validation.valid},
      {"tier", tier},
      {"version", validation.version},
      {"errorCode", validation.errorCode},
      {"diagnostics", validation.diagnostics},
      {"supportedFeatures", validation.supportedFeatures},
      {"requiredFeatures", requiredFeatures},
      {"missingRequiredFeatures", missingRequired},
  };

  if (const auto outArg = argValue(args, "--out"); outArg.has_value()) {
    writeJsonFile(*outArg, payload);
    std::cout << "Compatibility report: " << *outArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "External limiter compatibility\n";
    std::cout << "  Binary: " << binaryPath.string() << "\n";
    std::cout << "  Tier: " << tier << "\n";
    std::cout << "  Valid: " << (validation.valid ? "yes" : "no") << "\n";
    std::cout << "  Version: " << (validation.version.empty() ? "(none)" : validation.version) << "\n";
    if (!missingRequired.empty()) {
      std::cout << "  Missing required features:\n";
      for (const auto& feature : missingRequired) {
        std::cout << "    - " << feature << "\n";
      }
    }
    std::cout << "  Diagnostics: " << validation.diagnostics << "\n";
  }

  return compatible ? 0 : 1;
}

int commandGoldenEval(const std::vector<std::string>& args) {
  const auto baselineArg = argValue(args, "--baseline");
  const auto baselinePath = baselineArg.has_value()
                                ? std::filesystem::path(*baselineArg)
                                : findRepoPath("tests/regression/baselines.json")
                                      .value_or(std::filesystem::path("tests/regression/baselines.json"));
  const auto workDir = std::filesystem::path(argValue(args, "--work-dir")
                                                 .value_or((std::filesystem::temp_directory_path() / "automix_golden_eval").string()));

  const auto result = automix::regression::runRegressionSuite(baselinePath, workDir);
  nlohmann::json rendered = nlohmann::json::array();
  for (const auto& metrics : result.rendered) {
    rendered.push_back({
        {"fixtureName", metrics.fixtureName},
        {"pipelineName", metrics.pipelineName},
        {"integratedLufs", metrics.metrics.integratedLufs},
        {"truePeakDbtp", metrics.metrics.truePeakDbtp},
        {"monoCorrelation", metrics.metrics.monoCorrelation},
        {"spectrumLow", metrics.metrics.spectrumLow},
        {"spectrumMid", metrics.metrics.spectrumMid},
        {"spectrumHigh", metrics.metrics.spectrumHigh},
        {"stereoCorrelation", metrics.metrics.stereoCorrelation},
    });
  }

  nlohmann::json failures = nlohmann::json::array();
  for (const auto& failure : result.failures) {
    failures.push_back({
        {"fixtureName", failure.fixtureName},
        {"pipelineName", failure.pipelineName},
        {"metricName", failure.metricName},
        {"expected", failure.expected},
        {"actual", failure.actual},
        {"tolerance", failure.tolerance},
    });
  }

  const nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"baselinePath", baselinePath.string()},
      {"workDir", workDir.string()},
      {"success", result.success},
      {"rendered", rendered},
      {"failures", failures},
  };

  const auto outPath =
      std::filesystem::path(argValue(args, "--out").value_or((workDir / "golden_eval_report.json").string()));
  writeJsonFile(outPath, payload);

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Golden corpus evaluation\n";
    std::cout << "  Baseline: " << baselinePath.string() << "\n";
    std::cout << "  Rendered: " << result.rendered.size() << "\n";
    std::cout << "  Failures: " << result.failures.size() << "\n";
    std::cout << "  Report: " << outPath.string() << "\n";
  }

  return result.success ? 0 : 1;
}

int commandPlanDiff(const std::vector<std::string>& args) {
  const auto sessionArg = argValue(args, "--session");
  if (!sessionArg.has_value()) {
    std::cerr << "plan-diff requires --session <session.json>\n";
    return 2;
  }

  const auto mixModelArg = argValue(args, "--mix-model");
  const auto masterModelArg = argValue(args, "--master-model");
  const auto outPath =
      std::filesystem::path(argValue(args, "--out")
                                .value_or((std::filesystem::path(*sessionArg).replace_extension(".plan_diff.json")).string()));

  automix::engine::SessionRepository repository;
  auto session = repository.load(*sessionArg);

  automix::analysis::StemAnalyzer analyzer;
  const auto analysisEntries = analyzer.analyzeSession(session);

  const double dryWet = session.mixPlan.has_value() ? session.mixPlan->dryWet : 1.0;
  automix::automix::HeuristicAutoMixStrategy heuristicMixStrategy;
  const auto heuristicMixPlan = heuristicMixStrategy.buildPlan(session, analysisEntries, dryWet);

  std::vector<std::string> inferenceNotes;
  auto mixInference = buildPlanDiffInference(mixModelArg, "mix", &inferenceNotes);
  automix::ai::AutoMixStrategyAI mixStrategyAi;
  const auto aiMixPlan = mixStrategyAi.buildPlan(session, analysisEntries, heuristicMixPlan, mixInference.get());

  auto renderSettings = session.renderSettings;
  renderSettings.outputSampleRate = renderSettings.outputSampleRate > 0 ? renderSettings.outputSampleRate : 44100;
  renderSettings.blockSize = renderSettings.blockSize > 0 ? renderSettings.blockSize : 1024;
  renderSettings.outputBitDepth = renderSettings.outputBitDepth > 0 ? renderSettings.outputBitDepth : 24;

  automix::engine::OfflineRenderPipeline pipeline;
  auto heuristicSession = session;
  heuristicSession.mixPlan = heuristicMixPlan;
  const auto heuristicRaw = pipeline.renderRawMix(heuristicSession, renderSettings, {}, nullptr);
  if (heuristicRaw.cancelled) {
    std::cerr << "plan-diff aborted: heuristic render cancelled.\n";
    return 1;
  }

  auto aiSession = session;
  aiSession.mixPlan = aiMixPlan;
  const auto aiRaw = pipeline.renderRawMix(aiSession, renderSettings, {}, nullptr);
  if (aiRaw.cancelled) {
    std::cerr << "plan-diff aborted: model render cancelled.\n";
    return 1;
  }

  automix::automaster::HeuristicAutoMasterStrategy heuristicMasterStrategy;
  const auto masterPreset = session.masterPlan.has_value() ? session.masterPlan->preset
                                                           : automix::domain::MasterPreset::DefaultStreaming;
  const auto heuristicMasterPlan = heuristicMasterStrategy.buildPlan(masterPreset, heuristicRaw.mixBuffer);
  const auto heuristicMixMetrics = analyzer.analyzeBuffer(heuristicRaw.mixBuffer);

  auto masterInference = buildPlanDiffInference(masterModelArg, "master", &inferenceNotes);
  automix::ai::AutoMasterStrategyAI masterStrategyAi;
  const auto aiMasterPlan = masterStrategyAi.buildPlan(heuristicMixMetrics, heuristicMasterPlan, masterInference.get());

  nlohmann::json stemDeltas = nlohmann::json::array();
  std::map<std::string, automix::domain::StemMixDecision> heuristicByStem;
  std::map<std::string, automix::domain::StemMixDecision> aiByStem;
  for (const auto& decision : heuristicMixPlan.stemDecisions) {
    heuristicByStem[decision.stemId] = decision;
  }
  for (const auto& decision : aiMixPlan.stemDecisions) {
    aiByStem[decision.stemId] = decision;
  }

  std::set<std::string> stemIds;
  for (const auto& [id, _] : heuristicByStem) {
    stemIds.insert(id);
  }
  for (const auto& [id, _] : aiByStem) {
    stemIds.insert(id);
  }
  for (const auto& stemId : stemIds) {
    const auto heurIt = heuristicByStem.find(stemId);
    const auto aiIt = aiByStem.find(stemId);
    const double heurGain = heurIt == heuristicByStem.end() ? 0.0 : heurIt->second.gainDb;
    const double aiGain = aiIt == aiByStem.end() ? 0.0 : aiIt->second.gainDb;
    const double heurPan = heurIt == heuristicByStem.end() ? 0.0 : heurIt->second.pan;
    const double aiPan = aiIt == aiByStem.end() ? 0.0 : aiIt->second.pan;
    const double heurHighPass = heurIt == heuristicByStem.end() ? 0.0 : heurIt->second.highPassHz;
    const double aiHighPass = aiIt == aiByStem.end() ? 0.0 : aiIt->second.highPassHz;

    stemDeltas.push_back({
        {"stemId", stemId},
        {"heuristicGainDb", heurGain},
        {"modelGainDb", aiGain},
        {"deltaGainDb", aiGain - heurGain},
        {"heuristicPan", heurPan},
        {"modelPan", aiPan},
        {"deltaPan", aiPan - heurPan},
        {"heuristicHighPassHz", heurHighPass},
        {"modelHighPassHz", aiHighPass},
        {"deltaHighPassHz", aiHighPass - heurHighPass},
    });
  }

  const double heuristicIntegrated = heuristicMasterStrategy.measureIntegratedLufs(heuristicRaw.mixBuffer);
  const double aiIntegrated = heuristicMasterStrategy.measureIntegratedLufs(aiRaw.mixBuffer);

  const nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"sessionPath", *sessionArg},
      {"inferenceNotes", inferenceNotes},
      {"mixPlan",
       {
           {"heuristic", automix::domain::Json(heuristicMixPlan)},
           {"model", automix::domain::Json(aiMixPlan)},
           {"patch", nlohmann::json::diff(automix::domain::Json(heuristicMixPlan), automix::domain::Json(aiMixPlan))},
           {"stemDeltas", stemDeltas},
       }},
      {"masterPlan",
       {
           {"heuristic", automix::domain::Json(heuristicMasterPlan)},
           {"model", automix::domain::Json(aiMasterPlan)},
           {"patch", nlohmann::json::diff(automix::domain::Json(heuristicMasterPlan), automix::domain::Json(aiMasterPlan))},
           {"deltaTargetLufs", aiMasterPlan.targetLufs - heuristicMasterPlan.targetLufs},
           {"deltaPreGainDb", aiMasterPlan.preGainDb - heuristicMasterPlan.preGainDb},
           {"deltaLimiterCeilingDb", aiMasterPlan.limiterCeilingDb - heuristicMasterPlan.limiterCeilingDb},
           {"deltaGlueRatio", aiMasterPlan.glueRatio - heuristicMasterPlan.glueRatio},
       }},
      {"mixBusComparison",
       {
           {"heuristicIntegratedLufs", heuristicIntegrated},
           {"modelIntegratedLufs", aiIntegrated},
           {"deltaIntegratedLufs", aiIntegrated - heuristicIntegrated},
       }},
  };

  writeJsonFile(outPath, payload);
  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Generated heuristic-vs-model plan diff: " << outPath.string() << "\n";
    std::cout << "Mix patch ops: "
              << nlohmann::json::diff(automix::domain::Json(heuristicMixPlan), automix::domain::Json(aiMixPlan)).size()
              << ", master patch ops: "
              << nlohmann::json::diff(automix::domain::Json(heuristicMasterPlan), automix::domain::Json(aiMasterPlan)).size()
              << "\n";
  }

  return 0;
}

void printUsage() {
  std::cout << "Usage:\n";
  std::cout << "  automix_dev_tools export-features --session <session.json> --out <features.jsonl> [--manifest <manifest.json>] [--dataset-id <id>] [--source-tag <tag>] [--lineage-parents <id,id,...>]\n";
  std::cout << "  automix_dev_tools export-segments --session <session.json> --out-dir <dir> [--segment-seconds <sec>]\n";
  std::cout << "  automix_dev_tools validate-modelpack --pack <modelpack_dir>\n";
  std::cout << "  automix_dev_tools validate-external-limiter --binary <path> [--json]\n";
  std::cout << "  automix_dev_tools stem-health --session <session.json> [--out <path>] [--json]\n";
  std::cout << "  automix_dev_tools compare-renders --session <session.json> [--renderers <id,id,...>] [--out-dir <dir>] [--format <fmt>] [--external-binary <path>] [--json]\n";
  std::cout << "  automix_dev_tools catalog-process --input <folder> --output <folder> [--checkpoint <path>] [--resume] [--renderer <id>] [--format <fmt>] [--analysis-threads <n>] [--render-parallelism <n>] [--csv <path>] [--json <path>]\n";
  std::cout << "  automix_dev_tools session-diff --base <session.json> --head <session.json> [--out <patch.json>] [--summary]\n";
  std::cout << "  automix_dev_tools session-merge --base <session.json> --left <session.json> --right <session.json> --out <session.json> [--prefer <left|right>] [--report <report.json>] [--json]\n";
  std::cout << "  automix_dev_tools external-limiter-compat --binary <path> [--timeout-ms <ms>] [--required-features <f1,f2>] [--out <report.json>] [--json]\n";
  std::cout << "  automix_dev_tools golden-eval [--baseline <baselines.json>] [--work-dir <dir>] [--out <report.json>] [--json]\n";
  std::cout << "  automix_dev_tools plan-diff --session <session.json> [--mix-model <path>] [--master-model <path>] [--out <report.json>] [--json]\n";
  std::cout << "  automix_dev_tools list-supported-models\n";
  std::cout << "  automix_dev_tools install-supported-model --id <model_id> [--dest <assets/models>]\n";
  std::cout << "  automix_dev_tools list-supported-limiters\n";
  std::cout << "  automix_dev_tools install-supported-limiter --id <limiter_id> [--dest <assets/limiters>]\n";
  std::cout << "  automix_dev_tools install-lame-fallback [--force] [--json]\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 1; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }

    if (args.empty()) {
      printUsage();
      return 2;
    }

    const std::string command = args.front();
    if (command == "export-features") {
      return commandExportFeatures(args);
    }
    if (command == "export-segments") {
      return commandExportSegments(args);
    }
    if (command == "validate-modelpack") {
      return commandValidateModelPack(args);
    }
    if (command == "validate-external-limiter") {
      return commandValidateExternalLimiter(args);
    }
    if (command == "stem-health") {
      return commandStemHealth(args);
    }
    if (command == "compare-renders") {
      return commandCompareRenders(args);
    }
    if (command == "catalog-process") {
      return commandCatalogProcess(args);
    }
    if (command == "session-diff") {
      return commandSessionDiff(args);
    }
    if (command == "session-merge") {
      return commandSessionMerge(args);
    }
    if (command == "external-limiter-compat") {
      return commandExternalLimiterCompat(args);
    }
    if (command == "golden-eval") {
      return commandGoldenEval(args);
    }
    if (command == "plan-diff") {
      return commandPlanDiff(args);
    }
    if (command == "list-supported-models") {
      return commandListSupportedModels();
    }
    if (command == "install-supported-model") {
      return commandInstallSupportedModel(args);
    }
    if (command == "list-supported-limiters") {
      return commandListSupportedLimiters();
    }
    if (command == "install-supported-limiter") {
      return commandInstallSupportedLimiter(args);
    }
    if (command == "install-lame-fallback") {
      return commandInstallLameFallback(args);
    }

    printUsage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "Developer tool error: " << error.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Developer tool error: unknown exception\n";
    return 1;
  }
}
