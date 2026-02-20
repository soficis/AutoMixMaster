#include "commands/DevToolsUtils.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

#include "ai/OnnxModelInference.h"

namespace automix::devtools {

// --- Argument parsing ---

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

// --- String helpers ---

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

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

// --- Hash / timestamp ---

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

// --- Environment / path helpers ---

std::optional<std::string> readEnvironment(const std::string& key) {
  const char* value = std::getenv(key.c_str());
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  return std::string(value);
}

std::filesystem::path profileCatalogPath() {
  return std::filesystem::path("assets") / "profiles" / "project_profiles.json";
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

// --- File I/O ---

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

// --- Audio helpers ---

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

// --- Model helpers ---

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

// --- Supported packs / limiters ---

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

// --- Report metrics ---

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

// --- Batch status ---

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

// --- Deterministic inference ---

bool DeterministicPlanDiffInference::isAvailable() const { return true; }

bool DeterministicPlanDiffInference::loadModel(const std::filesystem::path&) {
  loaded_ = true;
  return true;
}

automix::ai::InferenceResult DeterministicPlanDiffInference::run(
    const automix::ai::InferenceRequest& request) const {
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

std::unique_ptr<automix::ai::IModelInference> buildPlanDiffInference(
    const std::optional<std::string>& modelPathArg,
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

// --- JSON merge internals ---

namespace {

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

    const auto mergedItem = automix::devtools::mergeJsonNode(baseItem,
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

} // namespace

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

// --- Project profile serialization ---

nlohmann::json projectProfileToJson(const automix::domain::ProjectProfile& profile) {
  return {
      {"id", profile.id},
      {"name", profile.name},
      {"platformPreset", profile.platformPreset},
      {"rendererName", profile.rendererName},
      {"outputFormat", profile.outputFormat},
      {"lossyBitrateKbps", profile.lossyBitrateKbps},
      {"mp3UseVbr", profile.mp3UseVbr},
      {"mp3VbrQuality", profile.mp3VbrQuality},
      {"gpuProvider", profile.gpuProvider},
      {"roleModelPackId", profile.roleModelPackId},
      {"mixModelPackId", profile.mixModelPackId},
      {"masterModelPackId", profile.masterModelPackId},
      {"safetyPolicyId", profile.safetyPolicyId},
      {"preferredStemCount", profile.preferredStemCount},
      {"metadataPolicy", profile.metadataPolicy},
      {"metadataTemplate", profile.metadataTemplate},
      {"pinnedRendererIds", profile.pinnedRendererIds},
  };
}

std::optional<automix::domain::ProjectProfile> projectProfileFromJson(const nlohmann::json& json) {
  if (!json.is_object()) {
    return std::nullopt;
  }

  automix::domain::ProjectProfile profile;
  profile.id = json.value("id", "");
  profile.name = json.value("name", profile.id);
  profile.platformPreset = json.value("platformPreset", "spotify");
  profile.rendererName = json.value("rendererName", "BuiltIn");
  profile.outputFormat = json.value("outputFormat", "wav");
  profile.lossyBitrateKbps = std::clamp(json.value("lossyBitrateKbps", 320), 64, 320);
  profile.mp3UseVbr = json.value("mp3UseVbr", false);
  profile.mp3VbrQuality = std::clamp(json.value("mp3VbrQuality", 4), 0, 9);
  profile.gpuProvider = json.value("gpuProvider", "auto");
  profile.roleModelPackId = json.value("roleModelPackId", "none");
  profile.mixModelPackId = json.value("mixModelPackId", "none");
  profile.masterModelPackId = json.value("masterModelPackId", "none");
  profile.safetyPolicyId = json.value("safetyPolicyId", "balanced");
  profile.preferredStemCount = std::clamp(json.value("preferredStemCount", 4), automix::domain::kMinPreferredStemCount, automix::domain::kMaxPreferredStemCount);
  profile.metadataPolicy = json.value("metadataPolicy", "copy_common");
  if (profile.metadataPolicy != "copy_all" &&
      profile.metadataPolicy != "copy_common" &&
      profile.metadataPolicy != "copy_common_only" &&
      profile.metadataPolicy != "strip" &&
      profile.metadataPolicy != "override_template") {
    profile.metadataPolicy = "copy_common";
  }
  if (json.contains("metadataTemplate") && json.at("metadataTemplate").is_object()) {
    profile.metadataTemplate = json.at("metadataTemplate").get<std::map<std::string, std::string>>();
  }
  if (json.contains("pinnedRendererIds") && json.at("pinnedRendererIds").is_array()) {
    profile.pinnedRendererIds = json.at("pinnedRendererIds").get<std::vector<std::string>>();
  }

  if (profile.id.empty() || profile.name.empty()) {
    return std::nullopt;
  }
  return profile;
}

} // namespace automix::devtools
