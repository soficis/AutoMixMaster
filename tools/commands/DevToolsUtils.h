#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ai/IModelInference.h"
#include "domain/BatchTypes.h"
#include "domain/ProjectProfile.h"
#include "engine/AudioBuffer.h"

namespace automix::devtools {

// --- Argument parsing ---

std::optional<std::string> argValue(const std::vector<std::string>& args, const std::string& key);
bool hasFlag(const std::vector<std::string>& args, const std::string& key);
std::optional<int> parseIntArg(const std::vector<std::string>& args, const std::string& key);
std::optional<double> parseDoubleArg(const std::vector<std::string>& args, const std::string& key);

// --- String helpers ---

std::string toLower(std::string value);
std::string sanitizeFileName(std::string value);
std::vector<std::string> splitCommaSeparated(const std::string& value);
std::string csvEscape(const std::string& value);

// --- Hash / timestamp ---

uint64_t fnv1a64(const std::string& input);
std::string toHex(uint64_t value);
std::string iso8601NowUtc();

// --- Environment / path helpers ---

std::optional<std::string> readEnvironment(const std::string& key);
std::filesystem::path profileCatalogPath();
std::string extensionForFormat(const std::string& format);
std::optional<std::filesystem::path> findRepoPath(const std::filesystem::path& relativePath);
void copyDirectory(const std::filesystem::path& source, const std::filesystem::path& destination);

// --- File I/O ---

std::optional<nlohmann::json> loadJsonFile(const std::filesystem::path& path);
std::optional<std::string> readTextFile(const std::filesystem::path& path);
void writeJsonFile(const std::filesystem::path& path, const nlohmann::json& payload);

// --- Audio helpers ---

automix::engine::AudioBuffer sliceBuffer(const automix::engine::AudioBuffer& input, int maxSamples);

// --- Model helpers ---

std::vector<double> deterministicFeatures(size_t count);
std::string taskFromModelType(const std::string& type);

// --- Supported packs / limiters ---

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

const std::vector<SupportedModelPack>& supportedModelPacks();
const std::vector<SupportedLimiter>& supportedLimiters();

// --- Report metrics ---

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

ReportMetrics readReportMetrics(const std::filesystem::path& reportPath);
double computeComparatorScore(const ReportMetrics& metrics);

// --- Batch status ---

automix::domain::BatchItemStatus batchStatusFromString(const std::string& value);

// --- Deterministic inference ---

class DeterministicPlanDiffInference final : public automix::ai::IModelInference {
 public:
  bool isAvailable() const override;
  bool loadModel(const std::filesystem::path& path) override;
  automix::ai::InferenceResult run(const automix::ai::InferenceRequest& request) const override;

 private:
  bool loaded_ = true;
};

std::unique_ptr<automix::ai::IModelInference> buildPlanDiffInference(
    const std::optional<std::string>& modelPathArg,
    const std::string& taskLabel,
    std::vector<std::string>* notes);

// --- JSON merge ---

struct JsonMergeTelemetry {
  bool preferRight = true;
  size_t conflictCount = 0;
  std::vector<std::string> conflictPaths;
};

std::optional<nlohmann::json> mergeJsonNode(const std::optional<nlohmann::json>& base,
                                            const std::optional<nlohmann::json>& left,
                                            const std::optional<nlohmann::json>& right,
                                            const std::string& path,
                                            JsonMergeTelemetry* telemetry);

// --- Project profile serialization ---

nlohmann::json projectProfileToJson(const automix::domain::ProjectProfile& profile);
std::optional<automix::domain::ProjectProfile> projectProfileFromJson(const nlohmann::json& json);

} // namespace automix::devtools
