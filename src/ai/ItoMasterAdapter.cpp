#include "ai/ItoMasterAdapter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "util/StringUtils.h"

namespace automix::ai {
namespace {

std::string trimString(const std::string& value) {
  return util::trim(value);
}

ItoTensorShape parseTensorDims(const std::string& dimsText) {
  ItoTensorShape shape;
  std::stringstream stream(dimsText);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = trimString(token);
    if (token.empty() || token == "N") {
      shape.dims.push_back(-1);
      continue;
    }
    try {
      shape.dims.push_back(std::stoll(token));
    } catch (...) {
      shape.dims.push_back(-1);
    }
  }
  return shape;
}

std::vector<ItoTensorShape> parseTensorSpecs(const std::string& text) {
  std::vector<ItoTensorShape> shapes;
  std::string remaining = text;
  while (true) {
    const auto open = remaining.find('[');
    if (open == std::string::npos) {
      break;
    }
    const auto close = remaining.find(']', open);
    if (close == std::string::npos) {
      break;
    }
    shapes.push_back(parseTensorDims(remaining.substr(open + 1, close - open - 1)));
    remaining = remaining.substr(close + 1);
  }
  return shapes;
}

bool parseGraphContract(const nlohmann::json& graphJson, ItoGraphContract& contract) {
  if (!graphJson.is_object()) {
    return false;
  }
  contract.file = graphJson.value("file", "");
  if (contract.file.empty()) {
    return false;
  }
  contract.inputShapes = parseTensorSpecs(graphJson.value("in", ""));
  contract.outputShapes = parseTensorSpecs(graphJson.value("out", ""));
  return !contract.inputShapes.empty() && !contract.outputShapes.empty();
}

// Fixed dims must match; dynamic dims (-1) are allowed to be anything.
bool shapeMatches(const ItoTensorShape& expected, const std::vector<int64_t>& actual) {
  if (expected.dims.size() != actual.size()) {
    return false;
  }
  for (size_t i = 0; i < expected.dims.size(); ++i) {
    if (expected.dims[i] >= 0 && expected.dims[i] != actual[i]) {
      return false;
    }
  }
  return true;
}

std::vector<double> orderedOutputs(const InferenceResult& result, const size_t maxCount) {
  std::vector<double> ordered;
  ordered.reserve(maxCount);
  for (size_t i = 0; i < maxCount; ++i) {
    const auto it = result.outputs.find("output_" + std::to_string(i));
    if (it == result.outputs.end()) {
      break;
    }
    ordered.push_back(it->second);
  }
  return ordered;
}

} // namespace

std::optional<ItoMasterConfig> ItoMasterAdapter::loadConfig(const std::filesystem::path& configPath) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(configPath, error) || error) {
    return std::nullopt;
  }

  std::ifstream in(configPath);
  if (!in.is_open()) {
    return std::nullopt;
  }

  nlohmann::json json;
  try {
    in >> json;
  } catch (...) {
    return std::nullopt;
  }
  if (!json.is_object()) {
    return std::nullopt;
  }

  ItoMasterConfig config;
  config.model = json.value("model", "");
  config.sampleRate = json.value("sample_rate", 44100.0);
  config.normalized = json.value("normalized", true);
  config.numParams = json.value("num_params", 0);

  if (json.contains("fx_order") && json.at("fx_order").is_array()) {
    config.fxOrder = json.at("fx_order").get<std::vector<std::string>>();
  }

  if (json.contains("params") && json.at("params").is_array()) {
    for (const auto& item : json.at("params")) {
      if (!item.is_object()) {
        return std::nullopt;
      }
      ItoParamSpec spec;
      spec.fx = item.value("fx", "");
      spec.name = item.value("name", "");
      if (spec.name.empty()) {
        return std::nullopt;
      }
      spec.min = item.value("min", 0.0);
      spec.max = item.value("max", 1.0);
      config.params.push_back(spec);
    }
  }

  if (json.contains("graphs") && json.at("graphs").is_object()) {
    const auto& graphs = json.at("graphs");
    if (graphs.contains("fxencoder")) {
      if (!parseGraphContract(graphs.at("fxencoder"), config.encoder)) {
        return std::nullopt;
      }
    }
    if (graphs.contains("predictor")) {
      if (!parseGraphContract(graphs.at("predictor"), config.predictor)) {
        return std::nullopt;
      }
    }
  }

  if (!validateConfig(config)) {
    return std::nullopt;
  }
  return config;
}

bool ItoMasterAdapter::validateConfig(const ItoMasterConfig& config) {
  if (config.numParams != kItoMasterParamCount) {
    return false;
  }
  if (config.params.size() != static_cast<size_t>(config.numParams)) {
    return false;
  }
  for (const auto& spec : config.params) {
    if (spec.fx.empty() || spec.name.empty() || spec.max < spec.min) {
      return false;
    }
  }
  if (config.encoder.file != kItoMasterEncoderFile || config.predictor.file != kItoMasterPredictorFile) {
    return false;
  }
  if (config.encoder.inputShapes.empty() || config.encoder.outputShapes.empty()) {
    return false;
  }
  if (config.predictor.inputShapes.size() < 2 || config.predictor.outputShapes.empty()) {
    return false;
  }
  return true;
}

bool ItoMasterAdapter::validateTensorContract(const ItoMasterConfig& config,
                                              const int numChannels,
                                              const int numSamples) {
  if (!validateConfig(config)) {
    return false;
  }
  if (numChannels != 2 || numSamples <= 0) {
    return false;
  }

  // fxencoder: ref_audio[1,2,N] -> embedding[1,2048]
  const auto& encoderInput = config.encoder.inputShapes[0];
  const auto& encoderOutput = config.encoder.outputShapes[0];
  if (!shapeMatches(encoderInput, {1, numChannels, -1})) {
    return false;
  }
  if (!shapeMatches(encoderOutput, {1, kItoMasterEmbeddingSize})) {
    return false;
  }

  // mastering_tcn: in_audio[1,2,N] + embedding[1,2048] -> params[1,46]
  const auto& predictorAudio = config.predictor.inputShapes[0];
  const auto& predictorEmbedding = config.predictor.inputShapes[1];
  const auto& predictorOutput = config.predictor.outputShapes[0];
  if (!shapeMatches(predictorAudio, {1, numChannels, -1})) {
    return false;
  }
  if (!shapeMatches(predictorEmbedding, {1, kItoMasterEmbeddingSize})) {
    return false;
  }
  if (!shapeMatches(predictorOutput, {1, kItoMasterParamCount})) {
    return false;
  }

  return true;
}

std::vector<double> ItoMasterAdapter::denormalize(const ItoMasterConfig& config,
                                                  const std::vector<double>& normalized) {
  std::vector<double> physical;
  physical.reserve(config.params.size());
  for (size_t i = 0; i < config.params.size(); ++i) {
    const auto& spec = config.params[i];
    const double norm = i < normalized.size() ? std::clamp(normalized[i], 0.0, 1.0) : 0.5;
    physical.push_back(norm * (spec.max - spec.min) + spec.min);
  }
  return physical;
}

std::vector<double> ItoMasterAdapter::normalize(const ItoMasterConfig& config,
                                                const std::vector<double>& physical) {
  std::vector<double> normalized;
  normalized.reserve(config.params.size());
  for (size_t i = 0; i < config.params.size(); ++i) {
    const auto& spec = config.params[i];
    const double range = (spec.max - spec.min) > 1.0e-12 ? (spec.max - spec.min) : 1.0;
    const double value = i < physical.size() ? std::clamp(physical[i], spec.min, spec.max) : spec.min;
    normalized.push_back(std::clamp((value - spec.min) / range, 0.0, 1.0));
  }
  return normalized;
}

dsp::ItoMasterChainSettings ItoMasterAdapter::toChainSettings(const ItoMasterConfig& config,
                                                              const std::vector<double>& physical) {
  dsp::ItoMasterChainSettings settings;
  if (physical.size() != static_cast<size_t>(config.numParams)) {
    return settings;
  }

  const auto value = [&](const std::string& fx, const std::string& name) -> double {
    for (size_t i = 0; i < config.params.size(); ++i) {
      if (config.params[i].fx == fx && config.params[i].name == name) {
        return physical[i];
      }
    }
    return 0.0;
  };

  // eq (18): low shelf + 4 peaking bands + high shelf
  settings.eq.lowShelf.gainDb = value("eq", "low_shelf_gain_db");
  settings.eq.lowShelf.freqHz = value("eq", "low_shelf_cutoff_freq");
  settings.eq.lowShelf.qFactor = value("eq", "low_shelf_q_factor");
  const char* bandNames[4] = {"band0", "band1", "band2", "band3"};
  for (int band = 0; band < 4; ++band) {
    const std::string prefix = bandNames[band];
    settings.eq.bands[static_cast<size_t>(band)].gainDb = value("eq", prefix + "_gain_db");
    settings.eq.bands[static_cast<size_t>(band)].freqHz = value("eq", prefix + "_cutoff_freq");
    settings.eq.bands[static_cast<size_t>(band)].qFactor = value("eq", prefix + "_q_factor");
  }
  settings.eq.highShelf.gainDb = value("eq", "high_shelf_gain_db");
  settings.eq.highShelf.freqHz = value("eq", "high_shelf_cutoff_freq");
  settings.eq.highShelf.qFactor = value("eq", "high_shelf_q_factor");

  // distortion (2)
  settings.distortion.driveDb = value("distortion", "drive_db");
  settings.distortion.parallelWeight = value("distortion", "parallel_weight_factor");

  // multiband_comp (21): crossovers + parallel blend + 3 bands x 6 dynamics
  settings.multiband.lowCrossoverHz = value("multiband_comp", "low_cutoff");
  settings.multiband.highCrossoverHz = value("multiband_comp", "high_cutoff");
  settings.multiband.parallelWeight = value("multiband_comp", "parallel_weight_factor");
  const char* bandPrefixes[3] = {"low_shelf", "mid_band", "high_shelf"};
  for (int band = 0; band < 3; ++band) {
    const std::string prefix = bandPrefixes[band];
    auto& bandSettings = settings.multiband.bands[static_cast<size_t>(band)];
    bandSettings.compThresholdDb = value("multiband_comp", prefix + "_comp_thresh");
    bandSettings.compRatio = value("multiband_comp", prefix + "_comp_ratio");
    bandSettings.expThresholdDb = value("multiband_comp", prefix + "_exp_thresh");
    bandSettings.expRatio = value("multiband_comp", prefix + "_exp_ratio");
    bandSettings.attackMs = value("multiband_comp", prefix + "_at");
    bandSettings.releaseMs = value("multiband_comp", prefix + "_rt");
  }

  // gain (1), imager (1), limiter (3)
  settings.gain.gainDb = value("gain", "gain_db");
  settings.imager.width = value("imager", "width");
  settings.limiter.thresholdDb = value("limiter", "threshold");
  settings.limiter.attackMs = value("limiter", "at");
  settings.limiter.releaseMs = value("limiter", "rt");

  return settings;
}

dsp::ItoMasterChainSettings ItoMasterAdapter::apply(const ItoMasterConfig& config,
                                                    const std::vector<double>& normalized) {
  return toChainSettings(config, denormalize(config, normalized));
}

std::optional<std::vector<double>> ItoMasterAdapter::extractParams(
    const std::vector<double>& flattenedOutputs) {
  if (flattenedOutputs.size() < static_cast<size_t>(kItoMasterParamCount)) {
    return std::nullopt;
  }
  std::vector<double> params;
  params.reserve(static_cast<size_t>(kItoMasterParamCount));
  for (int i = 0; i < kItoMasterParamCount; ++i) {
    params.push_back(std::clamp(flattenedOutputs[static_cast<size_t>(i)], 0.0, 1.0));
  }
  return params;
}

std::optional<std::vector<double>> ItoMasterAdapter::extractEmbedding(
    const std::vector<double>& flattenedOutputs) {
  if (flattenedOutputs.size() < static_cast<size_t>(kItoMasterEmbeddingSize)) {
    return std::nullopt;
  }
  std::vector<double> embedding;
  embedding.reserve(static_cast<size_t>(kItoMasterEmbeddingSize));
  for (int i = 0; i < kItoMasterEmbeddingSize; ++i) {
    embedding.push_back(flattenedOutputs[static_cast<size_t>(i)]);
  }
  return embedding;
}

bool ItoMasterModelRunner::load(const std::filesystem::path& packDir,
                                const ItoMasterConfig& config,
                                std::string* errorOut) {
  packDir_ = packDir;
  config_ = config;
  loaded_ = false;

  if (!ItoMasterAdapter::validateConfig(config)) {
    if (errorOut != nullptr) {
      *errorOut = "ITO-Master config contract is invalid.";
    }
    return false;
  }

  const auto encoderPath = packDir / kItoMasterEncoderFile;
  const auto predictorPath = packDir / kItoMasterPredictorFile;
  std::error_code error;
  if (!std::filesystem::is_regular_file(encoderPath, error) || error) {
    if (errorOut != nullptr) {
      *errorOut = "missing " + encoderPath.string();
    }
    return false;
  }
  if (!std::filesystem::is_regular_file(predictorPath, error) || error) {
    if (errorOut != nullptr) {
      *errorOut = "missing " + predictorPath.string();
    }
    return false;
  }

  encoder_.setWarmupEnabled(false);
  predictor_.setWarmupEnabled(false);

  if (!encoder_.loadModel(encoderPath)) {
    if (errorOut != nullptr) {
      *errorOut = "fxencoder load failed: " + encoder_.backendDiagnostics();
    }
    return false;
  }
  if (!predictor_.loadModel(predictorPath)) {
    if (errorOut != nullptr) {
      *errorOut = "mastering_tcn load failed: " + predictor_.backendDiagnostics();
    }
    return false;
  }

  loaded_ = true;
  return true;
}

bool ItoMasterModelRunner::usesNativeSession() const {
  return loaded_ && encoder_.usingNativeSession() && predictor_.usingNativeSession();
}

std::optional<std::vector<double>> ItoMasterModelRunner::predict(
    const engine::AudioBuffer& referenceAudio) const {
  if (!loaded_) {
    return std::nullopt;
  }
  const int channels = referenceAudio.getNumChannels();
  const int totalSamples = referenceAudio.getNumSamples();
  if (channels != 2 || totalSamples <= 0) {
    return std::nullopt;
  }

  // Reference window: the first N samples (10 s default), stereo-flattened to
  // 2*N features matching the [1,2,N] static contract.
  const int chunk = std::min(totalSamples, kItoMasterDefaultChunkSamples);
  std::vector<double> audioFeatures(static_cast<size_t>(2 * chunk), 0.0);
  for (int i = 0; i < chunk; ++i) {
    for (int c = 0; c < 2; ++c) {
      audioFeatures[static_cast<size_t>(c * chunk + i)] = referenceAudio.getSample(c, i);
    }
  }

  // fxencoder: audio -> style embedding [1,2048]
  const InferenceRequest encodeRequest{
      .task = "ito_fxencoder",
      .features = audioFeatures,
  };
  const auto encodeResult = encoder_.run(encodeRequest);
  const auto embedding = ItoMasterAdapter::extractEmbedding(orderedOutputs(encodeResult, 2048));

  // mastering_tcn: audio + embedding -> 46 normalized params [1,46].
  std::vector<double> predictorFeatures = audioFeatures;
  if (embedding.has_value()) {
    predictorFeatures.insert(predictorFeatures.end(), embedding->begin(), embedding->end());
  }
  const InferenceRequest predictRequest{
      .task = "ito_predictor",
      .features = predictorFeatures,
  };
  const auto predictResult = predictor_.run(predictRequest);

  // Only a native path that actually emitted the full 46-param tensor counts as
  // a usable prediction; the deterministic fallback returns nullopt so the
  // route never fabricates mastering parameters (no false pass).
  return ItoMasterAdapter::extractParams(orderedOutputs(predictResult, 46));
}

} // namespace automix::ai
