#include "ai/StemRoleClassifierAI.h"

#include <algorithm>
#include <array>
#include <cctype>

namespace automix::ai {
namespace {

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

RolePrediction predictFromName(const std::string& stemName) {
  const auto name = toLower(stemName);
  if (name.find("vocal") != std::string::npos || name.find("vox") != std::string::npos) {
    return {.role = domain::StemRole::Vocals, .confidence = 0.9};
  }
  if (name.find("kick") != std::string::npos || name.find("snare") != std::string::npos ||
      name.find("drum") != std::string::npos) {
    return {.role = domain::StemRole::Drums, .confidence = 0.8};
  }
  if (name.find("bass") != std::string::npos || name.find("sub") != std::string::npos) {
    return {.role = domain::StemRole::Bass, .confidence = 0.85};
  }
  if (name.find("fx") != std::string::npos || name.find("sfx") != std::string::npos) {
    return {.role = domain::StemRole::Fx, .confidence = 0.8};
  }
  return {};
}

RolePrediction predictFromSpectrum(const analysis::AnalysisResult& metrics) {
  if (metrics.lowEnergy > 0.66) {
    return {.role = domain::StemRole::Bass, .confidence = 0.65};
  }
  if (metrics.highEnergy > 0.63) {
    return {.role = domain::StemRole::Fx, .confidence = 0.62};
  }
  if (metrics.midEnergy > 0.58) {
    return {.role = domain::StemRole::Music, .confidence = 0.58};
  }
  return {.role = domain::StemRole::Unknown, .confidence = 0.5};
}

} // namespace

StemRoleClassifierAI::StemRoleClassifierAI(IModelInference* inference) : inference_(inference) {}

RolePrediction StemRoleClassifierAI::predict(const std::string& stemName,
                                             const analysis::AnalysisResult& metrics) const {
  if (inference_ != nullptr && inference_->isAvailable()) {
    InferenceRequest request{
        .task = "role_classifier",
        .features = {metrics.rmsDb, metrics.lowEnergy, metrics.midEnergy, metrics.highEnergy, metrics.artifactRisk},
    };
    const auto result = inference_->run(request);
    if (result.usedModel && !result.outputs.empty()) {
      struct Candidate {
        const char* key;
        domain::StemRole role;
      };
      constexpr std::array<Candidate, 4> candidates = {{
          {"prob_vocals", domain::StemRole::Vocals},
          {"prob_bass", domain::StemRole::Bass},
          {"prob_drums", domain::StemRole::Drums},
          {"prob_fx", domain::StemRole::Fx},
      }};

      RolePrediction best;
      for (const auto& candidate : candidates) {
        const auto it = result.outputs.find(candidate.key);
        if (it == result.outputs.end()) {
          continue;
        }
        if (it->second > best.confidence) {
          best.role = candidate.role;
          best.confidence = it->second;
        }
      }
      if (best.role != domain::StemRole::Unknown && best.confidence >= 0.55) {
        return best;
      }
    }
  }

  const auto byName = predictFromName(stemName);
  if (byName.role != domain::StemRole::Unknown) {
    return byName;
  }
  return predictFromSpectrum(metrics);
}

} // namespace automix::ai
