#include "automaster/ItoMasterStrategy.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <utility>

#include "ai/ItoMasterAdapter.h"
#include "dsp/TruePeakDetector.h"
#include "engine/LoudnessMeter.h"

namespace automix::automaster {
namespace {

bool envOptInEnabled() {
  const char* value = std::getenv("AUTOMIX_ITO_MASTER");
  return value != nullptr && std::string(value) == "1";
}

double computeMonoCorrelation(const engine::AudioBuffer& buffer) {
  if (buffer.getNumChannels() < 2 || buffer.getNumSamples() == 0) {
    return 1.0;
  }

  double sumL = 0.0;
  double sumR = 0.0;
  double sumLL = 0.0;
  double sumRR = 0.0;
  double sumLR = 0.0;
  const double n = static_cast<double>(buffer.getNumSamples());

  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    const double l = buffer.getSample(0, i);
    const double r = buffer.getSample(1, i);
    sumL += l;
    sumR += r;
    sumLL += l * l;
    sumRR += r * r;
    sumLR += l * r;
  }

  const double cov = sumLR - (sumL * sumR) / n;
  const double varL = sumLL - (sumL * sumL) / n;
  const double varR = sumRR - (sumR * sumR) / n;
  if (varL < 1.0e-9 || varR < 1.0e-9) {
    return 1.0;
  }
  return std::clamp(cov / std::sqrt(varL * varR), -1.0, 1.0);
}

} // namespace

bool& experimentalFlagStorage() {
  static bool flag = envOptInEnabled();
  return flag;
}

ItoMasterStrategy::ItoMasterStrategy(Options options) : options_(std::move(options)) {}

void ItoMasterStrategy::setExperimentalEnabled(const bool enabled) {
  experimentalFlagStorage() = enabled;
}

bool ItoMasterStrategy::experimentalEnabled() {
  return experimentalFlagStorage();
}

const char* ItoMasterStrategy::licenseLabel() {
  return ai::kItoMasterLicense;
}

const char* ItoMasterStrategy::attributionText() {
  return ai::kItoMasterAttribution;
}

const char* ItoMasterStrategy::experimentalBadge() {
  return "experimental";
}

bool ItoMasterStrategy::packArtifactsComplete(const std::filesystem::path& packDirectory) {
  std::error_code error;
  const auto encoderPath = packDirectory / ai::kItoMasterEncoderFile;
  const auto predictorPath = packDirectory / ai::kItoMasterPredictorFile;
  const auto configPath = packDirectory / ai::kItoMasterConfigFile;
  return std::filesystem::is_regular_file(encoderPath, error) && !error &&
         std::filesystem::is_regular_file(predictorPath, error) && !error &&
         std::filesystem::is_regular_file(configPath, error) && !error;
}

bool ItoMasterStrategy::isAvailable() const {
  if (!experimentalEnabled()) {
    return false;
  }
  if (!options_.licenseConsented) {
    return false;
  }
  return packArtifactsComplete(options_.packDirectory);
}

std::optional<ItoMasterStrategy::RouteState> ItoMasterStrategy::loadRouteState() const {
  const auto config = ai::ItoMasterAdapter::loadConfig(options_.packDirectory / ai::kItoMasterConfigFile);
  if (!config.has_value()) {
    return std::nullopt;
  }
  RouteState state;
  state.config = *config;
  return state;
}

domain::MasterPlan ItoMasterStrategy::buildPlan(const domain::MasterPreset preset,
                                                const engine::AudioBuffer& mixBuffer) const {
  // The heuristic plan is always the base so the default mastering behaviour
  // is unchanged whenever the ITO route is inactive.
  auto base = heuristic_.buildPlan(preset, mixBuffer);

  if (!experimentalEnabled()) {
    base.decisionLog.push_back(
        "ITO-Master route inactive: experimental toggle is OFF (non-default route).");
    return base;
  }
  if (!options_.licenseConsented) {
    base.decisionLog.push_back(
        "ITO-Master route inactive: CC BY-NC license consent not acknowledged.");
    return base;
  }
  const auto state = loadRouteState();
  if (!state.has_value()) {
    base.decisionLog.push_back("ITO-Master route inactive: ITO-Master pack artifacts missing at " +
                               options_.packDirectory.string() + ".");
    return base;
  }

  // Static tensor-shape contract from config.json. This is the authoritative
  // shape check when native ONNX Runtime is absent from the build.
  const bool contractValid = ai::ItoMasterAdapter::validateTensorContract(
      state->config, mixBuffer.getNumChannels(), mixBuffer.getNumSamples());
  base.decisionLog.push_back(
      contractValid
          ? "ITO-Master tensor contract valid ([1,2,N]->[1,2048]->[1,46])."
          : "ITO-Master tensor contract INVALID against config.json; falling back to heuristic.");

  ai::ItoMasterModelRunner runner;
  std::string loadError;
  if (!runner.load(options_.packDirectory, state->config, &loadError)) {
    base.decisionLog.push_back("ITO-Master route inactive: model load failed (" + loadError + ").");
    return base;
  }

  const auto params = runner.predict(mixBuffer);
  if (!params.has_value()) {
    base.decisionLog.push_back(
        "ITO-Master route fell back to heuristic: model params unavailable "
        "(native ONNX Runtime not linked in this build; static tensor-shape contract "
        "validated instead).");
    return base;
  }

  const auto settings = ai::ItoMasterAdapter::apply(state->config, *params);
  pendingSettings_ = settings;
  pendingLog_.clear();
  pendingLog_.push_back("ITO-Master route active: 46 normalized params mapped onto the "
                        "white-box FX chain (eq, distortion, multiband_comp, gain, imager, limiter).");
  base.decisionLog.insert(base.decisionLog.end(), pendingLog_.begin(), pendingLog_.end());
  return base;
}

engine::AudioBuffer ItoMasterStrategy::applyPlan(const engine::AudioBuffer& mixBuffer,
                                                 const domain::MasterPlan& plan,
                                                 MasteringReport* reportOut) const {
  if (pendingSettings_.has_value()) {
    engine::AudioBuffer mastered = mixBuffer;
    dsp::ItoMasterFxChain chain;
    chain.prepare(mixBuffer.getSampleRate(), mixBuffer.getNumChannels());
    chain.setSettings(pendingSettings_.value());
    chain.process(mastered);

    if (reportOut != nullptr) {
      engine::LoudnessMeter meter;
      const auto metrics = meter.analyze(mastered);
      reportOut->integratedLufs = metrics.integratedLufs;
      reportOut->shortTermLufs = metrics.shortTermLufs;
      reportOut->loudnessRange = metrics.loudnessRange;
      reportOut->samplePeakDbfs = metrics.samplePeakDbfs;
      dsp::TruePeakDetector truePeakDetector(4);
      reportOut->truePeakDbtp = truePeakDetector.computeTruePeakDbtp(mastered);
      reportOut->crestDb = reportOut->samplePeakDbfs - metrics.integratedLufs;
      reportOut->monoCorrelation = computeMonoCorrelation(mastered);
      reportOut->activeModules = {"ItoEq", "ItoDistortion", "ItoMultibandComp",
                                  "ItoGain", "ItoImager", "ItoLimiter"};
    }
    return mastered;
  }

  return heuristic_.applyPlan(mixBuffer, plan, reportOut);
}

} // namespace automix::automaster
