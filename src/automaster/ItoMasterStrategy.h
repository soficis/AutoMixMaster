#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "ai/ItoMasterAdapter.h"
#include "automaster/HeuristicAutoMasterStrategy.h"
#include "automaster/IAutoMasterStrategy.h"
#include "dsp/ItoMasterFxChain.h"

namespace automix::automaster {

// ─────────────────────────────────────────────────────────────────────────────
// ITO-Master AI mastering strategy (experimental, NON-DEFAULT mastering route).
//
// Implements the existing IAutoMasterStrategy interface. The route engages only
// when ALL of the following hold:
//   * the experimental toggle is on (static flag, default OFF; opt-in via
//     AUTOMIX_ITO_MASTER=1 or ItoMasterStrategy::setExperimentalEnabled),
//   * the caller has confirmed the CC BY-NC license consent gate
//     (ModelController::hasModelLicenseConsent in the app layer — injected as
//     Options::licenseConsented so the core stays app-independent),
//   * the ITO-Master pack directory carries all three artifacts
//     (fxencoder.onnx + mastering_tcn.onnx + config.json).
//
// When the route is inactive it defers to the sacred default heuristic chain
// (HeuristicAutoMasterStrategy), so the default mastering behaviour is
// byte-for-byte unchanged. When active it maps the 46 normalized ITO-Master
// parameters onto the native white-box FX chain (dsp::ItoMasterFxChain).
//
// Native ONNX Runtime is absent from this build, so the model stage validates
// the static tensor-shape contract parsed from config.json and falls back to
// the heuristic chain with a logged reason (the fallback never fabricates
// parameters). The license + attribution + experimental badge text below is the
// metadata surface the model card shows (reusing the T3.5 attribution string).
// ─────────────────────────────────────────────────────────────────────────────

class ItoMasterStrategy final : public IAutoMasterStrategy {
 public:
  struct Options {
    // Directory containing fxencoder.onnx, mastering_tcn.onnx and config.json
    // (the pack install directory, e.g. assets/modelhub/...).
    std::filesystem::path packDirectory;
    // T3.5 consent gate result (ModelController::hasModelLicenseConsent).
    bool licenseConsented = false;
  };

  explicit ItoMasterStrategy(Options options);

  // Experimental toggle (default OFF). The route is never the default path.
  static void setExperimentalEnabled(bool enabled);
  static bool experimentalEnabled();

  // Metadata surface for the model card (license + attribution + badge).
  static const char* licenseLabel();
  static const char* attributionText();
  static const char* experimentalBadge();

  // True when the route is fully engaged for this instance.
  bool isAvailable() const;

  domain::MasterPlan buildPlan(domain::MasterPreset preset, const engine::AudioBuffer& mixBuffer) const override;
  engine::AudioBuffer applyPlan(const engine::AudioBuffer& mixBuffer,
                                const domain::MasterPlan& plan,
                                MasteringReport* reportOut) const override;

 private:
  struct RouteState {
    ai::ItoMasterConfig config;
  };
  std::optional<RouteState> loadRouteState() const;
  static bool packArtifactsComplete(const std::filesystem::path& packDirectory);

  Options options_;
  HeuristicAutoMasterStrategy heuristic_;
  // Populated by buildPlan when the ITO route engaged; consumed by applyPlan.
  mutable std::optional<dsp::ItoMasterChainSettings> pendingSettings_;
  mutable std::vector<std::string> pendingLog_;
};

} // namespace automix::automaster
