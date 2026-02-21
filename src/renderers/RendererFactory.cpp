#include "renderers/RendererFactory.h"

#include "renderers/BuiltInRenderer.h"
#include "renderers/ExternalLimiterRenderer.h"
#include "renderers/FfmpegRenderer.h"
#include "renderers/PhaseLimiterRenderer.h"
#include "renderers/RsgainRenderer.h"
#include "renderers/SoxRenderer.h"

namespace automix::renderers {

std::unique_ptr<IRenderer> createRenderer(const std::string& preferredRenderer) {
  if (preferredRenderer.empty() || preferredRenderer == "BuiltIn") {
    return std::make_unique<BuiltInRenderer>();
  }

  if (preferredRenderer == "PhaseLimiter") {
    return std::make_unique<PhaseLimiterRenderer>();
  }

  if (preferredRenderer == "FFmpeg") {
    return std::make_unique<FfmpegRenderer>();
  }

  if (preferredRenderer == "SoX") {
    return std::make_unique<SoxRenderer>();
  }

  if (preferredRenderer == "rsgain") {
    return std::make_unique<RsgainRenderer>();
  }

  if (preferredRenderer == "ExternalLimiter" || preferredRenderer.rfind("ExternalUser", 0) == 0) {
    return std::make_unique<ExternalLimiterRenderer>();
  }

  // Any non-built-in renderer id discovered from registry descriptors is treated as
  // external limiter-compatible to avoid routing unexpected ids to the built-in renderer.
  return std::make_unique<ExternalLimiterRenderer>();
}

} // namespace automix::renderers
