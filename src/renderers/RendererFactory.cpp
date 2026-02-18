#include "renderers/RendererFactory.h"

#include "renderers/BuiltInRenderer.h"
#include "renderers/ExternalLimiterRenderer.h"
#include "renderers/PhaseLimiterRenderer.h"

namespace automix::renderers {

std::unique_ptr<IRenderer> createRenderer(const std::string& preferredRenderer) {
  if (preferredRenderer.empty() || preferredRenderer == "BuiltIn") {
    return std::make_unique<BuiltInRenderer>();
  }

  if (preferredRenderer == "PhaseLimiter") {
    return std::make_unique<PhaseLimiterRenderer>();
  }

  if (preferredRenderer == "ExternalLimiter" || preferredRenderer.rfind("ExternalUser", 0) == 0) {
    return std::make_unique<ExternalLimiterRenderer>();
  }

  // Any non-built-in renderer id discovered from registry descriptors is treated as
  // external limiter-compatible to avoid routing unexpected ids to the built-in renderer.
  return std::make_unique<ExternalLimiterRenderer>();
}

} // namespace automix::renderers
