#include "renderers/RendererFactory.h"

#include "renderers/BuiltInRenderer.h"
#include "renderers/ExternalLimiterRenderer.h"
#include "renderers/PhaseLimiterRenderer.h"

namespace automix::renderers {

std::unique_ptr<IRenderer> createRenderer(const std::string& preferredRenderer) {
  if (preferredRenderer == "PhaseLimiter") {
    return std::make_unique<PhaseLimiterRenderer>();
  }
  if (preferredRenderer == "ExternalLimiter" || preferredRenderer.rfind("ExternalUser", 0) == 0) {
    return std::make_unique<ExternalLimiterRenderer>();
  }

  return std::make_unique<BuiltInRenderer>();
}

} // namespace automix::renderers
