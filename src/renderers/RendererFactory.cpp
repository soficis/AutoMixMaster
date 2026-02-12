#include "renderers/RendererFactory.h"

#include "renderers/BuiltInRenderer.h"
#include "renderers/PhaseLimiterRenderer.h"

namespace automix::renderers {

std::unique_ptr<IRenderer> createRenderer(const std::string& preferredRenderer) {
  if (preferredRenderer == "PhaseLimiter") {
    return std::make_unique<PhaseLimiterRenderer>();
  }

  return std::make_unique<BuiltInRenderer>();
}

} // namespace automix::renderers
