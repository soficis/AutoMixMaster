#include "renderers/PostRendererFactory.h"

#include "renderers/RsgainRenderer.h"

namespace automix::renderers {

std::unique_ptr<IPostRenderer> createPostRenderer(const std::string& rendererId) {
  if (rendererId == "rsgain") {
    return std::make_unique<RsgainRenderer>();
  }

  return nullptr;
}

} // namespace automix::renderers
