#pragma once

#include <memory>

#include "renderers/IRenderer.h"

namespace automix::renderers {

std::unique_ptr<IRenderer> createRenderer(const std::string& preferredRenderer);

} // namespace automix::renderers
