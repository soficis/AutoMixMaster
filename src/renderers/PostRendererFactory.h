#pragma once

#include <memory>
#include <string>

#include "renderers/IPostRenderer.h"

namespace automix::renderers {

std::unique_ptr<IPostRenderer> createPostRenderer(const std::string& rendererId);

} // namespace automix::renderers
