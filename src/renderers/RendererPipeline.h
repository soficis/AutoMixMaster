#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "domain/Session.h"
#include "renderers/IRenderer.h"

namespace automix::renderers {

std::vector<std::string> resolveRendererChain(const domain::RenderSettings& settings);

RenderResult renderWithPipeline(const domain::Session& session,
                                const domain::RenderSettings& settings,
                                const IRenderer::ProgressCallback& onProgress,
                                std::atomic_bool* cancelFlag);

} // namespace automix::renderers
