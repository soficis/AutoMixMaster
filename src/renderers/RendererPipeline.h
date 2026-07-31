#pragma once

#include <atomic>
#include <functional>
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

int effectiveParallelism(const domain::RenderSettings& settings,
                         int taskCount);

void parallelFor(int taskCount,
                 int parallelism,
                 const std::function<void(int index)>& func);

} // namespace automix::renderers
