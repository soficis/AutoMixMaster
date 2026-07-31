#pragma once

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "engine/SimpleThreadPool.h"
#include "renderers/IRenderer.h"

namespace automix::renderers {

class ParallelRenderPipeline final {
 public:
  explicit ParallelRenderPipeline(int maxParallelStems = 0);

  ~ParallelRenderPipeline() = default;

  ParallelRenderPipeline(const ParallelRenderPipeline&) = delete;
  ParallelRenderPipeline& operator=(const ParallelRenderPipeline&) = delete;
  ParallelRenderPipeline(ParallelRenderPipeline&&) = delete;
  ParallelRenderPipeline& operator=(ParallelRenderPipeline&&) = delete;

  std::vector<RenderResult> renderStems(
      const std::vector<std::function<RenderResult()>>& stemRenderers);

  [[nodiscard]] int getMaxParallelStems() const noexcept;

 private:
  int resolveMaxParallelStems(int requested) const noexcept;

  int maxParallelStems_;
  std::unique_ptr<engine::SimpleThreadPool> pool_;
};

} // namespace automix::renderers
