#include "renderers/ParallelRenderPipeline.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace automix::renderers {

ParallelRenderPipeline::ParallelRenderPipeline(const int maxParallelStems)
    : maxParallelStems_(resolveMaxParallelStems(maxParallelStems)),
      pool_(std::make_unique<engine::SimpleThreadPool>(
          static_cast<unsigned int>(maxParallelStems_))) {}

int ParallelRenderPipeline::resolveMaxParallelStems(const int requested) const noexcept {
  if (requested > 0) {
    return requested;
  }
  return static_cast<int>(std::thread::hardware_concurrency());
}

int ParallelRenderPipeline::getMaxParallelStems() const noexcept {
  return maxParallelStems_;
}

std::vector<RenderResult> ParallelRenderPipeline::renderStems(
    const std::vector<std::function<RenderResult()>>& stemRenderers) {
  const auto total = static_cast<int>(stemRenderers.size());
  std::vector<RenderResult> results(total);
  if (total == 0) {
    return results;
  }

  const int batchSize = std::max(1, maxParallelStems_);

  for (int batchStart = 0; batchStart < total; batchStart += batchSize) {
    const int batchEnd = std::min(batchStart + batchSize, total);
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<size_t>(batchEnd - batchStart));

    for (int i = batchStart; i < batchEnd; ++i) {
      auto future = pool_->enqueue(
          [&results, &stemRenderers, i]() { results[i] = stemRenderers[i](); });
      futures.push_back(std::move(future));
    }

    for (auto& future : futures) {
      if (future.valid()) {
        future.get();
      }
    }
  }

  return results;
}

} // namespace automix::renderers
