#pragma once

#include <atomic>
#include <filesystem>
#include <string>

#include "renderers/IRenderer.h"

namespace automix::renderers {

class IPostRenderer {
 public:
  virtual ~IPostRenderer() = default;

  virtual bool isAvailable() const = 0;

  virtual RenderResult applyPostRender(const std::filesystem::path& audioPath,
                                       const std::string& existingReportPath,
                                       const IRenderer::ProgressCallback& onProgress,
                                       std::atomic_bool* cancelFlag) const = 0;
};

} // namespace automix::renderers
