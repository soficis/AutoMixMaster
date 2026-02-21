#pragma once

#include <filesystem>
#include <string>

#include "renderers/IPostRenderer.h"
#include "renderers/IRenderer.h"

namespace automix::renderers {

class RsgainRenderer final : public IRenderer, public IPostRenderer {
 public:
  bool isAvailable() const override;
  RenderResult render(const domain::Session& session,
                      const domain::RenderSettings& settings,
                      const ProgressCallback& onProgress,
                      std::atomic_bool* cancelFlag) const override;

  RenderResult applyPostRender(const std::filesystem::path& audioPath,
                               const std::string& existingReportPath,
                               const ProgressCallback& onProgress,
                               std::atomic_bool* cancelFlag) const override;
};

} // namespace automix::renderers
