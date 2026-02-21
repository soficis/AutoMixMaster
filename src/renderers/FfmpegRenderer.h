#pragma once

#include "renderers/IRenderer.h"

namespace automix::renderers {

class FfmpegRenderer final : public IRenderer {
 public:
  bool isAvailable() const override;
  RenderResult render(const domain::Session& session,
                      const domain::RenderSettings& settings,
                      const ProgressCallback& onProgress,
                      std::atomic_bool* cancelFlag) const override;
};

} // namespace automix::renderers
