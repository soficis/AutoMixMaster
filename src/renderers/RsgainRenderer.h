#pragma once

#include <filesystem>
#include <string>

#include "renderers/IRenderer.h"

namespace automix::renderers {

class RsgainRenderer final : public IRenderer {
 public:
  bool isAvailable() const override;
  RenderResult render(const domain::Session& session,
                      const domain::RenderSettings& settings,
                      const ProgressCallback& onProgress,
                      std::atomic_bool* cancelFlag) const override;

  // Apply ReplayGain tagging in-place on an already-rendered audio file.
  // Does not perform any audio rendering; only tags the file at audioPath.
  RenderResult applyTagging(const std::filesystem::path& audioPath,
                             const std::string& existingReportPath,
                             const ProgressCallback& onProgress,
                             std::atomic_bool* cancelFlag) const;
};

} // namespace automix::renderers
