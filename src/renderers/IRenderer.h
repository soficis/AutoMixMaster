#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "domain/Session.h"

namespace automix::renderers {

struct RenderResult {
  bool success = false;
  bool cancelled = false;
  std::string rendererName;
  std::string outputAudioPath;
  std::string reportPath;
  std::vector<std::string> logs;
};

class IRenderer {
 public:
  virtual ~IRenderer() = default;

  virtual bool isAvailable() const = 0;

  using ProgressCallback = std::function<void(double, const std::string&)>;

  virtual RenderResult render(const domain::Session& session,
                              const domain::RenderSettings& settings,
                              const ProgressCallback& onProgress,
                              std::atomic_bool* cancelFlag) const = 0;
};

} // namespace automix::renderers
