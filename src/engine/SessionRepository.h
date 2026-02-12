#pragma once

#include <filesystem>

#include "domain/Session.h"

namespace automix::engine {

class SessionRepository {
 public:
  void save(const std::filesystem::path& path, const domain::Session& session) const;
  domain::Session load(const std::filesystem::path& path) const;
};

} // namespace automix::engine
