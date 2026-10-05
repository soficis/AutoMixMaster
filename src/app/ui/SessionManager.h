#pragma once

#include <string>
#include <utility>

#include "domain/Session.h"

namespace automix::app {

/// Centralized session state container.
/// All session mutations go through this class.
class SessionManager {
 public:
  domain::Session& session();
  const domain::Session& session() const;

  void replaceSession(domain::Session session);

  /// Records the current session as the saved baseline for isModified().
  void markSaved();
  bool isModified() const;

 private:
  std::string snapshot() const;

  domain::Session session_;
  std::string savedSnapshot_;
};

} // namespace automix::app
