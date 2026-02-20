#pragma once

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

 private:
  domain::Session session_;
};

} // namespace automix::app
