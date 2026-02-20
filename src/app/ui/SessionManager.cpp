#include "app/ui/SessionManager.h"

namespace automix::app {

domain::Session& SessionManager::session() {
  return session_;
}

const domain::Session& SessionManager::session() const {
  return session_;
}

void SessionManager::replaceSession(domain::Session session) {
  session_ = std::move(session);
}

} // namespace automix::app
