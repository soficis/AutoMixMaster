#include "app/ui/SessionManager.h"

#include "domain/JsonSerialization.h"

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

std::string SessionManager::snapshot() const {
  const domain::Json json = session_;
  return json.dump();
}

void SessionManager::markSaved() {
  savedSnapshot_ = snapshot();
}

bool SessionManager::isModified() const {
  return snapshot() != savedSnapshot_;
}

} // namespace automix::app
