#include "engine/SessionRepository.h"

#include <fstream>
#include <stdexcept>

#include "domain/JsonSerialization.h"

namespace automix::engine {

void SessionRepository::save(const std::filesystem::path& path, const domain::Session& session) const {
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open session file for writing: " + path.string());
  }

  domain::Json json = session;
  out << json.dump(2);
}

domain::Session SessionRepository::load(const std::filesystem::path& path) const {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open session file for reading: " + path.string());
  }

  domain::Json json;
  in >> json;
  return json.get<domain::Session>();
}

} // namespace automix::engine
