// Catch2 main is provided by Catch2::Catch2WithMain.

#include <cstdlib>
#include <filesystem>
#include <string>

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace {

// Downloaded models default to the user's real profile
// (%LOCALAPPDATA%\AutoMixMaster\modelhub). Point every test at a throwaway
// hub instead, so test runs can never write fake packs or consents there.
class IsolatedModelHub final : public Catch::EventListenerBase {
 public:
  using Catch::EventListenerBase::EventListenerBase;

  void testRunStarting(const Catch::TestRunInfo&) override {
    const auto root = std::filesystem::temp_directory_path() / "automix_tests_modelhub";
#if defined(_WIN32)
    _putenv_s("AUTOMIX_MODEL_HUB_ROOT", root.string().c_str());
#else
    setenv("AUTOMIX_MODEL_HUB_ROOT", root.string().c_str(), 1);
#endif
  }
};

} // namespace

CATCH_REGISTER_LISTENER(IsolatedModelHub)
