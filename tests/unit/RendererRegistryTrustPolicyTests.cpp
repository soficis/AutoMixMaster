#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "renderers/RendererRegistry.h"

namespace {

uint64_t fnv1a64(const std::string& input) {
  uint64_t hash = 14695981039346656037ull;
  constexpr uint64_t prime = 1099511628211ull;
  for (const auto c : input) {
    hash ^= static_cast<uint8_t>(c);
    hash *= prime;
  }
  return hash;
}

std::string toHex(const uint64_t value) {
  std::ostringstream out;
  out << std::hex << value;
  return out.str();
}

class ScopedCurrentPath {
 public:
  explicit ScopedCurrentPath(const std::filesystem::path& next)
      : original_(std::filesystem::current_path()) {
    std::filesystem::current_path(next);
  }

  ~ScopedCurrentPath() { std::filesystem::current_path(original_); }

 private:
  std::filesystem::path original_;
};

} // namespace

TEST_CASE("Renderer registry enforces signed descriptors when policy requires it", "[renderer][registry][trust]") {
  const auto nonce = std::to_string(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count());
  const auto root = std::filesystem::temp_directory_path() / ("automix_renderer_trust_" + nonce);
  const auto renderersRoot = root / "assets" / "renderers";
  const auto unsignedDir = renderersRoot / "unsigned_vendor";
  const auto signedDir = renderersRoot / "signed_vendor";
  std::filesystem::create_directories(unsignedDir);
  std::filesystem::create_directories(signedDir);

  {
    nlohmann::json trustPolicy = {
        {"enforceSignedDescriptors", true},
        {"trustedSigners", nlohmann::json::array({"automix_official"})},
    };
    std::ofstream out(renderersRoot / "trust_policy.json");
    out << trustPolicy.dump(2);
  }

  {
    std::ofstream binary(unsignedDir / "vendor.bin");
    binary << "unsigned";
    nlohmann::json descriptor = {
        {"id", "UnsignedVendor"},
        {"name", "Unsigned Vendor"},
        {"version", "1.0"},
        {"licenseId", "MIT"},
        {"binaryPath", "vendor.bin"},
    };
    std::ofstream out(unsignedDir / "renderer.json");
    out << descriptor.dump(2);
  }

  {
    std::ofstream binary(signedDir / "vendor.bin");
    binary << "signed";

    nlohmann::json descriptor = {
        {"id", "SignedVendor"},
        {"name", "Signed Vendor"},
        {"version", "1.0"},
        {"licenseId", "MIT"},
        {"binaryPath", "vendor.bin"},
    };
    const auto signature = toHex(fnv1a64(descriptor.dump()));
    descriptor["signature"] = {
        {"signer", "automix_official"},
        {"algorithm", "fnv1a64"},
        {"value", signature},
    };

    std::ofstream out(signedDir / "renderer.json");
    out << descriptor.dump(2);
  }

  {
    ScopedCurrentPath guard(root);
    automix::renderers::RendererRegistry registry;
    const auto infos = registry.list();

    bool foundUnsigned = false;
    bool foundSigned = false;
    for (const auto& info : infos) {
      if (info.id == "UnsignedVendor") {
        foundUnsigned = true;
      }
      if (info.id == "SignedVendor") {
        foundSigned = true;
        REQUIRE(info.trustPolicyStatus == "signature_valid");
      }
    }

    REQUIRE_FALSE(foundUnsigned);
    REQUIRE(foundSigned);
  }

  std::filesystem::remove_all(root);
}
