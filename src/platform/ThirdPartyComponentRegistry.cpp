#include "platform/ThirdPartyComponentRegistry.h"

namespace automix::platform {

std::vector<ThirdPartyComponent> ThirdPartyComponentRegistry::all() {
  return {
      {
          .name = "JUCE",
          .version = "8.0.8",
          .licenseId = "AGPL-3.0-or-later OR JUCE-commercial",
          .linkMode = ThirdPartyLinkMode::InProcess,
          .shipsByDefault = true,
          .notes = "Core app framework. Distribution mode controls license obligations.",
      },
      {
          .name = "nlohmann_json",
          .version = "3.11.3",
          .licenseId = "MIT",
          .linkMode = ThirdPartyLinkMode::InProcess,
          .shipsByDefault = true,
          .notes = "JSON serialization for sessions, reports, and model manifests.",
      },
      {
          .name = "libebur128",
          .version = "1.2.6",
          .licenseId = "MIT",
          .linkMode = ThirdPartyLinkMode::InProcess,
          .shipsByDefault = true,
          .notes = "BS.1770 loudness metering backend when enabled in build config.",
      },
      {
          .name = "PhaseLimiter",
          .version = "external",
          .licenseId = "See assets/phaselimiter/licenses",
          .linkMode = ThirdPartyLinkMode::External,
          .shipsByDefault = false,
          .notes = "Optional external renderer selected by user; app degrades to BuiltIn if unavailable.",
      },
      {
          .name = "ExternalLimiterRenderer",
          .version = "internal-wrapper",
          .licenseId = "User-supplied tool license",
          .linkMode = ThirdPartyLinkMode::External,
          .shipsByDefault = false,
          .notes = "Generic external renderer wrapper with timeout/cancel and compliance post-check.",
      },
  };
}

std::string toString(const ThirdPartyLinkMode linkMode) {
  switch (linkMode) {
    case ThirdPartyLinkMode::InProcess:
      return "in-process";
    case ThirdPartyLinkMode::External:
      return "external";
  }
  return "unknown";
}

} // namespace automix::platform
