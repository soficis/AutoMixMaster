#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace automix::ai {

// Classifies a model license identifier the way the consent UX needs it.
//
// Why this exists: the consent decision used to be a hardcoded two-entry repo
// list, so any *other* non-commercial or simply undeclared-license model
// installed with no notice at all. That is the wrong default in both
// directions - a CC BY-NC weight is not usable commercially, and an undeclared
// license is not usable *provably* commercially either.
//
// Unknown/undeclared therefore requires consent rather than skipping it. That
// is deliberately the strict direction: the cost of an extra confirmation is
// one dialog, whereas the cost of the permissive default is shipping or
// redistributing a weight whose licence nobody has established.
namespace ModelLicensePolicy {

inline std::string normalize(std::string_view licenseId) {
  std::string out;
  out.reserve(licenseId.size());
  for (const auto ch : licenseId) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return out;
}

// True when the identifier denotes a non-commercial ("NC") licence. Matches on
// the normalised token so "cc-by-nc-4.0", "CC BY-NC 4.0" and
// "cc_by_nc_4_0" all classify identically.
inline bool isNonCommercial(std::string_view licenseId) {
  const auto normalized = normalize(licenseId);
  return normalized.find("nc") != std::string::npos ||
         normalized.find("noncommercial") != std::string::npos ||
         normalized.find("non-commercial") != std::string::npos;
}

// True when the licence is absent or too vague to establish permitted use.
// "unknown" is what inferLicense() returns when a model card declares neither
// cardData.license nor a license:* tag; "unverified" is the audit manifest's
// marker for a card that was checked and found to declare nothing.
inline bool isUndeclared(std::string_view licenseId) {
  const auto normalized = normalize(licenseId);
  return normalized.empty() || normalized == "unknown" || normalized == "unverified" ||
         normalized == "none" || normalized == "other" || normalized == "noassertion";
}

// The single question the consent UX asks. Non-commercial and undeclared
// licences both require an explicit, recorded per-model opt-in.
inline bool requiresUserConsent(std::string_view licenseId) {
  return isNonCommercial(licenseId) || isUndeclared(licenseId);
}

// Canonical deed URL for the licences the curated catalogue can present.
// Returns an empty view for anything unrecognised so the dialog can omit the
// link rather than print a wrong one.
inline std::string_view licenseUrl(std::string_view licenseId) {
  const auto normalized = normalize(licenseId);
  // The Creative Commons branch is gated on "cc" so an unrelated identifier that
  // merely contains "by" (e.g. a custom name) cannot be handed a CC deed URL.
  const auto isCreativeCommons = normalized.find("cc ") != std::string::npos ||
                                 normalized.find("cc-") != std::string::npos ||
                                 normalized.find("cc_") != std::string::npos ||
                                 normalized.rfind("cc", 0) == 0;
  if (isCreativeCommons) {
    if (normalized.find("nc-sa") != std::string::npos) {
      return "https://creativecommons.org/licenses/by-nc-sa/4.0/";
    }
    if (normalized.find("nc") != std::string::npos) {
      return "https://creativecommons.org/licenses/by-nc/4.0/";
    }
    if (normalized.find("sa") != std::string::npos) {
      return "https://creativecommons.org/licenses/by-sa/4.0/";
    }
    if (normalized.find("by") != std::string::npos) {
      return "https://creativecommons.org/licenses/by/4.0/";
    }
  }
  if (normalized.find("apache") != std::string::npos) {
    return "https://www.apache.org/licenses/LICENSE-2.0";
  }
  if (normalized.find("mit") != std::string::npos) {
    return "https://opensource.org/license/mit";
  }
  if (normalized.find("bsd-3") != std::string::npos) {
    return "https://opensource.org/license/bsd-3-clause";
  }
  if (normalized.find("bsd-2") != std::string::npos) {
    return "https://opensource.org/license/bsd-2-clause";
  }
  if (normalized.find("gpl-3") != std::string::npos || normalized.find("gpl3") != std::string::npos) {
    return "https://www.gnu.org/licenses/gpl-3.0.html";
  }
  return {};
}

// Human-facing reason, shown as the consent dialog's headline so the user
// knows *why* they are being asked rather than only that they are.
inline std::string_view consentReason(std::string_view licenseId) {
  if (isNonCommercial(licenseId)) {
    return "This model is published under a NON-COMMERCIAL licence. It may only be used for "
           "non-commercial purposes, and must not be redistributed or bundled into a product you "
           "sell.";
  }
  if (isUndeclared(licenseId)) {
    return "This model does not declare a licence that can be verified, so permitted use cannot be "
           "established. Treat it as research-only until the publisher clarifies the terms.";
  }
  return {};
}

} // namespace ModelLicensePolicy

} // namespace automix::ai
