#include "analysis/StemHealthAssistant.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include <nlohmann/json.hpp>

namespace automix::analysis {
namespace {

void addIssue(StemHealthReport* report,
              const StemAnalysisEntry& entry,
              const std::string& code,
              const std::string& message,
              const StemHealthSeverity severity,
              const double score) {
  report->issues.push_back(StemHealthIssue{
      .stemId = entry.stemId,
      .stemName = entry.stemName,
      .code = code,
      .message = message,
      .severity = severity,
      .score = score,
  });
  report->overallRisk = std::max(report->overallRisk, score);
  report->hasCriticalIssues = report->hasCriticalIssues || severity == StemHealthSeverity::Critical;
}

} // namespace

std::string toString(const StemHealthSeverity severity) {
  switch (severity) {
    case StemHealthSeverity::Info:
      return "info";
    case StemHealthSeverity::Warning:
      return "warning";
    case StemHealthSeverity::Critical:
      return "critical";
  }
  return "info";
}

StemHealthReport StemHealthAssistant::analyze(const domain::Session& session,
                                              const std::vector<StemAnalysisEntry>& analysisEntries) const {
  StemHealthReport report;
  if (analysisEntries.empty()) {
    return report;
  }

  double summedLowEnergy = 0.0;
  double summedMidEnergy = 0.0;
  double summedHighEnergy = 0.0;
  for (const auto& entry : analysisEntries) {
    summedLowEnergy += std::max(0.0, entry.metrics.lowEnergy);
    summedMidEnergy += std::max(0.0, entry.metrics.midEnergy);
    summedHighEnergy += std::max(0.0, entry.metrics.highEnergy);
  }
  (void)summedMidEnergy;

  for (const auto& entry : analysisEntries) {
    const auto& metrics = entry.metrics;

    if (metrics.artifactRisk > 0.72) {
      addIssue(&report,
               entry,
               "harshness_risk",
               "High artifact risk indicates likely harshness or separation residue.",
               StemHealthSeverity::Critical,
               std::min(1.0, metrics.artifactRisk));
    } else if (metrics.artifactRisk > 0.55) {
      addIssue(&report,
               entry,
               "harshness_risk",
               "Moderate artifact risk; consider de-harsh EQ or transient cleanup.",
               StemHealthSeverity::Warning,
               std::min(1.0, metrics.artifactRisk));
    }

    if (metrics.stereoCorrelation < 0.10) {
      addIssue(&report,
               entry,
               "mono_risk",
               "Low stereo correlation suggests mono-compatibility risk.",
               StemHealthSeverity::Warning,
               std::clamp(0.7 - metrics.stereoCorrelation, 0.0, 1.0));
    }

    if (metrics.silenceRatio > 0.72) {
      addIssue(&report,
               entry,
               "pumping_risk",
               "High silence ratio can trigger audible pumping after mastering compression.",
               StemHealthSeverity::Warning,
               std::clamp(metrics.silenceRatio, 0.0, 1.0));
    }

    const double lowShare = summedLowEnergy > 1.0e-9 ? metrics.lowEnergy / summedLowEnergy : 0.0;
    if (lowShare > 0.70 && entry.metrics.crestDb < 8.0) {
      addIssue(&report,
               entry,
               "masking_conflict",
               "Dominant low-band energy may mask kick/bass definition.",
               StemHealthSeverity::Warning,
               std::clamp(lowShare, 0.0, 1.0));
    }

    const double highShare = summedHighEnergy > 1.0e-9 ? metrics.highEnergy / summedHighEnergy : 0.0;
    if (highShare > 0.55 && metrics.spectralFlatness > 0.65) {
      addIssue(&report,
               entry,
               "spectral_masking",
               "Dense high-band texture may cause cymbal/vocal masking.",
               StemHealthSeverity::Warning,
               std::clamp(highShare + metrics.spectralFlatness * 0.2, 0.0, 1.0));
    }
  }

  if (session.mixPlan.has_value() && !session.mixPlan->stemDecisions.empty()) {
    for (const auto& decision : session.mixPlan->stemDecisions) {
      if (decision.enableCompressor && decision.compressorRatio > 4.0 && decision.compressorReleaseMs < 60.0) {
        auto issue = StemHealthIssue{};
        issue.stemId = decision.stemId;
        issue.stemName = decision.stemId;
        issue.code = "pumping_risk";
        issue.message = "Aggressive compression settings may create pumping artifacts.";
        issue.severity = StemHealthSeverity::Warning;
        issue.score = 0.72;
        report.issues.push_back(issue);
        report.overallRisk = std::max(report.overallRisk, issue.score);
      }
    }
  }

  std::sort(report.issues.begin(), report.issues.end(), [](const StemHealthIssue& a, const StemHealthIssue& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.stemId < b.stemId;
  });

  return report;
}

nlohmann::json StemHealthAssistant::toJson(const StemHealthReport& report) const {
  nlohmann::json issues = nlohmann::json::array();
  for (const auto& issue : report.issues) {
    issues.push_back({
        {"stemId", issue.stemId},
        {"stemName", issue.stemName},
        {"code", issue.code},
        {"message", issue.message},
        {"severity", toString(issue.severity)},
        {"score", issue.score},
    });
  }

  return {
      {"overallRisk", report.overallRisk},
      {"hasCriticalIssues", report.hasCriticalIssues},
      {"issues", issues},
  };
}

std::string StemHealthAssistant::toText(const StemHealthReport& report) const {
  std::ostringstream out;
  out << "Stem Health Report\n";
  out << "Overall risk: " << std::fixed << std::setprecision(2) << report.overallRisk << "\n";
  out << "Critical issues: " << (report.hasCriticalIssues ? "yes" : "no") << "\n";

  if (report.issues.empty()) {
    out << "No pre-export stem health issues detected.";
    return out.str();
  }

  for (const auto& issue : report.issues) {
    out << "- [" << toString(issue.severity) << "] " << issue.stemName << " (" << issue.code << ")"
        << " score=" << std::fixed << std::setprecision(2) << issue.score
        << " :: " << issue.message << "\n";
  }
  return out.str();
}

} // namespace automix::analysis
