#include "commands/CommandRegistry.h"
#include "commands/DevToolsUtils.h"

#include <fstream>
#include <iostream>
#include <map>

#include "analysis/StemHealthAssistant.h"
#include "analysis/StemAnalyzer.h"
#include "domain/JsonSerialization.h"
#include "domain/Session.h"
#include "engine/SessionRepository.h"

namespace {

using namespace automix::devtools;

int commandSessionDiff(const CommandArgs& args) {
  const auto baseArg = argValue(args, "--base");
  const auto headArg = argValue(args, "--head");
  if (!baseArg.has_value() || !headArg.has_value()) {
    std::cerr << "session-diff requires --base <session.json> --head <session.json>\n";
    return 2;
  }

  const auto baseJson = loadJsonFile(*baseArg);
  const auto headJson = loadJsonFile(*headArg);
  if (!baseJson.has_value() || !headJson.has_value()) {
    std::cerr << "Failed to load session JSON for diff.\n";
    return 1;
  }

  const auto patch = nlohmann::json::diff(*baseJson, *headJson);
  std::map<std::string, int> opCounts;
  for (const auto& op : patch) {
    if (op.is_object()) {
      opCounts[op.value("op", "unknown")] += 1;
    }
  }

  if (const auto outPathArg = argValue(args, "--out"); outPathArg.has_value()) {
    writeJsonFile(*outPathArg, patch);
    std::cout << "Wrote JSON patch to " << *outPathArg << "\n";
  } else {
    std::cout << patch.dump(2) << "\n";
  }

  if (hasFlag(args, "--summary")) {
    std::cout << "Patch operations: total=" << patch.size();
    for (const auto& [op, count] : opCounts) {
      std::cout << " " << op << "=" << count;
    }
    std::cout << "\n";
  }

  return 0;
}

int commandSessionMerge(const CommandArgs& args) {
  const auto baseArg = argValue(args, "--base");
  const auto leftArg = argValue(args, "--left");
  const auto rightArg = argValue(args, "--right");
  const auto outArg = argValue(args, "--out");
  if (!baseArg.has_value() || !leftArg.has_value() || !rightArg.has_value() || !outArg.has_value()) {
    std::cerr << "session-merge requires --base <session.json> --left <session.json> --right <session.json> --out <session.json>\n";
    return 2;
  }

  const auto baseJson = loadJsonFile(*baseArg);
  const auto leftJson = loadJsonFile(*leftArg);
  const auto rightJson = loadJsonFile(*rightArg);
  if (!baseJson.has_value() || !leftJson.has_value() || !rightJson.has_value()) {
    std::cerr << "Failed to load session JSON for merge.\n";
    return 1;
  }

  JsonMergeTelemetry telemetry;
  telemetry.preferRight = toLower(argValue(args, "--prefer").value_or("right")) != "left";

  const auto merged = mergeJsonNode(baseJson, leftJson, rightJson, "", &telemetry);
  if (!merged.has_value()) {
    std::cerr << "Merged session resolved to null unexpectedly.\n";
    return 1;
  }

  automix::domain::Session mergedSession;
  try {
    mergedSession = merged->get<automix::domain::Session>();
  } catch (const std::exception& error) {
    std::cerr << "Merged JSON is not a valid Session schema: " << error.what() << "\n";
    return 1;
  }

  automix::engine::SessionRepository repository;
  repository.save(*outArg, mergedSession);

  nlohmann::json report = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"base", *baseArg},
      {"left", *leftArg},
      {"right", *rightArg},
      {"out", *outArg},
      {"preferredSide", telemetry.preferRight ? "right" : "left"},
      {"conflictCount", telemetry.conflictCount},
      {"conflictPaths", telemetry.conflictPaths},
  };

  if (const auto reportArg = argValue(args, "--report"); reportArg.has_value()) {
    writeJsonFile(*reportArg, report);
    std::cout << "Merge report: " << *reportArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << report.dump(2) << "\n";
  } else {
    std::cout << "Merged session written to " << *outArg
              << " (conflicts resolved=" << telemetry.conflictCount
              << ", preferred=" << (telemetry.preferRight ? "right" : "left") << ")\n";
  }

  return 0;
}

int commandSessionReview(const CommandArgs& args) {
  const auto baseArg = argValue(args, "--base");
  const auto headArg = argValue(args, "--head");
  if (!baseArg.has_value() || !headArg.has_value()) {
    std::cerr << "session-review requires --base <session.json> --head <session.json>\n";
    return 2;
  }

  const auto baseJson = loadJsonFile(*baseArg);
  const auto headJson = loadJsonFile(*headArg);
  if (!baseJson.has_value() || !headJson.has_value()) {
    std::cerr << "Failed to load session files for review.\n";
    return 1;
  }

  const auto patch = nlohmann::json::diff(*baseJson, *headJson);
  std::map<std::string, int> topLevelCounts;
  std::vector<std::string> highlights;

  for (const auto& op : patch) {
    if (!op.is_object()) {
      continue;
    }
    const auto path = op.value("path", "");
    if (path.empty()) {
      continue;
    }
    const auto slash = path.find('/', 1);
    const auto top = slash == std::string::npos ? path : path.substr(0, slash);
    topLevelCounts[top] += 1;

    if (path.find("/renderSettings") == 0 ||
        path.find("/timeline") == 0 ||
        path.find("/mixPlan") == 0 ||
        path.find("/masterPlan") == 0) {
      highlights.push_back(op.value("op", "op") + " " + path);
    }
  }

  nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"base", *baseArg},
      {"head", *headArg},
      {"patchOperations", patch.size()},
      {"topLevelChangeCounts", topLevelCounts},
      {"highlights", highlights},
      {"patch", patch},
  };

  if (const auto outArg = argValue(args, "--out"); outArg.has_value()) {
    writeJsonFile(*outArg, payload);
    std::cout << "Session review written to " << *outArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Session review summary:\n";
    for (const auto& [top, count] : topLevelCounts) {
      std::cout << "  " << top << ": " << count << "\n";
    }
    if (!highlights.empty()) {
      std::cout << "Highlights:\n";
      for (const auto& line : highlights) {
        std::cout << "  - " << line << "\n";
      }
    }
  }

  return 0;
}

int commandProfileExport(const CommandArgs& args) {
  const auto outArg = argValue(args, "--out");
  if (!outArg.has_value()) {
    std::cerr << "profile-export requires --out <path>\n";
    return 2;
  }

  const auto idArg = argValue(args, "--id");
  const auto profiles = automix::domain::loadProjectProfiles(std::filesystem::current_path());
  nlohmann::json payload = nlohmann::json::array();

  if (idArg.has_value()) {
    const auto profile = automix::domain::findProjectProfile(profiles, *idArg);
    if (!profile.has_value()) {
      std::cerr << "Profile not found: " << *idArg << "\n";
      return 1;
    }
    payload.push_back(projectProfileToJson(profile.value()));
  } else {
    for (const auto& profile : profiles) {
      payload.push_back(projectProfileToJson(profile));
    }
  }

  writeJsonFile(*outArg, payload);
  std::cout << "Exported " << payload.size() << " profile(s) to " << *outArg << "\n";
  return 0;
}

int commandProfileImport(const CommandArgs& args) {
  const auto inArg = argValue(args, "--in");
  if (!inArg.has_value()) {
    std::cerr << "profile-import requires --in <path>\n";
    return 2;
  }

  const auto source = loadJsonFile(*inArg);
  if (!source.has_value()) {
    std::cerr << "Failed to read profile file: " << *inArg << "\n";
    return 1;
  }

  std::vector<automix::domain::ProjectProfile> imported;
  if (source->is_array()) {
    for (const auto& item : *source) {
      if (const auto parsed = projectProfileFromJson(item); parsed.has_value()) {
        imported.push_back(parsed.value());
      }
    }
  } else if (source->is_object()) {
    if (const auto parsed = projectProfileFromJson(*source); parsed.has_value()) {
      imported.push_back(parsed.value());
    }
  }

  if (imported.empty()) {
    std::cerr << "No valid profile records found in " << *inArg << "\n";
    return 1;
  }

  const auto outPath = argValue(args, "--out").value_or(profileCatalogPath().string());
  nlohmann::json existing = loadJsonFile(outPath).value_or(nlohmann::json::array());
  if (!existing.is_array()) {
    existing = nlohmann::json::array();
  }

  std::map<std::string, nlohmann::json> byId;
  for (const auto& item : existing) {
    if (!item.is_object()) {
      continue;
    }
    const auto id = item.value("id", "");
    if (!id.empty()) {
      byId[id] = item;
    }
  }
  for (const auto& profile : imported) {
    byId[profile.id] = projectProfileToJson(profile);
  }

  nlohmann::json merged = nlohmann::json::array();
  for (const auto& [id, item] : byId) {
    (void)id;
    merged.push_back(item);
  }

  writeJsonFile(outPath, merged);
  std::cout << "Imported " << imported.size() << " profile(s) into " << outPath << "\n";
  return 0;
}

int commandStemHealth(const CommandArgs& args) {
  const auto sessionPathArg = argValue(args, "--session");
  if (!sessionPathArg.has_value()) {
    std::cerr << "stem-health requires --session <session.json>\n";
    return 2;
  }

  const auto outPathArg = argValue(args, "--out");
  const bool jsonOutput = hasFlag(args, "--json");

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  automix::analysis::StemAnalyzer analyzer;
  const auto analysisEntries = analyzer.analyzeSession(session);
  automix::analysis::StemHealthAssistant assistant;
  const auto report = assistant.analyze(session, analysisEntries);
  const auto reportJson = assistant.toJson(report);
  const auto reportText = assistant.toText(report);

  if (outPathArg.has_value()) {
    std::filesystem::path outPath(*outPathArg);
    std::filesystem::path jsonPath = outPath;
    std::filesystem::path textPath = outPath;

    if (outPath.extension() == ".json") {
      textPath.replace_extension(".txt");
    } else if (outPath.extension() == ".txt") {
      jsonPath.replace_extension(".json");
    } else {
      jsonPath += ".json";
      textPath += ".txt";
    }

    writeJsonFile(jsonPath, reportJson);
    std::ofstream textOut(textPath);
    if (textOut.is_open()) {
      textOut << reportText << "\n";
    }
    std::cout << "Wrote stem health report JSON: " << jsonPath.string() << "\n";
    std::cout << "Wrote stem health report text: " << textPath.string() << "\n";
  }

  if (jsonOutput) {
    std::cout << reportJson.dump(2) << "\n";
  } else {
    std::cout << reportText << "\n";
  }

  return 0;
}

int commandAdaptiveAssistant(const CommandArgs& args) {
  const auto sessionArg = argValue(args, "--session");
  if (!sessionArg.has_value()) {
    std::cerr << "adaptive-assistant requires --session <session.json>\n";
    return 2;
  }

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionArg);
  automix::analysis::StemAnalyzer analyzer;
  const auto analysisEntries = analyzer.analyzeSession(session);
  automix::analysis::StemHealthAssistant assistant;
  const auto health = assistant.analyze(session, analysisEntries);

  nlohmann::json fixes = nlohmann::json::array();
  for (const auto& issue : health.issues) {
    if (issue.code == "harshness_risk") {
      fixes.push_back({
          {"priority", issue.severity == automix::analysis::StemHealthSeverity::Critical ? "high" : "medium"},
          {"stemId", issue.stemId},
          {"action", "reduce_high_band_harshness"},
          {"details", "Apply de-harsh EQ in 3k-8k region and lower clipping-prone gains."},
      });
    } else if (issue.code == "pumping_risk") {
      fixes.push_back({
          {"priority", "medium"},
          {"stemId", issue.stemId},
          {"action", "relax_compression"},
          {"details", "Increase compressor release and lower ratio/threshold aggressiveness."},
      });
    } else if (issue.code == "masking_conflict" || issue.code == "spectral_masking") {
      fixes.push_back({
          {"priority", "medium"},
          {"stemId", issue.stemId},
          {"action", "rebalance_masking"},
          {"details", "Cut overlapping bands and spread conflicting stems with mild pan/EQ separation."},
      });
    } else if (issue.code == "mono_risk") {
      fixes.push_back({
          {"priority", "medium"},
          {"stemId", issue.stemId},
          {"action", "improve_mono_compatibility"},
          {"details", "Narrow extreme stereo content and verify mono fold-down phase correlation."},
      });
    }
  }

  const auto compareArg = argValue(args, "--compare-report");
  if (compareArg.has_value()) {
    if (const auto compare = loadJsonFile(*compareArg); compare.has_value() &&
                                                  compare->contains("ranking") &&
                                                  compare->at("ranking").is_array() &&
                                                  !compare->at("ranking").empty()) {
      const auto top = compare->at("ranking").front();
      const auto bestRenderer = top.value("rendererId", "");
      const auto score = top.value("score", 0.0);
      if (!bestRenderer.empty()) {
        fixes.push_back({
            {"priority", "medium"},
            {"stemId", ""},
            {"action", "renderer_recommendation"},
            {"details", "Comparator top renderer is '" + bestRenderer + "' (score=" + std::to_string(score) +
                            "). Consider profile pinning to this renderer."},
        });
      }
    }
  }

  if (fixes.empty()) {
    fixes.push_back({
        {"priority", "low"},
        {"stemId", ""},
        {"action", "no_critical_changes"},
        {"details", "No major corrective chain detected; keep current profile and run final compliance check."},
    });
  }

  const nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"sessionPath", *sessionArg},
      {"overallRisk", health.overallRisk},
      {"hasCriticalIssues", health.hasCriticalIssues},
      {"issueCount", health.issues.size()},
      {"fixChain", fixes},
  };

  if (const auto outArg = argValue(args, "--out"); outArg.has_value()) {
    writeJsonFile(*outArg, payload);
    std::cout << "Adaptive assistant report: " << *outArg << "\n";
  }

  if (hasFlag(args, "--json")) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Adaptive assistant generated " << fixes.size() << " fix-chain step(s).\n";
  }
  return 0;
}

} // namespace

void registerSessionCommands(automix::devtools::CommandRegistry& registry) {
  registry.add("session-diff", commandSessionDiff);
  registry.add("session-merge", commandSessionMerge);
  registry.add("session-review", commandSessionReview);
  registry.add("profile-export", commandProfileExport);
  registry.add("profile-import", commandProfileImport);
  registry.add("stem-health", commandStemHealth);
  registry.add("adaptive-assistant", commandAdaptiveAssistant);
}
