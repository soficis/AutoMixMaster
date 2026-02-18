#include "commands/CommandRegistry.h"
#include "commands/DevToolsUtils.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include "analysis/StemAnalyzer.h"
#include "domain/Session.h"
#include "domain/Stem.h"
#include "domain/StemOrigin.h"
#include "domain/StemRole.h"
#include "engine/AudioFileIO.h"
#include "engine/BatchQueueRunner.h"
#include "engine/OfflineRenderPipeline.h"
#include "engine/SessionRepository.h"
#include "renderers/RendererFactory.h"
#include "ai/FeatureSchema.h"
#include "util/WavWriter.h"

namespace {

using namespace automix::devtools;

int commandCompareRenders(const CommandArgs& args) {
  const auto sessionPathArg = argValue(args, "--session");
  if (!sessionPathArg.has_value()) {
    std::cerr << "compare-renders requires --session <session.json>\n";
    return 2;
  }

  const auto outDirArg = argValue(args, "--out-dir").value_or("comparison_out");
  const auto renderersArg = argValue(args, "--renderers").value_or("BuiltIn,PhaseLimiter");
  auto rendererIds = splitCommaSeparated(renderersArg);
  if (rendererIds.empty()) {
    rendererIds = {"BuiltIn"};
  }

  const auto formatOverride = argValue(args, "--format");
  const auto externalBinaryArg = argValue(args, "--external-binary");
  const bool jsonOutput = hasFlag(args, "--json");

  struct ComparatorRow {
    std::string rendererId;
    bool success = false;
    bool cancelled = false;
    std::string outputPath;
    std::string reportPath;
    std::string message;
    ReportMetrics metrics;
    double score = 0.0;
  };

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  const std::filesystem::path outDir(outDirArg);
  std::filesystem::create_directories(outDir);
  const auto stem = sanitizeFileName(session.sessionName.empty() ? "session" : session.sessionName);

  std::vector<ComparatorRow> rows;
  rows.reserve(rendererIds.size());

  for (const auto& rendererId : rendererIds) {
    auto settings = session.renderSettings;
    settings.rendererName = rendererId;
    if (formatOverride.has_value()) {
      settings.outputFormat = *formatOverride;
    }
    if (settings.outputFormat.empty() || settings.outputFormat == "auto") {
      settings.outputFormat = "wav";
    }
    if (externalBinaryArg.has_value()) {
      settings.externalRendererPath = *externalBinaryArg;
    }

    const auto outputName = stem + "_" + sanitizeFileName(rendererId) + extensionForFormat(settings.outputFormat);
    settings.outputPath = (outDir / outputName).string();

    ComparatorRow row;
    row.rendererId = rendererId;
    try {
      auto renderer = automix::renderers::createRenderer(rendererId);
      const auto result = renderer->render(session, settings, {}, nullptr);
      row.success = result.success;
      row.cancelled = result.cancelled;
      row.outputPath = result.outputAudioPath;
      row.reportPath = result.reportPath;
      row.message = result.logs.empty() ? "" : result.logs.back();

      std::filesystem::path reportCandidate(result.reportPath);
      if (reportCandidate.empty()) {
        reportCandidate = std::filesystem::path(settings.outputPath + ".report.json");
      }
      row.metrics = readReportMetrics(reportCandidate);
      row.score = row.success && row.metrics.loaded ? computeComparatorScore(row.metrics)
                                                    : (row.success ? 50.0 : 0.0);
    } catch (const std::exception& error) {
      row.success = false;
      row.message = error.what();
    }

    rows.push_back(row);
  }

  std::sort(rows.begin(), rows.end(), [](const ComparatorRow& a, const ComparatorRow& b) {
    if (a.success != b.success) {
      return a.success > b.success;
    }
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.rendererId < b.rendererId;
  });

  nlohmann::json ranking = nlohmann::json::array();
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows[i];
    ranking.push_back({
        {"rank", i + 1},
        {"rendererId", row.rendererId},
        {"success", row.success},
        {"cancelled", row.cancelled},
        {"score", row.score},
        {"outputPath", row.outputPath},
        {"reportPath", row.reportPath},
        {"message", row.message},
        {"metrics",
         {
             {"integratedLufs", row.metrics.integratedLufs},
             {"targetLufs", row.metrics.targetLufs},
             {"truePeakDbtp", row.metrics.truePeakDbtp},
             {"targetTruePeakDbtp", row.metrics.targetTruePeakDbtp},
             {"monoCorrelation", row.metrics.monoCorrelation},
             {"stereoCorrelation", row.metrics.stereoCorrelation},
             {"artifactRisk", row.metrics.artifactRisk},
         }},
    });
  }

  nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"sessionPath", *sessionPathArg},
      {"outDir", outDir.string()},
      {"renderers", rendererIds},
      {"ranking", ranking},
  };
  const auto jsonReportPath = outDir / "comparison_report.json";
  writeJsonFile(jsonReportPath, payload);

  const auto csvPath = outDir / "comparison_report.csv";
  std::ofstream csv(csvPath);
  if (csv.is_open()) {
    csv << "rank,renderer,success,cancelled,score,integrated_lufs,target_lufs,true_peak_dbtp,target_true_peak_dbtp,mono_corr,stereo_corr,artifact_risk,output_path,report_path,message\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto& row = rows[i];
      csv << (i + 1) << ","
          << csvEscape(row.rendererId) << ","
          << (row.success ? "true" : "false") << ","
          << (row.cancelled ? "true" : "false") << ","
          << row.score << ","
          << row.metrics.integratedLufs << ","
          << row.metrics.targetLufs << ","
          << row.metrics.truePeakDbtp << ","
          << row.metrics.targetTruePeakDbtp << ","
          << row.metrics.monoCorrelation << ","
          << row.metrics.stereoCorrelation << ","
          << row.metrics.artifactRisk << ","
          << csvEscape(row.outputPath) << ","
          << csvEscape(row.reportPath) << ","
          << csvEscape(row.message) << "\n";
    }
  }

  if (jsonOutput) {
    std::cout << payload.dump(2) << "\n";
  } else {
    std::cout << "Multi-render comparison complete. Ranking:\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto& row = rows[i];
      std::cout << "  " << (i + 1) << ". " << row.rendererId
                << " success=" << (row.success ? "yes" : "no")
                << " score=" << std::fixed << std::setprecision(2) << row.score
                << " LUFS=" << row.metrics.integratedLufs
                << " TP=" << row.metrics.truePeakDbtp << "\n";
    }
    std::cout << "JSON report: " << jsonReportPath.string() << "\n";
    std::cout << "CSV report: " << csvPath.string() << "\n";
  }

  const auto successes = std::count_if(rows.begin(), rows.end(), [](const ComparatorRow& row) { return row.success; });
  return successes > 0 ? 0 : 1;
}

int commandCatalogProcess(const CommandArgs& args) {
  const auto inputArg = argValue(args, "--input");
  const auto outputArg = argValue(args, "--output");
  if (!inputArg.has_value() || !outputArg.has_value()) {
    std::cerr << "catalog-process requires --input <folder> --output <folder>\n";
    return 2;
  }

  const std::filesystem::path inputDir(*inputArg);
  const std::filesystem::path outputDir(*outputArg);
  std::filesystem::create_directories(outputDir);

  const auto checkpointPath =
      std::filesystem::path(argValue(args, "--checkpoint").value_or((outputDir / "catalog_checkpoint.json").string()));
  const bool resume = hasFlag(args, "--resume");
  const auto csvPath =
      std::filesystem::path(argValue(args, "--csv").value_or((outputDir / "catalog_deliverables.csv").string()));
  const auto jsonPath =
      std::filesystem::path(argValue(args, "--json").value_or((outputDir / "catalog_deliverables.json").string()));

  automix::engine::BatchQueueRunner runner;
  auto discoveredItems = runner.buildItemsFromFolder(inputDir, outputDir);
  if (discoveredItems.empty()) {
    std::cout << "No audio items found in " << inputDir.string() << "\n";
    nlohmann::json emptyPayload = {
        {"generatedAtUtc", iso8601NowUtc()},
        {"inputDir", inputDir.string()},
        {"outputDir", outputDir.string()},
        {"summary", {{"total", 0}, {"completed", 0}, {"failed", 0}, {"cancelled", 0}}},
        {"items", nlohmann::json::array()},
    };
    writeJsonFile(jsonPath, emptyPayload);
    return 0;
  }

  std::unordered_map<std::string, nlohmann::json> checkpointBySession;
  if (resume) {
    if (const auto checkpoint = loadJsonFile(checkpointPath); checkpoint.has_value() && checkpoint->contains("items")) {
      for (const auto& item : checkpoint->at("items")) {
        if (!item.is_object()) {
          continue;
        }
        const auto sessionName = item.value("sessionName", "");
        if (!sessionName.empty()) {
          checkpointBySession[sessionName] = item;
        }
      }
    }
  }

  std::vector<automix::domain::BatchItem> completedFromCheckpoint;
  std::vector<automix::domain::BatchItem> pendingItems;
  completedFromCheckpoint.reserve(discoveredItems.size());
  pendingItems.reserve(discoveredItems.size());

  for (auto& item : discoveredItems) {
    const auto it = checkpointBySession.find(item.session.sessionName);
    if (it != checkpointBySession.end()) {
      const auto& checkpointItem = it->second;
      item.status = batchStatusFromString(checkpointItem.value("status", "pending"));
      item.error = checkpointItem.value("error", "");
      item.reportPath = checkpointItem.value("reportPath", "");
      if (checkpointItem.contains("outputPath")) {
        item.outputPath = checkpointItem.value("outputPath", item.outputPath.string());
      }
    }

    const bool checkpointCompleted =
        item.status == automix::domain::BatchItemStatus::Completed &&
        !item.reportPath.empty() &&
        std::filesystem::exists(item.reportPath);

    if (checkpointCompleted) {
      completedFromCheckpoint.push_back(item);
      continue;
    }

    item.status = automix::domain::BatchItemStatus::Pending;
    pendingItems.push_back(item);
  }

  automix::domain::BatchJob job;
  job.items = std::move(pendingItems);
  job.settings.outputFolder = outputDir;
  job.settings.renderSettings.rendererName = argValue(args, "--renderer").value_or("BuiltIn");
  job.settings.renderSettings.outputFormat = argValue(args, "--format").value_or("wav");
  job.settings.analysisThreads = parseIntArg(args, "--analysis-threads").value_or(1);
  job.settings.renderParallelism = parseIntArg(args, "--render-parallelism").value_or(1);
  job.settings.parallelAnalysis = !hasFlag(args, "--serial-analysis");

  std::atomic_bool cancelFlag {false};
  const auto processResult = runner.process(job, {}, &cancelFlag);

  std::vector<automix::domain::BatchItem> allItems = completedFromCheckpoint;
  allItems.insert(allItems.end(), job.items.begin(), job.items.end());
  std::sort(allItems.begin(), allItems.end(), [](const automix::domain::BatchItem& a, const automix::domain::BatchItem& b) {
    return a.session.sessionName < b.session.sessionName;
  });

  int completed = 0;
  int failed = 0;
  int cancelled = 0;
  nlohmann::json itemPayload = nlohmann::json::array();
  for (const auto& item : allItems) {
    completed += item.status == automix::domain::BatchItemStatus::Completed ? 1 : 0;
    failed += item.status == automix::domain::BatchItemStatus::Failed ? 1 : 0;
    cancelled += item.status == automix::domain::BatchItemStatus::Cancelled ? 1 : 0;

    const auto metrics = item.reportPath.empty() ? ReportMetrics{} : readReportMetrics(item.reportPath);
    itemPayload.push_back({
        {"sessionName", item.session.sessionName},
        {"status", automix::domain::toString(item.status)},
        {"outputPath", item.outputPath.string()},
        {"reportPath", item.reportPath},
        {"error", item.error},
        {"integratedLufs", metrics.integratedLufs},
        {"truePeakDbtp", metrics.truePeakDbtp},
        {"artifactRisk", metrics.artifactRisk},
    });
  }

  const nlohmann::json payload = {
      {"generatedAtUtc", iso8601NowUtc()},
      {"inputDir", inputDir.string()},
      {"outputDir", outputDir.string()},
      {"checkpoint", checkpointPath.string()},
      {"resumeEnabled", resume},
      {"processedInThisRun",
       {
           {"completed", processResult.completed},
           {"failed", processResult.failed},
           {"cancelled", processResult.cancelled},
       }},
      {"summary",
       {
           {"total", static_cast<int>(allItems.size())},
           {"completed", completed},
           {"failed", failed},
           {"cancelled", cancelled},
       }},
      {"items", itemPayload},
  };

  writeJsonFile(jsonPath, payload);
  writeJsonFile(checkpointPath, payload);

  std::ofstream csv(csvPath);
  if (csv.is_open()) {
    csv << "session_name,status,output_path,report_path,integrated_lufs,true_peak_dbtp,artifact_risk,error\n";
    for (const auto& item : allItems) {
      const auto metrics = item.reportPath.empty() ? ReportMetrics{} : readReportMetrics(item.reportPath);
      csv << csvEscape(item.session.sessionName) << ","
          << csvEscape(automix::domain::toString(item.status)) << ","
          << csvEscape(item.outputPath.string()) << ","
          << csvEscape(item.reportPath) << ","
          << metrics.integratedLufs << ","
          << metrics.truePeakDbtp << ","
          << metrics.artifactRisk << ","
          << csvEscape(item.error) << "\n";
    }
  }

  std::cout << "Catalog processing complete. total=" << allItems.size()
            << " completed=" << completed
            << " failed=" << failed
            << " cancelled=" << cancelled << "\n";
  std::cout << "Deliverables JSON: " << jsonPath.string() << "\n";
  std::cout << "Deliverables CSV: " << csvPath.string() << "\n";
  std::cout << "Checkpoint: " << checkpointPath.string() << "\n";

  return failed == 0 && cancelled == 0 ? 0 : 1;
}

int commandExportSegments(const CommandArgs& args) {
  const auto sessionPathArg = argValue(args, "--session");
  const auto outDirArg = argValue(args, "--out-dir");
  if (!sessionPathArg.has_value() || !outDirArg.has_value()) {
    std::cerr << "export-segments requires --session <session.json> and --out-dir <directory>\n";
    return 2;
  }

  double segmentSeconds = 5.0;
  if (const auto secondsArg = argValue(args, "--segment-seconds"); secondsArg.has_value()) {
    segmentSeconds = std::max(0.5, std::stod(*secondsArg));
  }

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  const std::filesystem::path outDir(*outDirArg);
  const std::filesystem::path stemDir = outDir / "stems";
  std::filesystem::create_directories(stemDir);

  automix::engine::AudioFileIO fileIO;
  automix::util::WavWriter writer;
  for (const auto& stem : session.stems) {
    if (!stem.enabled) {
      continue;
    }

    const auto buffer = fileIO.readAudioFile(stem.filePath);
    const int maxSamples = static_cast<int>(segmentSeconds * buffer.getSampleRate());
    const auto segment = sliceBuffer(buffer, maxSamples);
    const auto outPath = stemDir / (sanitizeFileName(stem.id + "_" + stem.name) + ".wav");
    writer.write(outPath, segment, 24);
  }

  automix::domain::RenderSettings settings = session.renderSettings;
  settings.outputSampleRate = settings.outputSampleRate > 0 ? settings.outputSampleRate : 44100;
  settings.blockSize = settings.blockSize > 0 ? settings.blockSize : 1024;
  settings.outputBitDepth = 24;
  settings.outputPath = (outDir / "mix_full.wav").string();

  automix::engine::OfflineRenderPipeline pipeline;
  const auto rawMix = pipeline.renderRawMix(session, settings, {}, nullptr);
  if (rawMix.cancelled) {
    std::cerr << "Raw mix render cancelled while exporting segments.\n";
    return 1;
  }

  const int maxMixSamples = static_cast<int>(segmentSeconds * rawMix.mixBuffer.getSampleRate());
  const auto mixSegment = sliceBuffer(rawMix.mixBuffer, maxMixSamples);
  writer.write(outDir / "mix_segment.wav", mixSegment, 24);

  nlohmann::json manifest = {
      {"sessionName", session.sessionName},
      {"segmentSeconds", segmentSeconds},
      {"stemCount", session.stems.size()},
      {"mixSegmentPath", (outDir / "mix_segment.wav").string()},
  };
  std::ofstream manifestOut(outDir / "manifest.json");
  manifestOut << manifest.dump(2);

  std::cout << "Exported stem and mix segments to " << outDir.string() << "\n";
  return 0;
}

int commandExportFeatures(const CommandArgs& args) {
  const auto sessionPathArg = argValue(args, "--session");
  const auto outPathArg = argValue(args, "--out");
  if (!sessionPathArg.has_value() || !outPathArg.has_value()) {
    std::cerr << "export-features requires --session <session.json> and --out <features.jsonl>\n";
    return 2;
  }
  const auto manifestPathArg = argValue(args, "--manifest");
  const auto datasetIdArg = argValue(args, "--dataset-id");
  const auto sourceTagArg = argValue(args, "--source-tag");
  const auto lineageParentsArg = argValue(args, "--lineage-parents");

  automix::engine::SessionRepository repository;
  const auto session = repository.load(*sessionPathArg);

  std::unordered_map<std::string, automix::domain::Stem> stemsById;
  for (const auto& stem : session.stems) {
    stemsById.emplace(stem.id, stem);
  }

  automix::analysis::StemAnalyzer analyzer;
  const auto entries = analyzer.analyzeSession(session);

  std::ofstream out(*outPathArg);
  if (!out.is_open()) {
    std::cerr << "Failed to open output file: " << *outPathArg << "\n";
    return 1;
  }

  for (const auto& entry : entries) {
    nlohmann::json line;
    line["sessionName"] = session.sessionName;
    line["stemId"] = entry.stemId;
    line["stemName"] = entry.stemName;
    line["metrics"] = {
        {"peakDb", entry.metrics.peakDb},
        {"rmsDb", entry.metrics.rmsDb},
        {"crestDb", entry.metrics.crestDb},
        {"crestFactor", entry.metrics.crestFactor},
        {"lowEnergy", entry.metrics.lowEnergy},
        {"midEnergy", entry.metrics.midEnergy},
        {"highEnergy", entry.metrics.highEnergy},
        {"silenceRatio", entry.metrics.silenceRatio},
        {"stereoCorrelation", entry.metrics.stereoCorrelation},
        {"stereoWidth", entry.metrics.stereoWidth},
        {"dcOffset", entry.metrics.dcOffset},
        {"subEnergy", entry.metrics.subEnergy},
        {"bassEnergy", entry.metrics.bassEnergy},
        {"lowMidEnergy", entry.metrics.lowMidEnergy},
        {"highMidEnergy", entry.metrics.highMidEnergy},
        {"presenceEnergy", entry.metrics.presenceEnergy},
        {"airEnergy", entry.metrics.airEnergy},
        {"spectralCentroidHz", entry.metrics.spectralCentroidHz},
        {"spectralSpreadHz", entry.metrics.spectralSpreadHz},
        {"spectralFlatness", entry.metrics.spectralFlatness},
        {"spectralFlux", entry.metrics.spectralFlux},
        {"onsetStrength", entry.metrics.onsetStrength},
        {"mfccCoefficients", entry.metrics.mfccCoefficients},
        {"constantQBins", entry.metrics.constantQBins},
        {"channelBalanceDb", entry.metrics.channelBalanceDb},
        {"artifactRisk", entry.metrics.artifactRisk},
        {"artifactSwirlRisk", entry.metrics.artifactProfile.swirlRisk},
        {"artifactSmearRisk", entry.metrics.artifactProfile.smearRisk},
        {"artifactNoiseDominance", entry.metrics.artifactProfile.noiseDominance},
        {"artifactHarmonicity", entry.metrics.artifactProfile.harmonicity},
        {"artifactPhaseInstability", entry.metrics.artifactProfile.phaseInstability},
    };

    const auto it = stemsById.find(entry.stemId);
    if (it != stemsById.end()) {
      line["origin"] = automix::domain::toString(it->second.origin);
      line["role"] = automix::domain::toString(it->second.role);
    }

    out << line.dump() << "\n";
  }

  std::cout << "Exported " << entries.size() << " feature rows to " << *outPathArg << "\n";

  if (manifestPathArg.has_value()) {
    const auto sessionText = readTextFile(*sessionPathArg).value_or("");
    const auto featureText = readTextFile(*outPathArg).value_or("");
    const auto lineageParents =
        lineageParentsArg.has_value() ? splitCommaSeparated(*lineageParentsArg) : std::vector<std::string>{};
    const auto datasetId = datasetIdArg.has_value() && !datasetIdArg->empty()
                               ? *datasetIdArg
                               : (sanitizeFileName(session.sessionName) + "_" + toHex(fnv1a64(iso8601NowUtc())));

    nlohmann::json manifest = {
        {"schemaVersion", 1},
        {"generatedAtUtc", iso8601NowUtc()},
        {"datasetId", datasetId},
        {"sourceTag", sourceTagArg.value_or("manual")},
        {"sourceSessionPath", *sessionPathArg},
        {"sessionName", session.sessionName},
        {"rowCount", entries.size()},
        {"featureSchemaVersion", "v1"},
        {"featureCountPerStem", automix::ai::FeatureSchemaV1::featureCount()},
        {"featureFilePath", *outPathArg},
        {"lineageParents", lineageParents},
        {"sessionHashFnv1a64", toHex(fnv1a64(sessionText))},
        {"featureFileHashFnv1a64", toHex(fnv1a64(featureText))},
        {"columns", nlohmann::json::array({
                        "peakDb", "rmsDb", "crestDb", "crestFactor", "lowEnergy", "midEnergy",
                        "highEnergy", "silenceRatio", "stereoCorrelation", "stereoWidth", "dcOffset",
                        "subEnergy", "bassEnergy", "lowMidEnergy", "highMidEnergy", "presenceEnergy",
                        "airEnergy", "spectralCentroidHz", "spectralSpreadHz", "spectralFlatness",
                        "spectralFlux", "onsetStrength", "mfccCoefficients", "constantQBins",
                        "channelBalanceDb", "artifactRisk", "artifactSwirlRisk", "artifactSmearRisk",
                        "artifactNoiseDominance", "artifactHarmonicity", "artifactPhaseInstability",
                    })},
    };

    writeJsonFile(*manifestPathArg, manifest);
    std::cout << "Wrote feature lineage manifest to " << *manifestPathArg << "\n";
  }

  return 0;
}

} // namespace

void registerRenderCommands(automix::devtools::CommandRegistry& registry) {
  registry.add("compare-renders", commandCompareRenders);
  registry.add("catalog-process", commandCatalogProcess);
  registry.add("export-segments", commandExportSegments);
  registry.add("export-features", commandExportFeatures);
}
