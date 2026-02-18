#include "domain/JsonSerialization.h"

#include <algorithm>

#include "domain/ProjectProfile.h"

namespace automix::domain {

void to_json(Json& j, const Stem& value) {
  j = Json{{"id", value.id},
           {"name", value.name},
           {"filePath", value.filePath},
           {"role", toString(value.role)},
           {"origin", toString(value.origin)},
           {"enabled", value.enabled}};

  if (value.busId.has_value()) {
    j["busId"] = value.busId.value();
  }
  if (value.separationConfidence.has_value()) {
    j["separationConfidence"] = value.separationConfidence.value();
  }
  if (value.separationArtifactRisk.has_value()) {
    j["separationArtifactRisk"] = value.separationArtifactRisk.value();
  }
}

void from_json(const Json& j, Stem& value) {
  value.id = j.value("id", "");
  value.name = j.value("name", "");
  value.filePath = j.value("filePath", "");
  value.role = stemRoleFromString(j.value("role", "unknown"));
  value.origin = stemOriginFromString(j.value("origin", "recorded"));
  value.enabled = j.value("enabled", true);

  if (j.contains("busId")) {
    value.busId = j.at("busId").get<std::string>();
  } else {
    value.busId.reset();
  }

  if (j.contains("separationConfidence") && !j.at("separationConfidence").is_null()) {
    value.separationConfidence = j.at("separationConfidence").get<double>();
  } else {
    value.separationConfidence.reset();
  }

  if (j.contains("separationArtifactRisk") && !j.at("separationArtifactRisk").is_null()) {
    value.separationArtifactRisk = j.at("separationArtifactRisk").get<double>();
  } else {
    value.separationArtifactRisk.reset();
  }
}

void to_json(Json& j, const Bus& value) {
  j = Json{{"id", value.id},
           {"name", value.name},
           {"type", toString(value.type)},
           {"gainDb", value.gainDb}};
}

void from_json(const Json& j, Bus& value) {
  value.id = j.value("id", "");
  value.name = j.value("name", "");
  value.type = busTypeFromString(j.value("type", "stem_group"));
  value.gainDb = j.value("gainDb", 0.0);
}

void to_json(Json& j, const RenderSettings& value) {
  j = Json{{"outputSampleRate", value.outputSampleRate},
           {"blockSize", value.blockSize},
           {"outputBitDepth", value.outputBitDepth},
           {"outputPath", value.outputPath},
           {"outputFormat", value.outputFormat},
           {"exportSpeedMode", value.exportSpeedMode},
           {"gpuExecutionProvider", value.gpuExecutionProvider},
           {"lossyBitrateKbps", value.lossyBitrateKbps},
           {"lossyQuality", value.lossyQuality},
           {"mp3UseVbr", value.mp3UseVbr},
           {"mp3VbrQuality", value.mp3VbrQuality},
           {"processingThreads", value.processingThreads},
           {"preferHardwareAcceleration", value.preferHardwareAcceleration},
           {"metadataPolicy", value.metadataPolicy},
           {"metadataTemplate", value.metadataTemplate},
           {"rendererName", value.rendererName},
           {"externalRendererPath", value.externalRendererPath},
           {"externalRendererTimeoutMs", value.externalRendererTimeoutMs}};
}

void from_json(const Json& j, RenderSettings& value) {
  value.outputSampleRate = j.value("outputSampleRate", 44100);
  value.blockSize = j.value("blockSize", 1024);
  value.outputBitDepth = j.value("outputBitDepth", 24);
  value.outputPath = j.value("outputPath", "");
  value.outputFormat = j.value("outputFormat", "auto");
  value.exportSpeedMode = j.value("exportSpeedMode", "final");
  if (value.exportSpeedMode != "final" &&
      value.exportSpeedMode != "balanced" &&
      value.exportSpeedMode != "quick") {
    value.exportSpeedMode = "final";
  }
  value.gpuExecutionProvider = j.value("gpuExecutionProvider", "auto");
  value.lossyBitrateKbps = std::clamp(j.value("lossyBitrateKbps", 320), 48, 512);
  value.lossyQuality = std::clamp(j.value("lossyQuality", 7), 0, 10);
  value.mp3UseVbr = j.value("mp3UseVbr", false);
  value.mp3VbrQuality = std::clamp(j.value("mp3VbrQuality", 4), 0, 9);
  value.processingThreads = std::max(0, j.value("processingThreads", 0));
  value.preferHardwareAcceleration = j.value("preferHardwareAcceleration", true);
  value.metadataPolicy = j.value("metadataPolicy", "copy_all");
  if (value.metadataPolicy != "copy_all" &&
      value.metadataPolicy != "copy_common" &&
      value.metadataPolicy != "copy_common_only" &&
      value.metadataPolicy != "strip" &&
      value.metadataPolicy != "override_template") {
    value.metadataPolicy = "copy_all";
  }
  value.metadataTemplate = j.value("metadataTemplate", std::map<std::string, std::string>{});
  value.rendererName = j.value("rendererName", "BuiltIn");
  value.externalRendererPath = j.value("externalRendererPath", "");
  value.externalRendererTimeoutMs = j.value("externalRendererTimeoutMs", 300000);
}

void to_json(Json& j, const StemMixDecision& value) {
  j = Json{{"stemId", value.stemId},
           {"gainDb", value.gainDb},
           {"pan", value.pan},
           {"highPassHz", value.highPassHz},
           {"mudCutDb", value.mudCutDb},
           {"enableCompressor", value.enableCompressor},
           {"compressorThresholdDb", value.compressorThresholdDb},
           {"compressorRatio", value.compressorRatio},
           {"compressorReleaseMs", value.compressorReleaseMs},
           {"enableExpander", value.enableExpander},
           {"expanderThresholdDb", value.expanderThresholdDb},
           {"expanderRatio", value.expanderRatio}};
}

void from_json(const Json& j, StemMixDecision& value) {
  value.stemId = j.value("stemId", "");
  value.gainDb = j.value("gainDb", 0.0);
  value.pan = j.value("pan", 0.0);
  value.highPassHz = j.value("highPassHz", 0.0);
  value.mudCutDb = j.value("mudCutDb", 0.0);
  value.enableCompressor = j.value("enableCompressor", false);
  value.compressorThresholdDb = j.value("compressorThresholdDb", -16.0);
  value.compressorRatio = j.value("compressorRatio", 3.0);
  value.compressorReleaseMs = j.value("compressorReleaseMs", 80.0);
  value.enableExpander = j.value("enableExpander", false);
  value.expanderThresholdDb = j.value("expanderThresholdDb", -45.0);
  value.expanderRatio = j.value("expanderRatio", 1.3);
}

void to_json(Json& j, const MixPlan& value) {
  j = Json{{"dryWet", value.dryWet},
           {"mixBusHeadroomDb", value.mixBusHeadroomDb},
           {"stemDecisions", value.stemDecisions},
           {"decisionLog", value.decisionLog}};
}

void from_json(const Json& j, MixPlan& value) {
  value.dryWet = j.value("dryWet", 1.0);
  value.mixBusHeadroomDb = j.value("mixBusHeadroomDb", 6.0);
  value.stemDecisions = j.value("stemDecisions", std::vector<StemMixDecision>{});
  value.decisionLog = j.value("decisionLog", std::vector<std::string>{});
}

void to_json(Json& j, const MultibandBandSettings& value) {
  j = Json{{"enabled", value.enabled},
           {"thresholdDb", value.thresholdDb},
           {"ratio", value.ratio},
           {"makeupGainDb", value.makeupGainDb},
           {"width", value.width}};
}

void from_json(const Json& j, MultibandBandSettings& value) {
  value.enabled = j.value("enabled", true);
  value.thresholdDb = j.value("thresholdDb", -18.0);
  value.ratio = j.value("ratio", 2.0);
  value.makeupGainDb = j.value("makeupGainDb", 0.0);
  value.width = j.value("width", 1.0);
}

void to_json(Json& j, const MultibandSettings& value) {
  j = Json{{"crossoverHz", value.crossoverHz},
           {"bands", value.bands},
           {"linearPhase", value.linearPhase}};
}

void from_json(const Json& j, MultibandSettings& value) {
  value.crossoverHz = j.value("crossoverHz", std::vector<double>{120.0, 500.0, 2000.0, 8000.0});
  value.bands = j.value("bands", std::vector<MultibandBandSettings>{
                                      MultibandBandSettings{},
                                      MultibandBandSettings{},
                                      MultibandBandSettings{},
                                      MultibandBandSettings{},
                                      MultibandBandSettings{},
                                  });
  value.linearPhase = j.value("linearPhase", false);
}

void to_json(Json& j, const MasterPlan& value) {
  j = Json{{"preset", toString(value.preset)},
           {"presetName", value.presetName},
           {"targetLufs", value.targetLufs},
           {"truePeakDbtp", value.truePeakDbtp},
           {"preGainDb", value.preGainDb},
           {"applyEq", value.applyEq},
           {"glueThresholdDb", value.glueThresholdDb},
           {"glueRatio", value.glueRatio},
           {"limiterCeilingDb", value.limiterCeilingDb},
           {"limiterTruePeakEnabled", value.limiterTruePeakEnabled},
           {"limiterLookaheadMs", value.limiterLookaheadMs},
           {"limiterAttackMs", value.limiterAttackMs},
           {"limiterReleaseMs", value.limiterReleaseMs},
           {"ditherBitDepth", value.ditherBitDepth},
           {"enableDeEsser", value.enableDeEsser},
           {"deEsserStrength", value.deEsserStrength},
           {"enableDeHarshEq", value.enableDeHarshEq},
           {"deHarshStrength", value.deHarshStrength},
           {"enableLowMono", value.enableLowMono},
           {"lowMonoHz", value.lowMonoHz},
           {"stereoWidth", value.stereoWidth},
           {"enableSoftClipper", value.enableSoftClipper},
           {"softClipDrive", value.softClipDrive},
           {"enableMultibandCompressor", value.enableMultibandCompressor},
           {"multibandSettings", value.multibandSettings},
           {"decisionLog", value.decisionLog}};
}

void from_json(const Json& j, MasterPlan& value) {
  value.preset = masterPresetFromString(j.value("preset", "default_streaming"));
  value.presetName = j.value("presetName", "DefaultStreaming");
  value.targetLufs = j.value("targetLufs", -14.0);
  value.truePeakDbtp = j.value("truePeakDbtp", -1.0);
  value.preGainDb = j.value("preGainDb", 0.0);
  value.applyEq = j.value("applyEq", true);
  value.glueThresholdDb = j.value("glueThresholdDb", -18.0);
  value.glueRatio = j.value("glueRatio", 2.0);
  value.limiterCeilingDb = j.value("limiterCeilingDb", -1.0);
  value.limiterTruePeakEnabled = j.value("limiterTruePeakEnabled", true);
  value.limiterLookaheadMs = j.value("limiterLookaheadMs", 7.0);
  value.limiterAttackMs = j.value("limiterAttackMs", 1.0);
  value.limiterReleaseMs = j.value("limiterReleaseMs", 80.0);
  value.ditherBitDepth = j.value("ditherBitDepth", 24);
  value.enableDeEsser = j.value("enableDeEsser", false);
  value.deEsserStrength = j.value("deEsserStrength", 0.35);
  value.enableDeHarshEq = j.value("enableDeHarshEq", false);
  value.deHarshStrength = j.value("deHarshStrength", 0.30);
  value.enableLowMono = j.value("enableLowMono", false);
  value.lowMonoHz = j.value("lowMonoHz", 120.0);
  value.stereoWidth = j.value("stereoWidth", 1.0);
  value.enableSoftClipper = j.value("enableSoftClipper", false);
  value.softClipDrive = j.value("softClipDrive", 1.15);
  value.enableMultibandCompressor = j.value("enableMultibandCompressor", false);
  value.multibandSettings = j.value("multibandSettings", MultibandSettings{});
  value.decisionLog = j.value("decisionLog", std::vector<std::string>{});
}

void to_json(Json& j, const Session& value) {
  j = Json{{"schemaVersion", value.schemaVersion},
           {"sessionName", value.sessionName},
           {"residualBlend", value.residualBlend},
           {"stems", value.stems},
           {"buses", value.buses},
           {"renderSettings", value.renderSettings},
           {"projectProfileId", value.projectProfileId},
           {"safetyPolicyId", value.safetyPolicyId},
           {"preferredStemCount", value.preferredStemCount},
           {"timeline",
            Json{
                {"loopEnabled", value.timeline.loopEnabled},
                {"loopInSeconds", value.timeline.loopInSeconds},
                {"loopOutSeconds", value.timeline.loopOutSeconds},
                {"zoom", value.timeline.zoom},
                {"fineScrub", value.timeline.fineScrub},
            }}};

  if (value.originalMixPath.has_value()) {
    j["originalMixPath"] = value.originalMixPath.value();
  }

  if (value.mixPlan.has_value()) {
    j["mixPlan"] = value.mixPlan.value();
  }
  if (value.masterPlan.has_value()) {
    j["masterPlan"] = value.masterPlan.value();
  }
}

void from_json(const Json& j, Session& value) {
  value.schemaVersion = j.value("schemaVersion", 2);
  value.sessionName = j.value("sessionName", "Untitled Session");
  value.residualBlend = std::clamp(j.value("residualBlend", 0.0), 0.0, 10.0);
  value.stems = j.value("stems", std::vector<Stem>{});
  value.buses = j.value("buses", std::vector<Bus>{});
  value.projectProfileId = j.value("projectProfileId", "default");
  value.safetyPolicyId = j.value("safetyPolicyId", "balanced");
  value.preferredStemCount = std::clamp(j.value("preferredStemCount", 4), kMinPreferredStemCount, kMaxPreferredStemCount);

  const auto timelineJson = j.value("timeline", Json::object());
  value.timeline.loopEnabled = timelineJson.value("loopEnabled", false);
  value.timeline.loopInSeconds = std::max(0.0, timelineJson.value("loopInSeconds", 0.0));
  value.timeline.loopOutSeconds = std::max(0.0, timelineJson.value("loopOutSeconds", 0.0));
  value.timeline.zoom = std::clamp(timelineJson.value("zoom", 1.0), 1.0, 64.0);
  value.timeline.fineScrub = timelineJson.value("fineScrub", false);
  if (value.timeline.loopOutSeconds <= value.timeline.loopInSeconds) {
    value.timeline.loopEnabled = false;
  }

  if (j.contains("originalMixPath") && !j.at("originalMixPath").is_null()) {
    value.originalMixPath = j.at("originalMixPath").get<std::string>();
  } else {
    value.originalMixPath.reset();
  }

  if (j.contains("renderSettings")) {
    value.renderSettings = j.at("renderSettings").get<RenderSettings>();
  } else {
    value.renderSettings = RenderSettings{};
  }

  if (j.contains("mixPlan") && !j.at("mixPlan").is_null()) {
    value.mixPlan = j.at("mixPlan").get<MixPlan>();
  } else {
    value.mixPlan.reset();
  }

  if (j.contains("masterPlan") && !j.at("masterPlan").is_null()) {
    value.masterPlan = j.at("masterPlan").get<MasterPlan>();
  } else {
    value.masterPlan.reset();
  }
}

} // namespace automix::domain
