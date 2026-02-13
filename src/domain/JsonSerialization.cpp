#include "domain/JsonSerialization.h"

#include <algorithm>

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
           {"lossyBitrateKbps", value.lossyBitrateKbps},
           {"lossyQuality", value.lossyQuality},
           {"processingThreads", value.processingThreads},
           {"preferHardwareAcceleration", value.preferHardwareAcceleration},
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
  value.lossyBitrateKbps = std::clamp(j.value("lossyBitrateKbps", 192), 48, 512);
  value.lossyQuality = std::clamp(j.value("lossyQuality", 7), 0, 10);
  value.processingThreads = std::max(0, j.value("processingThreads", 0));
  value.preferHardwareAcceleration = j.value("preferHardwareAcceleration", true);
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
  value.decisionLog = j.value("decisionLog", std::vector<std::string>{});
}

void to_json(Json& j, const Session& value) {
  j = Json{{"schemaVersion", value.schemaVersion},
           {"sessionName", value.sessionName},
           {"residualBlend", value.residualBlend},
           {"stems", value.stems},
           {"buses", value.buses},
           {"renderSettings", value.renderSettings}};

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
