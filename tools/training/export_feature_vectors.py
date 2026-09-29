#!/usr/bin/env python3
"""Export AutoMixMaster feature vectors from an analysis JSON file.

The vector layout is owned by ``tools/training/feature_schema_v1.json`` and must
stay compatible with the runtime ``automix::ai::FeatureSchemaV1`` extractor
(66 floats per stem, schema version 1.0.0). Names are read from that file rather
than hardcoded, so a schema bump cannot silently desynchronise this tool from
the app. See ``README.md`` in this directory.
"""
import argparse
import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("feature_schema_v1.json")

# Maps each schema feature name to the key the analysis export uses. The two
# vocabularies are NOT a mechanical snake_case -> camelCase conversion: for
# example ``low_energy_ratio`` is exported as ``lowEnergy``, not
# ``lowEnergyRatio``. Scalars map to a string key, ``mfcc_*`` indexes into
# ``mfccCoefficients`` and ``cqt_*`` indexes into ``constantQBins``.
SCALAR_KEYS = {
    "rms_db": "rmsDb",
    "peak_db": "peakDb",
    "crest_db": "crestDb",
    "dc_offset": "dcOffset",
    "low_energy_ratio": "lowEnergy",
    "mid_energy_ratio": "midEnergy",
    "high_energy_ratio": "highEnergy",
    "sub_energy_ratio": "subEnergy",
    "bass_energy_ratio": "bassEnergy",
    "low_mid_energy_ratio": "lowMidEnergy",
    "high_mid_energy_ratio": "highMidEnergy",
    "presence_energy_ratio": "presenceEnergy",
    "air_energy_ratio": "airEnergy",
    "spectral_centroid_hz": "spectralCentroidHz",
    "spectral_spread_hz": "spectralSpreadHz",
    "spectral_flatness": "spectralFlatness",
    "spectral_flux": "spectralFlux",
    "silence_ratio": "silenceRatio",
    "stereo_correlation": "stereoCorrelation",
    "stereo_width": "stereoWidth",
    "channel_balance_db": "channelBalanceDb",
    "artifact_risk": "artifactRisk",
    "artifact_swirl_risk": "artifactSwirlRisk",
    "artifact_smear_risk": "artifactSmearRisk",
    "artifact_noise_dominance": "artifactNoiseDominance",
    "artifact_harmonicity": "artifactHarmonicity",
    "artifact_phase_instability": "artifactPhaseInstability",
    "crest_factor": "crestFactor",
    "onset_strength": "onsetStrength",
}

MFCC_ARRAY_KEY = "mfccCoefficients"
CQT_ARRAY_KEY = "constantQBins"
MFCC_COUNT = 13
CQT_COUNT = 24


def load_schema() -> tuple[list[str], str]:
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    return list(schema["features"]), str(schema["version"])


def _indexed(name: str, prefix: str, count: int):
    if not name.startswith(prefix + "_"):
        return None
    suffix = name[len(prefix) + 1 :]
    if not suffix.isdigit():
        return None
    index = int(suffix)
    return index if 0 <= index < count else None


def _value(raw) -> float:
    # A missing metric is not an error: the runtime extractor also defaults
    # absent values to 0.0, so exporting a zero keeps the two in step.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    return float(raw)


def _value_at(raw, index: int) -> float:
    if not isinstance(raw, list) or index >= len(raw):
        return 0.0
    return _value(raw[index])


def stem_to_vector(stem: dict, features: list[str]) -> list[float]:
    vector: list[float] = []
    for name in features:
        mfcc_index = _indexed(name, "mfcc", MFCC_COUNT)
        if mfcc_index is not None:
            vector.append(_value_at(stem.get(MFCC_ARRAY_KEY), mfcc_index))
            continue
        cqt_index = _indexed(name, "cqt", CQT_COUNT)
        if cqt_index is not None:
            vector.append(_value_at(stem.get(CQT_ARRAY_KEY), cqt_index))
            continue
        key = SCALAR_KEYS.get(name)
        vector.append(_value(stem.get(key) if key else None))
    return vector


def export_vectors(input_path: Path, output_path: Path) -> None:
    features, schema_version = load_schema()
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stems = data.get("stems", [])
    vectors = []
    for stem in stems:
        vectors.append(
            {
                "stem_id": stem.get("stemId", ""),
                "stem_name": stem.get("stemName", ""),
                "features": stem_to_vector(stem, features),
                "schema_version": schema_version,
            }
        )

    out = {"feature_names": features, "schema_version": schema_version, "items": vectors}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AutoMixMaster feature vectors from analysis JSON.")
    parser.add_argument("--input", required=True, help="Input analysis JSON path")
    parser.add_argument("--output", required=True, help="Output feature JSON path")
    args = parser.parse_args()

    export_vectors(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
