#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

FEATURES = [
    "rms_db",
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "artifact_risk",
]


def stem_to_vector(stem: dict) -> list[float]:
    return [
        float(stem.get("rmsDb", -120.0)),
        float(stem.get("lowEnergy", 0.0)),
        float(stem.get("midEnergy", 0.0)),
        float(stem.get("highEnergy", 0.0)),
        float(stem.get("artifactRisk", 0.0)),
    ]


def export_vectors(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stems = data.get("stems", [])
    vectors = []
    for stem in stems:
        vectors.append(
            {
                "stem_id": stem.get("stemId", ""),
                "stem_name": stem.get("stemName", ""),
                "features": stem_to_vector(stem),
                "schema_version": "1.0.0",
            }
        )

    out = {"feature_names": FEATURES, "items": vectors}
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
