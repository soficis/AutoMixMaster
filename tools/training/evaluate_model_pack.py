#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED = [
    "id",
    "type",
    "engine",
    "model_file",
    "license",
    "source",
    "feature_schema_version",
]


def evaluate(pack_dir: Path) -> dict:
    model_json = pack_dir / "model.json"
    if not model_json.exists():
        return {"valid": False, "error": "model.json missing"}

    with model_json.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    missing = [key for key in REQUIRED if not metadata.get(key)]
    model_path = pack_dir / metadata.get("model_file", "")

    valid = not missing and model_path.exists()
    return {
        "valid": valid,
        "missing_fields": missing,
        "model_exists": model_path.exists(),
        "pack_id": metadata.get("id", ""),
        "schema_version": metadata.get("feature_schema_version", ""),
        "license": metadata.get("license", ""),
        "source": metadata.get("source", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize an AutoMixMaster model pack.")
    parser.add_argument("--pack", required=True, help="Path to model pack folder")
    parser.add_argument("--output", required=False, help="Optional output JSON path")
    args = parser.parse_args()

    report = evaluate(Path(args.pack))
    text = json.dumps(report, indent=2)
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
