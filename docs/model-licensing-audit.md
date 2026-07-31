# Model Hub Licensing Audit

Audit of every curated model available through the built-in Hugging Face model hub, plus the ITO-Master ONNX model being added for the mastering-assistant task.

- **Scope date**: 2026-07-31
- **Verification method**: Hugging Face model card API (`huggingface.co/api/models/<repo>`) checked live on the scope date for every flagged or uncertain entry. Unreachable or undeclared cards are marked unverified.
- **Machine-checkable twin**: `docs/model-licensing-audit.json` (schemaVersion 1). A static test (`tests/unit/AiExtensionTests.cpp`) fails if any `curatedModelIds()` entry lacks a manifest row.

## Status legend

| Mark | Meaning |
| :--- | :--- |
| ✅ | Clean: commercially usable without restriction |
| ⚠️ | Flagged: use restrictions (attribution required, non-commercial, or license unverified) |
| ❌ | Non-commercial: must not ship in a commercial product |

## Audit table

| Model id | Author | Verified license | Commercial status | Action |
| :--- | :--- | :--- | :--- | :--- |
| `rysertio/Demucs-onnx` | rysertio (Demucs port) | MIT | ✅ clean | None |
| `onnx-community/whisper-tiny.en` | onnx-community (OpenAI Whisper export) | MIT | ✅ clean | None |
| `onnx-community/whisper-small.en` | onnx-community (OpenAI Whisper export) | MIT | ✅ clean | None |
| `onnx-community/Speech-Emotion-Classification-ONNX` | onnx-community | ⚠️ unverified (no license declared on card; wav2vec2-based) | ⚠️ flagged | Verify license before commercial ship |
| `onnx-community/Musical-Instrument-Classification-ONNX` | onnx-community | MIT (declared on card, verified 2026-07-31) | ✅ clean | None |
| `onnx-community/Musical-genres-Classification-Hubert-V1-ONNX` | onnx-community | ⚠️ unverified (no license declared on card; HuBERT-derived, likely CC BY-NC 4.0) | ⚠️ flagged | Verify license before commercial ship |
| `openai/whisper-tiny` | OpenAI | MIT | ✅ clean | None |
| `laion/clap-htsat-unfused` | LAION | CC BY 4.0 (LAION research page; HF card currently lists apache-2.0) | ⚠️ flagged (attribution) | Attribute in credits; commercially usable with attribution |
| `pranjal-pravesh/PANNs_CNN14_ONNX` | pranjal-pravesh (PANNs port) | MIT | ✅ clean | None |
| `SonyCSLParis/music2latent` | Sony CSL Paris | CC BY-NC 4.0 | ⚠️ flagged | **Non-commercial.** Keep as user opt-in download; never bundle in a commercial installer |
| `StemSplitio/htdemucs-ft-onnx` | StemSplitio (Demucs 4-stem port) | MIT | ✅ clean | None |
| `StemSplitio/htdemucs-6s-onnx` | StemSplitio (Demucs 6-stem port) | MIT | ✅ clean | None |
| `kramp/ito-master-onnx` | kramp (ITO-Master port) | CC BY-NC 4.0 | ⚠️ flagged | **Non-commercial.** User opt-in download only (added for the mastering-assistant task); never bundle in a commercial installer |

## Summary

| Status | Count | Models |
| :--- | :--- | :--- |
| ✅ clean | 8 | Demucs-onnx, whisper-tiny.en, whisper-small.en, Musical-Instrument-Classification, openai/whisper-tiny, PANNs_CNN14_ONNX, htdemucs-ft-onnx, htdemucs-6s-onnx |
| ⚠️ flagged (attribution / unverified / non-commercial) | 5 | Speech-Emotion-Classification, Musical-genres-Classification-Hubert-V1, clap-htsat-unfused, music2latent, ito-master-onnx |
| ❌ non-commercial | 0 directly (2 of the 5 flagged are non-commercial: music2latent, ito-master-onnx) | |

## Notes and caveats

- **Never bundle NC weights in a commercial installer.** `music2latent` and `ito-master-onnx` are CC BY-NC 4.0. They must remain user-opt-in, on-demand downloads, never part of a distributed binary, installer, or bundle.
- **Unverified cards.** `onnx-community/Speech-Emotion-Classification-ONNX` and `onnx-community/Musical-genres-Classification-Hubert-V1-ONNX` declare no license in their HF card metadata. HuBERT-derived onnx-community re-distributions are typically CC BY-NC 4.0, but the source model card should be checked before any commercial use.
- **CLAP license discrepancy.** The LAION research repository states CC BY 4.0 for `clap-htsat-unfused`; the HF card metadata now lists apache-2.0. Both are attribution-permissive and commercially usable, so this does not change the commercial assessment, but the attribution requirement applies either way.
- **Demucs heritage.** The upstream Meta Demucs repository is MIT (code) with CC BY-NC 4.0 for some MUSDB18-trained weights. The specific ONNX ports curated here (`rysertio/Demucs-onnx`, `StemSplitio/htdemucs-ft-onnx`, `StemSplitio/htdemucs-6s-onnx`) are MIT-licensed exports.
- **Schema.** The machine-readable manifest `docs/model-licensing-audit.json` mirrors this table one-for-one. Adding a curated model without adding a manifest row fails the `[ai][licensing]` static test.
