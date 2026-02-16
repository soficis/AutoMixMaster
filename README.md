# AutoMixMaster

AutoMixMaster is a JUCE/CMake desktop app for deterministic stem mixing and mastering:

`Analysis -> MixPlan -> MasterPlan -> Renderer -> Audio + JSON report`

This branch implements the **v2 roadmap** in `docs/roadmap.md` and follow-up Phase V/Future Suggestion work, including an automated Hugging Face model browser/downloader, metadata policy profiles, and expanded automation APIs/tooling.

## Since Last Published `ai_dev` Commit

Compared to published `origin/ai_dev` commit `a6b7173`, this working tree adds:

1. Automatic AI model browser + downloader (Hugging Face)
- New app `Models` menu with:
  - `Browse & Download Models`
  - `Installed Models`
  - `Check Updates`
  - `Integrity & Licenses`
  - `Open Model Hub Folder`
- New `src/ai/HuggingFaceModelHub.*` service for discovery, model-info fetch, revision-pinned install, install registry/logging, and update checks.
- Discovery is dynamic (no fixed manual catalog), focused on proven music-audio model families (`demucs`, `mdx23c`, `bs-roformer`, `mel-band-roformer`, `open-unmix`, `clap`, `panns`, `basic-pitch`).

2. Secure Hugging Face token handling
- Gated/private access supports env-based token resolution:
  - `AUTOMIX_HF_TOKEN`
  - `HF_TOKEN`
  - `HUGGINGFACE_TOKEN`
  - `HUGGINGFACE_HUB_TOKEN`
- Tokens are not persisted in config files/logs; install metadata stores only `tokenUsed` boolean.

3. Future suggestions implemented end-to-end
- Profile sharing simplified to import/export workflows (`profile-export`, `profile-import`).
- Adaptive fix-chain assistant (`adaptive-assistant`).
- Guided collaboration review (`session-review`).
- Batch Studio remote API (`batch-studio-api` launcher + `tools/batch_studio_api.py`).
- Continuous eval trend automation (`eval-trend`) and nightly CI workflow.
- Metadata policy profiles fully wired through domain, renderer pipeline, and reports.

4. Metadata policy profile system
- Added profile/session/render settings fields:
  - `metadataPolicy`
  - `metadataTemplate`
- Implemented policy engine with `copy_all`, `copy_common`/`copy_common_only`, `strip`, and `override_template`.
- Built-in and external renderer paths now apply policy before writing exports.

5. CI + test coverage updates
- Added nightly trend workflow: `.github/workflows/nightly_golden_eval.yml`.
- Added metadata policy unit tests and expanded profile/session serialization coverage.

6. Non-functional repository normalization
- Large repository-wide line-ending/style normalization touched many files (source, config, tests, and bundled PhaseLimiter license/resource files).
- Effective logic changes remain concentrated in the files listed above; most other touched files are formatting/EOL-only churn.

## Implemented Scope (v2)

1. Native ONNX runtime sessions (Phase G)
- `OnnxModelInference` supports real ONNX Runtime C++ sessions when `ENABLE_ONNX=ON` and runtime headers/libs are present.
- Provider-aware tuning matrix (CPU/CUDA/DirectML/CoreML), thread controls, and optional profiling artifact export.
- Deterministic adapter fallback remains available when native runtime is unavailable.

2. Advanced separation model workflow (Phase H)
- Multi-variant separation packs (2/4/6 stem targets) with automatic selection and override support.
- Chunked overlap-add processing with GPU memory budget and stream scheduling controls.
- Separation QA report bundle with `energyLeakage`, `residualDistortion`, and `transientRetention`.

3. Precision UX and timeline state (Phase I)
- Loop in/out markers, looped transport playback, timeline zoom, and fine-scrub behavior.
- Timeline loop/zoom/fine-scrub state persists in session JSON.
- Task center panel with timestamped background-task history.

4. External DSP marketplace foundation (Phase J)
- External renderer descriptor signature metadata and trust policy loading.
- Signed/unsigned policy handling and trusted-signer filtering.
- Capability snapshot artifacts written for discovered external renderers.

5. Evaluation/training operations (Phase K)
- Golden corpus evaluator command backed by regression harness baselines.
- Heuristic vs model plan diff report generation.
- Dataset lineage/manifest output for feature export workflows.

## Product Suggestions Implemented

1. Profile Sharing (import/export)
- Simplified marketplace scope to direct sharing workflows.
- Added `automix_dev_tools profile-export` and `profile-import`.

2. Adaptive Assistant
- Added `automix_dev_tools adaptive-assistant` to generate fix chains from stem health + optional comparator context.

3. Continuous Evaluation
- Added `automix_dev_tools eval-trend`.
- Added nightly CI workflow for golden-eval trend artifacts.

4. Batch Studio API
- Added `tools/batch_studio_api.py` (`/health`, `/v1/catalog/process`, `/v1/reports/ingest`).
- Added launcher command `automix_dev_tools batch-studio-api`.

5. Guided Collaboration Review
- Added `automix_dev_tools session-review` with semantic highlights over session diffs.

6. Metadata Policy Profiles
- Added profile-level metadata policy + template support through serialization, profiles, and renderers.

7. Asynchronous Mastering Engine
- Long-running render/master tasks run in worker threads with cancellation checkpoints and adaptive chunk sizing.

## Build

### Configure + build (Linux/WSL/macOS)

```bash
cmake -S . -B build-codex -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_TOOLS=ON
cmake --build build-codex --parallel
```

When `ENABLE_ONNX=ON`, CMake attempts to locate native ONNX Runtime headers/library and enables native session support automatically when found.

### Configure + build (Windows, VS 2026)

Windows Visual Studio builds are pinned to the VS 2026 generator (`Visual Studio 18 2026`).

```powershell
cmake -S . -B build_win_vs2026_release -G "Visual Studio 18 2026" -A x64 -DBUILD_TESTING=OFF -DBUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build_win_vs2026_release --config Release --parallel
```

Built executable:
- `build_win_vs2026_release/AutoMixMasterApp_artefacts/Release/AutoMixMaster.exe`

## Run

### App

```bash
./build-codex/AutoMixMasterApp_artefacts/AutoMixMaster
```

### Tests

```bash
ctest --test-dir build-codex --output-on-failure -j4
```

As of **February 15, 2026**, this branch passes **58/58 tests**.
Coverage includes project profile loading, renderer trust-policy enforcement, transport loop behavior, separator variant/QA outputs, and session schema round-trip fields.

## GUI Workflow

1. Import stems (or one mixed track with `AI-separated stems` enabled).
2. Optional: load original mix for A/B and residual blend.
3. Select project profile, AI packs, and ML provider (`auto/cpu/directml/coreml/cuda`).
4. Run `Auto Mix`.
5. Run `Auto Master` and choose master/platform preset.
6. Use transport/loop/zoom/fine-scrub for preview.
7. Review task center and stem health diagnostics.
8. Export single song or batch folder (`Quick` mode for speed, strict safety policy can block non-pinned renderers).

Outputs:
- `<name>.<ext>`
- `<name>.<ext>.report.json`

## Session Schema Additions

- `renderSettings`: `exportSpeedMode`, `mp3UseVbr`, `mp3VbrQuality`
- `timeline`: `loopEnabled`, `loopInSeconds`, `loopOutSeconds`, `zoom`, `fineScrub`
- `session`: `projectProfileId`, `safetyPolicyId`, `preferredStemCount`

## Renderers

- `BuiltIn`: always available in-process renderer.
- `PhaseLimiter`: external binary discovery with fallback-safe behavior.
- Descriptor-driven external renderers from:
  - `assets/limiters/**/renderer.json`
  - `assets/renderers/**/renderer.json`
  - `Assets/Limiters/**/renderer.json`

Trust policy candidates:
- `assets/renderers/trust_policy.json`
- `assets/limiters/trust_policy.json`

Discovery side-effect:
- External renderer validation writes capability snapshots next to binaries as `<rendererId>.capabilities.snapshot.json`.

## Codec Availability

Runtime export codec probing is surfaced in the GUI.

Supported format targets:
- Lossless: `wav`, `aiff`, `flac`
- Lossy: `mp3`, `ogg`

MP3 export modes:
- `CBR`: target bitrate via `lossyBitrateKbps` (default `320 kbps`)
- `VBR`: quality ladder via `mp3VbrQuality` (0 best .. 9 smallest)

Export speed modes:
- `Final`: default quality path (`blockSize=1024`, `24-bit`)
- `Balanced`: moderate throughput (`blockSize=2048`, `24-bit`)
- `Quick`: fast-turnaround preset (`blockSize=4096`, `16-bit`) with default codec target `MP3 VBR`
  - Stem-health preflight is skipped in Quick mode to reduce turnaround time.
  - If MP3 is unavailable at runtime, Quick mode falls back to the first available codec.

MP3 fallback order:
- JUCE MP3/LAME writer path (when available)
- External `lame` binary path discovery (`LAME_BIN`, app-adjacent, `PATH`)
- Optional on-demand LAME downloader cache

Metadata retention:
- Exports preserve source metadata from the original mix (or first valid stem fallback when original mix is not set).
- External LAME MP3 export maps common metadata to ID3 tags (`title`, `artist`, `album`, `year`, `track`, `genre`, `comment`).

## Developer Tools (`automix_dev_tools`)

Built when `-DBUILD_TOOLS=ON`.

```bash
automix_dev_tools export-features --session <session.json> --out <features.jsonl> [--manifest <manifest.json>] [--dataset-id <id>] [--source-tag <tag>] [--lineage-parents <id,id,...>]
automix_dev_tools export-segments --session <session.json> --out-dir <dir> [--segment-seconds <sec>]
automix_dev_tools validate-modelpack --pack <modelpack_dir>
automix_dev_tools validate-external-limiter --binary <path> [--json]
automix_dev_tools stem-health --session <session.json> [--out <path>] [--json]
automix_dev_tools compare-renders --session <session.json> [--renderers <id,id,...>] [--out-dir <dir>] [--format <fmt>] [--external-binary <path>] [--json]
automix_dev_tools catalog-process --input <folder> --output <folder> [--checkpoint <path>] [--resume] [--renderer <id>] [--format <fmt>] [--analysis-threads <n>] [--render-parallelism <n>] [--csv <path>] [--json <path>]
automix_dev_tools session-diff --base <session.json> --head <session.json> [--out <patch.json>] [--summary]
automix_dev_tools session-merge --base <session.json> --left <session.json> --right <session.json> --out <session.json> [--prefer <left|right>] [--report <report.json>] [--json]
automix_dev_tools profile-export --out <profiles.json> [--id <profile_id>]
automix_dev_tools profile-import --in <profiles.json> [--out <assets/profiles/project_profiles.json>]
automix_dev_tools adaptive-assistant --session <session.json> [--compare-report <comparison_report.json>] [--out <fixes.json>] [--json]
automix_dev_tools session-review --base <session.json> --head <session.json> [--out <review.json>] [--json]
automix_dev_tools model-browse [--limit <n>] [--token-env <ENV_VAR>] [--out <catalog.json>] [--json]
automix_dev_tools model-install --repo <org/model> [--dest <assets/modelhub>] [--token-env <ENV_VAR>] [--force] [--out <report.json>] [--json]
automix_dev_tools model-health [--root <assets/modelhub>] [--out <report.json>] [--json]
automix_dev_tools external-limiter-compat --binary <path> [--timeout-ms <ms>] [--required-features <f1,f2>] [--out <report.json>] [--json]
automix_dev_tools golden-eval [--baseline <baselines.json>] [--work-dir <dir>] [--out <report.json>] [--json]
automix_dev_tools eval-trend [--baseline <baselines.json>] [--work-dir <dir>] [--trend <trend.json>] [--out <summary.json>] [--json]
automix_dev_tools plan-diff --session <session.json> [--mix-model <path>] [--master-model <path>] [--out <report.json>] [--json]
automix_dev_tools batch-studio-api [--host <ip>] [--port <n>] [--automix-bin <path>] [--output-root <dir>] [--api-key <key>]
automix_dev_tools list-supported-models
automix_dev_tools install-supported-model --id <model_id> [--dest <assets/models>]
automix_dev_tools list-supported-limiters
automix_dev_tools install-supported-limiter --id <limiter_id> [--dest <assets/limiters>]
automix_dev_tools install-lame-fallback [--force] [--json]
```

## External Limiter Contract

Schema:
- `docs/external_limiter_contract.schema.json`

Validation taxonomy:
- `binary_missing`
- `launch_failed`
- `timeout`
- `exit_code`
- `invalid_json`
- `missing_version`
- `missing_supported_features`
- `schema_incompatible`

## AI Model Packs

`ModelManager` scans:
- configured roots
- `assets/models`
- `assets/modelpacks`
- `assets/ModelPacks`
- `Assets/ModelPacks`
- `AUTOMIX_MODELPACK_PATHS`

Runtime specialization metadata supported:
- `preferredPrecision`
- `providerAffinity`
- `defaultIntraOpThreads`
- `defaultInterOpThreads`
- `enableProfiling`

## Hugging Face Model Hub

Model browser/downloader installs into `assets/modelhub` and tracks:
- per-model metadata: `<install-dir>/modelhub.json`
- install registry: `assets/modelhub/install_registry.json`
- append-only install log: `assets/modelhub/install_log.jsonl`

Token support for gated models:
- `AUTOMIX_HF_TOKEN`
- `HF_TOKEN`
- `HUGGINGFACE_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

Security notes:
- Tokens are read from environment variables at runtime.
- Raw token values are not written to disk by AutoMixMaster.
- Model metadata records only whether a token was used (`tokenUsed`).

## Key CMake Options

- `BUILD_TESTING=ON|OFF`
- `BUILD_TOOLS=ON|OFF`
- `ENABLE_ONNX=ON|OFF`
- `ENABLE_RTNEURAL=ON|OFF`
- `RTNEURAL_XSIMD=ON|OFF`
- `RTNEURAL_USE_AVX=ON|OFF`
- `ENABLE_LIBEBUR128=ON|OFF`
- `ENABLE_PHASELIMITER=ON|OFF`
- `ENABLE_EXTERNAL_TOOL_SUPPORT=ON|OFF`
- `ENABLE_BUNDLED_LAME=ON|OFF`
- `DISTRIBUTION_MODE=OSS|PROPRIETARY`

## Known Limits

- Native ONNX sessions require external ONNX Runtime headers/library at configure time; otherwise the deterministic adapter path is used.
- Signature verification currently uses descriptor-embedded `fnv1a64` checks (policy and signer lists are data-driven but not PKI-backed yet).
- Collaboration merge is deterministic and semantic for core session structures, but not yet CRDT-based real-time merge.

## License

AutoMixMaster is GPLv3 in OSS mode (`DISTRIBUTION_MODE=OSS`).

Third-party dependencies include JUCE, nlohmann/json, Catch2, and libebur128 under their respective licenses.
