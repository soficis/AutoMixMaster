# ITO-Master AI Mastering Route — Validation

**Task:** T3.8 — ITO-Master mastering stage (native white-box FX chain + 46-param adapter).
**Status:** experimental, NON-DEFAULT (off unless `AUTOMIX_ITO_MASTER=1` or the app enables the toggle).
**Invariants:** the sacred default chain (`RendererPipeline.cpp`) and `ModelStrategy::applyOverrides`
are untouched — the ITO route is an *additional* `IAutoMasterStrategy` implementation.

## 1. What the route is

ITO-Master (Koo et al., Sony Research) is an AI mastering model. The hub pack
`kramp/ito-master-onnx` (curated id, T3.5) carries **three artifacts** consumed
as one model pack under `task: mastering-assistant`:

| Artifact | Role |
| --- | --- |
| `fxencoder.onnx` | reference audio `[1,2,N]` → style embedding `[1,2048]` |
| `mastering_tcn.onnx` | audio `[1,2,N]` + embedding `[1,2048]` → 46 normalized params `[1,46]` in [0,1] |
| `config.json` | the **static contract**: param order, per-param min/max, tensor shapes, fx order |

The mastering stage is **white-box**: the 46 normalized params are denormalized
(`value = norm*(max-min)+min`) and mapped onto a native DSP chain
(`src/dsp/ItoMasterFxChain.*`): EQ ×18 (JUCE biquads: low shelf + 4 peaking +
high shelf), distortion ×2 (tanh + parallel), multiband compressor ×21
(Linkwitz-Riley LR4 crossovers + per-band comp/expander), gain ×1, imager ×1
(M/S width), limiter ×3 (lookahead ceiling limiter). Total 46 params, matching
`config.json` exactly.

## 2. Gating (all three required, else heuristic fallback)

1. **Experimental toggle ON** — default OFF. Opt-in via env `AUTOMIX_ITO_MASTER=1`
   or `ItoMasterStrategy::setExperimentalEnabled(true)`.
2. **License consent** — CC BY-NC 4.0 gate (T3.5): the app injects
   `ModelController::hasModelLicenseConsent("huggingface:kramp/ito-master-onnx")`
   as `ItoMasterStrategy::Options::licenseConsented`.
3. **Complete pack on disk** — all three artifacts present (the hub install
   downloads `mastering_tcn.onnx` + `config.json` alongside the primary
   `fxencoder.onnx`; the pack schema now carries `auxiliary_files`).

When any gate fails, `buildPlan`/`applyPlan` defer to `HeuristicAutoMasterStrategy`
with a decisionLog reason. **The default mastering path is byte-identical.**

## 3. Native ONNX Runtime status (this build)

`AUTOMIX_HAS_NATIVE_ORT` is **OFF** in this build → `OnnxModelInference` runs the
deterministic fallback (no real tensors). Consequences, by design:

- **Tensor-shape validation happens against the static contract parsed from
  `config.json`** (`ItoMasterAdapter::validateTensorContract`): encoder
  `[1,2,N]→[1,2048]`, predictor `[1,2,N]+[1,2048]→[1,46]`.
- The model runner (`ItoMasterModelRunner`) still loads both ONNX artifacts
  through `OnnxModelInference` (EP chain + recovery are real) and attempts the
  forward pass; the deterministic fallback emits no 2048-dim embedding and no
  46-param tensor, so `predict()` returns `nullopt` — **no false pass**, and the
  strategy falls back with a logged reason.
- Documented integration seam: `OnnxModelInference` accepts a **single** input
  tensor, while `mastering_tcn.onnx` declares two inputs. In a future
  ORT-enabled build the encoder stage (single input) runs natively; the
  two-input predictor remains outside the single-input adapter contract and is
  documented in `src/ai/ItoMasterAdapter.h`. No change to `OnnxModelInference.*`
  was required for this task.

## 4. How to validate (golden track set)

1. Install the pack in the Model Browser (accepts the CC BY-NC consent), or drop
   the artifacts manually into
   `assets/modelhub/huggingface_kramp__ito-master-onnx/`.
2. Run with `AUTOMIX_ITO_MASTER=1 AutoMixMaster` so the experimental route is on.
3. Run Auto Master with the ITO pack set as active for `master`, then repeat with
   the heuristic path (toggle off). Compare:

| Metric | Source | Criterion |
| --- | --- | --- |
| Integrated LUFS | report (`integratedLufs`) | within 0.5 LU of target, both routes |
| Sample peak dBFS | report (`samplePeakDbfs`) | ≤ limiter ceiling + 0.1 dB |
| True peak dBTP | report (`truePeakDbtp`) | ≤ ceiling + 0.2 dBTP |
| Loudness range (LRA) | report (`loudnessRange`) | ITO ≤ heuristic +2 LU (tonal shaping) |
| Spectrum | plot before/after per route | ITO EQ bands track config (low shelf/4 bands/high shelf); no >3 dB HF lift beyond params |
| DR | DR meter on both masters | ITO within ±2 of heuristic on loud material |

### Manual listening checklist

- [ ] Low end: tight, no mud at the `low_cutoff`/low-shelf settings; mono below 120 Hz intact.
- [ ] Mids: vocal/snare clarity without sibilance buildup (band1/band2 gains).
- [ ] Highs: cymbal air present, no harshness (band3/high-shelf, distortion drive).
- [ ] Transients: limiter `at` short enough — no pumping; `rt` release natural.
- [ ] Stereo: imager `width` — widening is tasteful, mono-compatible (correlation ≥ +0.6).
- [ ] Distortion: `drive_db` adds weight on bus/mix but never audible grit on quiet passages.
- [ ] Loudness: target LUFS hit, ceiling never exceeded, no clipping artefacts.

## 5. Automated tests (registered in CMakeLists `automix_tests`)

| Test file | Covers |
| --- | --- |
| `ItoMasterAdapterTests.cpp` | (1) tensor-shape smoke `[1,2,N]→[1,2048]→[1,46]` vs static config contract (live under `AUTOMIX_HAS_NATIVE_ORT`, contract-assert otherwise); (2) denorm round-trip norm 0/1 → min/max + clamp; (4) 46-param → typed settings, all in bounds; runner no-false-pass on deterministic fallback |
| `ItoMasterFxChainTests.cpp` | (3) biquad peak gain ≈ setting at centre freq; compressor GR under overdrive; limiter ceiling ≤ ceiling+ε; LR4 crossover sum ≈ unity magnitude; imager/chain sanity |
| `ItoMasterStrategyTests.cpp` | gating (toggle/consent/pack), static-contract validation + heuristic fallback, metadata surface, (5) integration quality (model-gated — skips cleanly when the pack is absent) |
| `AiExtensionTests.cpp` (append) | pack schema carries `auxiliary_files`; incomplete packs rejected |

## 6. Files

**New:** `src/dsp/ItoMasterFxChain.{h,cpp}`, `src/ai/ItoMasterAdapter.{h,cpp}`,
`src/automaster/ItoMasterStrategy.{h,cpp}`, `tests/unit/ItoMasterAdapterTests.cpp`,
`tests/unit/ItoMasterFxChainTests.cpp`, `tests/unit/ItoMasterStrategyTests.cpp`,
`tests/unit/ito_master_fixture.h`, this document.

**Edited:** `CMakeLists.txt` (source + test registration only),
`src/ai/ModelPackLoader.{h,cpp}` (`ModelPack::auxiliaryFiles`),
`src/ai/HuggingFaceModelHub.{h,cpp}` (download auxiliary artifacts at install),
`src/ai/ModelCatalogValidator.cpp` (manifest `auxiliary_files` + ITO `intended_use`),
`tests/unit/AiExtensionTests.cpp` (pack-schema test, append only).

**Untouched (byte-identical to d5b6c10):** `src/renderers/RendererPipeline.cpp`,
`src/ai/ModelStrategy.cpp`, `src/app/ui/MainLayout.cpp`, `src/app/ui/TransportBar.cpp`,
`src/app/ui/ModelBrowserPanel.cpp`, `src/ai/OnnxModelInference.{h,cpp}`.
