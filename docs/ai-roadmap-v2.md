# AI Roadmap v2.x: Reference-Match Mastering (Deferred)

Status: decided, v1.1 does not ship it. This file records the decision and the v2.x plan so the work is not re-litigated during the v1.1 cycle.

## Decision

Full reference-match mastering in the style of [sergree/matchering](https://github.com/sergree/matchering) is **deferred to v2.x**. v1.1 keeps the existing bounded soft-target reference blend (`OriginalMixReference::applySoftTarget`) already wired into the external-limiter path and does not introduce a new reference-matching DSP subsystem.

## Why deferred

- **Licensing.** sergree/matchering is GPL-3.0. Linking it into this application would force the whole binary under GPL, which is off the table for a commercial desktop app. A clean-room reimplementation of the *technique* is legal, but...
- **Scope.** A clean-room reference-matching engine is a new DSP subsystem: RMS/FR/peak/stereo-width matching, an optimization loop, and its own test surface. It competes with v1.1 scope rather than contributing to it.
- **v1.1 is already full.** The release carries the P0/P1 backlog, realtime audio, external renderers, and ITO-Master. Adding a second mastering philosophy now risks both.

## What v1.1 ships instead

The existing `OriginalMixReference` soft-target path stays as the reference-blend mechanism.

### Location

- `src/automaster/OriginalMixReference.h` / `.cpp`: the `applySoftTarget` implementation.
- `src/renderers/ExternalLimiterRenderer.cpp` (lines 286-299): integration point. When no session master plan exists and `session.originalMixPath` is set, the renderer loads the reference, resamples it to the mix rate if needed, and applies the soft target before writing the external-limiter request. Failures are caught and logged as "Original mix reference skipped", never fatal.

### Behavior (the contract the regression tests lock)

`applySoftTarget(basePlan, stemMix, originalMix, strategy, analyzer)` returns a new `MasterPlan`. It is a pure function of its inputs: bounded, deterministic, and channel-count agnostic.

1. **Guards.** Empty `stemMix` or `originalMix`, or a sample-rate mismatch, returns the base plan with a single decision-log entry (skip path).
2. **Loudness blend.** `targetLufs` becomes `0.75 * current + 0.25 * reference`, clamped to `[-30, -8]` dB LUFS.
3. **Gain.** `preGainDb += 0.35 * (referenceLUFS - mixLUFS)`, clamped to `[-9, 9]` dB.
4. **Dynamics.** `glueRatio` is nudged by the crest-factor delta, clamped to `[1.1, 6.0]`.
5. **Tone.** If the spectral-tilt delta (high minus low energy) between reference and mix exceeds 0.06, `applyEq` is enabled.
6. **Trace.** The applied path pushes three decision-log entries (applied marker, reference/mix LUFS, adjusted target).

Because every adjusted field is clamped and every measurement is a deterministic function of the audio, the output stays in bounds and repeated calls with identical inputs produce identical plans. A mono reference is downmixed during analysis, so channel-count mismatch is handled without crashing or tripping the skip guards.

## v2.x roadmap

### (a) Full reference-match mastering, clean-room

Reimplement the *documented* matchering technique from scratch: RMS and frequency-response matching, true-peak limiting, and stereo-width matching. The technique and its published description are not copyrightable; the GPL-3.0 license covers matchering's *code*, not the algorithm it implements. A clean-room implementation written from the technique description (no code derived from the repository) keeps the app under a permissive license.

Acceptance for v2.x: reference-match renders from arbitrary input/reference pairs with an optimization budget bounded in real time, plus the full unit/integration/E2E ladder.

### (b) MIT training-seed path for a mastering-parameter predictor

Seed candidates, all MIT-licensed with released weights:

- `sony/FxNorm-automix` (audio normalizer, deep learning, ONNX-ready)
- `jhtonyKoo/music_mixing_style_transfer` (style-transfer mixing)
- `barry-mir/stemfx` (per-stem FX parameter prediction)

Plan: fine-tune or train a bespoke mastering-parameter predictor from the seeds and export to ONNX for inference at export time.

Each seed is revisited against three criteria before adoption:

1. **ONNX export.** The trained model must export to ONNX without runtime dependency on the source framework.
2. **Permissive license.** MIT (or compatible) for both code and weights; no GPL/CC-BY-NC contamination in the training data path.
3. **`applyOverrides` contract.** The predictor's output must map onto the existing mastering-parameter surface (the fields `applyOverrides` already mutates) so the runtime path stays shared with the heuristic and soft-target paths rather than forking a new one.

If a seed fails a criterion, it is dropped and the remaining seeds are re-evaluated. No seed enters v1.1.
