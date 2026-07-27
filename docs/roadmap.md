# Roadmap — status against the YSP project plan

Tracks Dr. Malaska's *YSP Borehole Sonification Project student project plan*
(updated 2026-06-25) plus his emailed feedback. Items listed as blocked are
recorded here so they are not silently dropped.

## Complete

| Plan item | Notes |
|---|---|
| **Phase 1** — V0 sonification script | CSV load, volume / bins / tone range / playback speed, log10-ln-linear switches, band auto-detection with user confirmation, WAV export |
| **Phase 2** — visual display | Dots/circles, user display options, video output |
| **Phase 3** — parameter switching | `--tone-source`, `--intensity-source`, and the plan's stated default **tone = λmax** via `--tone-source lambda_max` |
| **Phase 4** — standalone V1 | Trail display with position line, minimap overview, arbitrary column counts, `--output-name`, grouped CLI help |
| Display feedback (2026-07-09) | Larger pixels (`--marker-size`), square pixels (`--marker-shape`), scale (`--show-colorbar`) |
| Auto-gain (2026-07-09) | 8 modes: max / 90th percentile / median / mean, each linear or log |
| User-set playback speed (2026-07-09) | Interactive prompt plus `--playback-speed` |
| "Add a sustain component?" (2026-07-09) | `--reverb-tail-ms` — a *decaying* tail, so notes ring out without merging into a drone |
| Two-function design (2026-07-24) | Trigger (`--threshold` / `--trigger-type` / `--target-tones`) separated from intensity encoding (`--scale` / `--gain-mode`) |
| Threshold study (2026-07-09) | `sonify/events.py` reproduces the spreadsheet exactly; `--target-tones` inverts it |
| "Frequencies not in the dataset" (2026-07-24) | Investigated and answered — see [frequency_investigation.md](frequency_investigation.md) |

## Partially complete

**Phase 3 — up to 2048 channels.** Detection and rendering handle 2048 channels;
synthesis is vectorized (≈2× faster) but a full 4000-row × 2048-channel render
still takes ~15 minutes. The cost is inherent — 3.6 × 10¹⁰ oscillator samples.
Making this interactive needs a different algorithm (inverse-FFT additive
synthesis), not more optimization. Realistic channel counts (8–32) render in
under 30 seconds.

## Blocked — needs input

**Other borehole datasets / the "glommed" dataset** (Phases 4 and 6).
The engine is dataset-agnostic and ready; we need the files from Dr. Malaska.

**Phase 5 — derived data sonification.** Dayana is working in her own codebase,
and no derived tables exist yet. When they land, Dr. Malaska's "Solution B"
(parallel tables, one per parameter) is the lower-risk option, and it is
*partly* served already: `--tone-source column` and `--intensity-source column`
read any per-row column. What is missing is a sound dimension for **FWHM** and
**symmetry** — candidates are timbre brightness (filter cutoff) for FWHM and
stereo pan or slight detune for skew. Needs a design decision once real data
exists.

**Baseline-shift "dark regions"** (2026-07-24). Dr. Malaska assigned the
smoothing/levelling routine to the Dayana project.

**Phase 6 pre-processing expansion scripts.** Overlaps Dayana's
baseline-subtraction work; needs direction on ownership.

## Not started — non-code

- Consulting external sonification experts (Phase 6)
- Museum contacts and public release (Phase 6)
- End-of-YSP presentations and talks

## Open question for Dr. Malaska

The dataset documentation describes a descent from 89.225 m to **95.879 m**, but
`depth` in the CSV runs 89.217 m to **93.879 m**. Likely a typo in the
documentation, but worth confirming that the file is not truncated.
