# Changelog

## v1.0 — 2026-07-27

First tagged release. Adds Dr. Malaska's two-function trigger design, λmax pitch
mapping, and objective regression guards for the sound.

### Added

- **Event-driven sonification** (`--preset event`). A linear **trigger** decides
  which peaks sound, entirely separately from the **intensity encoding** that
  decides how loud — the design from Dr. Malaska's 2026-07-24 email.
  - `sonify/events.py`: `row_trigger_mask`, `apply_trigger`,
    `find_event_clusters`, `threshold_for_target_tones`
  - `--threshold`, `--trigger-type {linear,log}`, `--target-tones N`
  - `--target-tones` inverts his 2026-07-09 threshold study, which the test
    suite reproduces exactly (all 13 rows).
- **λmax pitch mapping** (`--tone-source lambda_max`). Each row is pitched by
  its dominant emission wavelength and collapsed to a single voice. This is the
  project plan's stated Phase 3 default and had never been implemented.
- **`--reverb-tail-ms`** — Dr. Malaska's "add a sustain component?" (2026-07-09),
  implemented as a *decaying* tail so notes ring out without merging into a drone.
- **`sonify/quality.py` + `scripts/analyze_audio.py`** — objective metrics
  (articulation, onset rate, mean amplitude, crest factor, spectral flatness).
- **`LICENSE`** (MIT). The README claimed MIT but no license file existed, so the
  project was not actually open-source.
- `docs/frequency_investigation.md` — answers Dr. Malaska's question about
  sonified frequencies absent from the source data.
- `docs/roadmap.md` — status against every plan phase, including blocked items.

### Fixed

- **Note articulation regression.** A "legato" envelope merged adjacent rows into
  sustained tones, collapsing the chime into a drone (199 onsets → 4;
  articulation 0.974 → 0.516). Reverted, and now guarded by a test.
- `spectral_flatness` counted digital silence, which is perfectly flat, so sparse
  event renders falsely scored ~0.99 ("noise"). Silent frames are now excluded.
- **Trigger precedence.** A preset's `--target-tones` overrode an explicitly
  given `--threshold`, so `--preset event --threshold 400/600/900` produced
  byte-identical audio. An explicit threshold now wins; the three settings
  correctly yield 50 / 25 / 10 events, matching Dr. Malaska's table.
- `apply_reverb_tail` was first written as a one-pole IIR over the waveform,
  which is a sub-hertz lowpass — it erased the audio instead of adding a tail.
  Now an exponentially decaying noise impulse response applied with FFT
  convolution (also O(n log n) rather than O(n·kernel), which had made a 160 s
  render hang).

### Changed

- **Synthesis vectorized.** The ADSR envelope is built once per channel instead
  of once per row per channel, the channel loop uses numpy broadcasting, and the
  mix runs in cache-friendly channel blocks. 2048 channels: ~30 → ~15 min.
  Output is pinned by golden-value tests and matches the previous implementation
  to ~1e-12 (float summation order only).
- CLI flags grouped into Core / Trigger / Audio / Mapping / Visual / Output.
- Comments labelled "Phase 5" retitled — that work was Dr. Malaska's July 9
  feedback, not the plan's Phase 5 (which is Dayana's derived data).
- 15 root-level planning documents moved to `docs/history/`.

### Removed

- **`--sustain`** — accepted a value, documented behaviour, and did nothing. The
  real feature is `--reverb-tail-ms`.
- `mapping.apply_threshold` — superseded by `events.apply_trigger`, which gates
  whole rows (correct for spectral events) rather than individual cells.
- Dead `normalize_per_channel` imports in `visualize.py` and `run_sonify.py`.
- `scripts/gen_spec.py`, `generate_spectrograms.py`,
  `generate_spectrogram_fixed.py` — replaced by `scripts/analyze_audio.py`.

### Repository

- Generated audio/video purged from git history (87 MB → ~10 MB). Spectrograms
  under `outputs/` and `diagnostics/outputs/` are deliberately kept: the
  presentation build embeds them.
- Dr. Malaska's threshold spreadsheet committed under `data/threshold/` as the
  provenance for the trigger defaults.

### Note on defaults

`chime`, `ambient`, and `scientific` are unchanged — `chime` output is verified
byte-identical to the previous release. Event mode ships as an opt-in preset so
Dr. Malaska can compare before any default changes.
