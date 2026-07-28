# Changelog

## v1.1 — 2026-07-28

Fixes the harshness that made the output unpleasant to listen to. Roughness on
the `chime` preset dropped **0.371 -> 0.049 (-87%)**, `ambient` **1.031 -> 0.066
(-94%)**, both measured on rendered audio.

### The problem

The v1.0 quality guards (`articulation`, `onset_rate`) measure whether notes are
separated in *time*. They say nothing about whether notes sounding *together*
clash — a dense, harsh cluster chord scores a perfect 0.998 articulation. A
Sethares/Plomp-Levelt analysis found:

- **Density.** 74% of rows sounded 5+ voices and 23.8% sounded all eight; only
  1.4% sounded a single voice. A real wind chime strikes one tube at a time.
- **Timbre, not tuning.** The same pentatonic pitches as pure sines measure 0.179
  dissonance; the actual preset measured 0.649 — worse than a major triad.
- **The mechanism.** `bell`'s 2x/3x/4x harmonics landed 55-75 Hz from *other*
  channels' fundamentals (550 Hz against 495 Hz), the peak of the Plomp-Levelt
  roughness curve. This was 66% of all measured dissonance; the inharmonic
  `chime` partial was only 9%.

### Added

- **`--max-voices N`** (`mapping.limit_voices`) — per row, only the loudest N
  channels sound. The single biggest fix. Retains ~64% of per-row amplitude, and
  the loudest channel varies across all eight and changes on 82% of row
  transitions, so dominant-band information survives.
- **`soft` timbre** (fundamental + quiet octave) replacing `bell` in the low
  partition group, so overtones no longer sit in a neighbour's critical band.
- **`--preset chime-legacy`** — reproduces the exact pre-tuning rendering, so
  anything Dr. Malaska has already heard stays available.
- **Harmonic quality metrics** in `sonify/quality.py`: `sensory_dissonance`,
  `spectral_roughness`, `polyphony`. `analyze_audio.py` reports roughness, and
  `tests/test_quality_dissonance.py` guards it — each guard also asserts the
  untuned configuration fails, so it has teeth.
- **Advisory** when the ADSR envelope outlasts a row by 50%+, which piles notes
  up (how `ambient` was silently degraded).

### Changed

- `chime`: `max_voices=3`. **Pitches deliberately unchanged.** Raising the root
  or widening octaves scored better on the partial-spectrum model, but that is
  partly an artefact of critical bands widening with frequency: on real audio it
  bought only 0.049 -> 0.044 while pushing 15% of energy above 3 kHz, which is
  piercing and breaches the toolkit's own 2500 Hz ceiling.
- `ambient`: `max_voices=3`, and `playback_speed=5.0` — it is documented as
  "slow, meditative" but was inheriting the 10 rows/s global default, where its
  `slow` envelope piled notes up (roughness 1.031 vs 0.066 at 5 rows/s).
- `scientific` is **deliberately exempt** from voice limiting, so a fully
  faithful all-channel rendering still exists.
- Synthesis golden values regenerated for the two *partitioned* cases only. The
  three non-partitioned cases are bit-identical, confirming the synthesis core
  did not drift.

### Removed

- **`--reverb-tail-ms`**, replaced by **`--tail-ms`** (`mapping.apply_envelope_tail`).
  The old implementation convolved the mix with a decaying-*noise* impulse
  response — a room model, not a sustain. It smeared every note across the
  spectrum (roughness 1.80 vs 0.71 for an envelope tail on the same material).

### Known limitation

`--tail-ms` defaults to **0 (off) everywhere**, including `event`. A
forward-decaying amplitude tail currently sounds like tremolo rather than
sustain, because the ADSR still retriggers on every row: an 800 ms tail at
25 rows/s is 20 re-attacks, and measured roughness rose from 0.44 to 16.6.
Dr. Malaska's sustain request needs note-level envelopes that span rows — a
synthesis architecture change tracked in `docs/roadmap.md`. Shipping it on by
default would have made the audio worse.


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
- `tests/test_config.py` — `SonificationConfig.validate()` had no test file at
  all, despite gating every CLI input. Covers the trigger and reverb rules added
  here plus the pre-existing range checks.

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

- Generated audio/video purged from git history (87 MB → 12 MB). Spectrograms
  under `outputs/` and `diagnostics/outputs/` are deliberately kept: the
  presentation build embeds them, so the floor is the 3.2 MB source dataset plus
  ~5 MB of referenced figures.
- Dr. Malaska's threshold spreadsheet committed under `data/threshold/` as the
  provenance for the trigger defaults.

### Note on defaults

`chime`, `ambient`, and `scientific` are unchanged — `chime` output is verified
byte-identical to the previous release. Event mode ships as an opt-in preset so
Dr. Malaska can compare before any default changes.
