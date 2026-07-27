# Where the "frequencies that do not correspond to the original dataset" come from

**In response to Dr. Malaska's 2026-07-24 note:**

> "However, there are also sonified frequencies that do not correspond to the original
> dataset. It might be worth seeing where those come from."

**Short answer: you were right, and they are our doing, not the ice's.** In the `chime`
preset, **97% of the spectral peaks appear even when the data is replaced by a constant
value.** They are produced by the sonification's own design, not by the fluorescence.

A previous reply from this project claimed these lines were "just the synthesized bell/chime
partials, not a bug." That answer was incomplete — partials alone account for only 17 of 34
measured peaks. The full accounting is below.

---

## Method

All numbers are reproducible from the repository:

```bash
py scripts/analyze_audio.py outputs/chime_full.wav --spectrogram
```

Spectral peaks were extracted with Welch's method (`nperseg=16384`, ≈2.7 Hz resolution) and
`find_peaks(prominence=3 dB)` over 0–3000 Hz.

---

## Finding 1 — Pitch is fixed, so most of the spectrum is predetermined

The `chime` preset rebins to 8 channels and assigns each a **fixed pentatonic pitch**:

```
220.0  275.0  366.7  495.0  660.0  880.0  1100.0  1467.0 Hz
```

These never move. The data only changes how *loud* each one is, never *which* pitch sounds.
So the spectrum is essentially the same picture regardless of what the instrument recorded.

## Finding 2 — Timbre adds overtones that are not in the data

`timbre_partition` splits the 8 channels into three groups with different voices:

| Group | Voice | Partials | Overtones added |
|---|---|---|---|
| 0 (low) | bell | 1×, 2×, 3×, 4× | harmonics of 220 / 275 / 366.7 |
| 1 (mid) | chime | 1×, 2.756× | 1364, 1819, 2425 Hz |
| 2 (high) | sine | 1× | none |

The 2.756× ratio is the tubular-bell inharmonic overtone. It is a deliberate musical choice —
but it puts energy at frequencies with no counterpart in the spectrometer output.

## Finding 3 — The per-row envelope creates sidebands

Retriggering an amplitude envelope once per row amplitude-modulates every carrier, which
places sidebands at `carrier ± n × (row rate)`. Verified by synthesizing a single 1100 Hz
sine at constant amplitude and measuring the spacing:

| Playback speed | Predicted spacing | Measured |
|---|---|---|
| 5 rows/s | 5.0 Hz | 4.71 Hz |
| 8 rows/s | 8.0 Hz | 8.07 Hz |
| 10 rows/s | 10.0 Hz | 10.09 Hz |
| 20 rows/s | 20.0 Hz | 20.19 Hz |

This is why `chime_full.wav` (5 rows/s) shows clusters of closely spaced lines straddling each
fundamental — for example around 1100 Hz: 1015, 1039, 1066, **1101**, 1136, 1160, 1184, 1214.
Pure additive synthesis cannot invent partials, so the envelope is the only possible source.

## Finding 4 — The decisive test

Render the same preset twice: once with the real borehole amplitudes, once with every value
replaced by a single constant.

| Render | Peaks below 3 kHz |
|---|---|
| Real data amplitudes | 36 |
| Constant amplitude | 40 |
| **Shared (present regardless of data)** | **35** |
| Present only with real data | 1 |

**97% of the spectrum is independent of the data.** The chime preset is closer to a fixed
instrument being played at varying volume than to a picture of the fluorescence.

---

## What we changed

The new **`--preset event`** implements the two-function design from your same email — a
linear trigger deciding *which* peaks sound, separately from a log intensity encoding deciding
*how loud* — and additionally makes **pitch data-driven** via `--tone-source lambda_max`: each
event is pitched by its own dominant emission wavelength (the Phase 3 default from the project
plan, which had never been implemented).

Re-running the decisive test on event mode, this time shuffling the *spectral shape* while
holding intensities fixed:

| Render | Peaks below 3 kHz |
|---|---|
| Real spectra | 47 |
| Shuffled spectra | 53 |
| **Peaks that change when the spectrum changes** | **78 of 89 (88%)** |

So the picture inverts: **97% data-independent in `chime`, 88% data-responsive in `event`.**

```bash
# Event-driven, pitch follows lambda max, ~25 tones (your "probably above 10")
py scripts/run_sonify.py --yes --preset event --output outputs/event.wav
```

Worked example — the recurring 385.3 nm feature (Band 21) always lands on the same 880 Hz
note, so that fluorophore becomes audibly recognizable:

| Depth (m) | Peak band | λmax (nm) | Pitch (Hz) |
|---|---|---|---|
| 89.435 | 10 | 324.6 | 330.0 |
| 89.432 | 21 | 385.3 | 880.0 |
| 89.572 | 21 | 385.3 | 880.0 |
| 89.623 | 8 | 313.7 | 275.0 |
| 90.157 | 19 | 374.3 | 733.5 |
| 91.557 | 21 | 385.3 | 880.0 |

---

## Caveat

Event mode still snaps pitches to a pentatonic scale, so pitch is *quantized* — it reports
which fluorophore dominates, not an exact wavelength. Scale snapping can be disabled with
`--freq-mode index` if you want a continuous wavelength → pitch mapping instead. There is a
real trade-off here (musicality vs. spectral precision) and it is worth your call which one
the default should be.
