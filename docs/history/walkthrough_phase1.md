# Phase 1 — Implementation Walkthrough

## What was built

The complete Phase 1 sonification toolkit per the revised implementation plan. **11 components**, **39 unit tests**, all passing. **10 tuning WAVs** generated covering every parameter combination and code path.

### Engine modules (`sonify/`)

| Module | Purpose |
|--------|---------|
| [config.py](file:///d:/Freelancing/YSP/sonify/config.py) | `SonificationConfig` dataclass with all CLI params + `validate()` |
| [data_io.py](file:///d:/Freelancing/YSP/sonify/data_io.py) | Generic `load_csv()` — no knowledge of band naming |
| [band_detect.py](file:///d:/Freelancing/YSP/sonify/band_detect.py) | Auto-detect band columns via regex, exclude noise columns, return `(names, indices)` |
| [preprocess.py](file:///d:/Freelancing/YSP/sonify/preprocess.py) | `clean()` (NaN->0, neg->0), `sort_by_row_order()`, `rebin()`, `rebin_wavelengths()` with shared `_contiguous_groups()` |
| [mapping.py](file:///d:/Freelancing/YSP/sonify/mapping.py) | `scale_values()` (no re-clip), `normalize_per_channel()`, `assign_frequencies()` (index + wavelength), `load_wavelength_table()` |
| [synth.py](file:///d:/Freelancing/YSP/sonify/synth.py) | Phase-continuous additive synthesis with capped raised-cosine fades |
| [playback.py](file:///d:/Freelancing/YSP/sonify/playback.py) | `sounddevice` -> `simpleaudio` fallback chain with clear error messages |
| [export.py](file:///d:/Freelancing/YSP/sonify/export.py) | WAV export with float->int16 conversion |

### Additional changes

- **Unicode fix**: Replaced `->`, `x`, `-` for all print statements in [run_sonify.py](file:///d:/Freelancing/YSP/scripts/run_sonify.py) — Windows cp1252 consoles can't encode `\u2192`, `\u00d7`, `\u2013`

---

## Verification

### Automated tests: 39/39 passed
```
tests/test_band_detect.py    7 passed
tests/test_mapping.py       14 passed
tests/test_preprocess.py    12 passed
tests/test_synth.py          5 passed (incl. WAV round-trip)
```

### Fade-cap implementation verified

The code at [synth.py:55](file:///d:/Freelancing/YSP/sonify/synth.py#L55) uses `segment_samples // 4` (integer floor division), matching the revised plan. At 40 rows/sec:

- `segment_samples = round(1/40 * 44100) = round(1102.5) = 1102` (Python banker's rounding)
- `segment_samples // 4 = 1102 // 4 = 275`
- `default_fade = round(0.010 * 44100) = 441`
- `fade_samples = min(441, 275) = 275`

So 275 samples fade on each side, leaving 552 samples (50.1%) at full amplitude — comfortably above the "at least half" guarantee.

### Generated tuning WAVs

| File | Params | Duration |
|------|--------|----------|
| `borehole_phase1.wav` | 200 rows, log10, index | 20.0s |
| `speed5.wav` | 4000 rows, 5 rps | 800.0s |
| `speed20.wav` | 4000 rows, 20 rps | 200.0s |
| `speed40.wav` | 4000 rows, 40 rps, 25ms/row | 100.0s |
| `linear.wav` | linear scale, full dataset | 400.0s |
| `log10_full.wav` | **log10 scale, full dataset** | 400.0s |
| `ln.wav` | ln scale, full dataset | 400.0s |
| `wavelength.wav` | wavelength mode, 32ch | 400.0s |
| `8bins.wav` | 8 bins, index mode | 400.0s |
| `wavelength_8bins.wav` | **wavelength + rebin combo** | 400.0s |

The three-way scale comparison is now apples-to-apples: `linear.wav`, `log10_full.wav`, and `ln.wav` are all the full 4000 rows at 10 rps.

### Playback fallback chain: logic verified, audio not yet

Ran without `--output` — synthesis completed (2.5s waveform), then the fallback chain correctly tried `sounddevice` (not in the `py` environment), tried `simpleaudio` (also not present), and raised a clear `RuntimeError` with install instructions. **This verifies the fallback logic and error messaging, but not that sound actually comes out of speakers.** To confirm real audio playback, run from the activated venv where `sounddevice` is installed:

```bash
.venv\Scripts\activate
python scripts/run_sonify.py --row-end 50 --playback-speed 20 --yes
```

### What to listen for in the tuning set

1. **`speed40.wav`** — tones should still be audible, not just fade ramps. The fade-cap gives 275 samples (6.2ms) per side, leaving 552 samples (12.5ms) at full amplitude per 25ms segment.
2. **`linear.wav` vs `log10_full.wav` vs `ln.wav`** — clean three-way comparison at identical duration. Log scaling should reveal significantly more low-level structure given the dataset's wide intensity range (0 to ~300+).
3. **`wavelength_8bins.wav`** — should sound like smoothly ordered low-to-high tones, not scrambled. Rebinned wavelengths (283.3-437.7 nm) confirmed to match hand-calculated group averages.
4. **Any file** — no audible clicks/pops at row boundaries (phase continuity + fades).

### Remaining manual step

Listen to the WAVs, pick the best `--playback-speed`, `--scale`, and `--freq-mode` defaults, and write those choices + reasoning into the README per PHASE1_SPEC.md §9. That's a human judgment call, not an implementation task.
