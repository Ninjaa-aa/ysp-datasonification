# Phase 1 — Borehole Sonification Toolkit Implementation Plan (Revised)

## Goal

Build a complete, generic, dataset-agnostic sonification pipeline per [PHASE1_SPEC.md](file:///d:/Freelancing/YSP/PHASE1_SPEC.md). The ice borehole dataset is the test case, not a hardcoded dependency. The pipeline reads any multi-channel CSV, maps channels → frequencies, values → amplitudes, rows → time, and outputs either live audio or a `.wav` file.

**Revision note:** this version resolves a normalization contradiction in the prior draft (it described both per-channel and global normalization for the same function), fixes wavelength-mode frequency mapping breaking under rebinning, removes a redundant clipping step, caps the synthesis fade length so it can't exceed a row's segment duration at high playback speeds, and restores the `sounddevice` → `simpleaudio` playback fallback. Changes are marked **[FIXED]** inline.

---

## Proposed Changes

### Component 1: Configuration — `sonify/config.py`

#### [NEW] [config.py](file:///d:/Freelancing/YSP/sonify/config.py)

- `SonificationConfig` dataclass with all user-tunable parameters:
  - `input_path: str`
  - `row_start: int | None = None`, `row_end: int | None = None`
  - `n_bins: int | None = None` (defaults to detected channel count)
  - `min_freq: float = 150.0`, `max_freq: float = 2500.0`
  - `playback_speed: float = 10.0` (rows per second)
  - `volume: float = 0.8` (0.0–1.0)
  - `scale: Literal["linear", "log10", "ln"] = "log10"`
  - `freq_mode: Literal["index", "wavelength"] = "index"`
  - `sample_rate: int = 44100`
  - `output: str | None = None` (path for .wav; None = play through speakers)
  - `yes: bool = False` (skip interactive confirmation)
  - `wavelength_path: str | None = None` (path to wavelength reference CSV)
- `validate()` method: checks `max_freq > min_freq`, `playback_speed > 0`, `0.0 <= volume <= 1.0`, `n_bins >= 1`, `sample_rate > 0`, etc. Raises clear `ValueError` messages.

---

### Component 2: Data I/O — `sonify/data_io.py`

#### [NEW] [data_io.py](file:///d:/Freelancing/YSP/sonify/data_io.py)

- `load_csv(path: str) -> pd.DataFrame` — Generic CSV reader via `pandas.read_csv()`. No knowledge of band naming. Just loads and returns.

---

### Component 3: Band Detection — `sonify/band_detect.py`

#### [NEW] [band_detect.py](file:///d:/Freelancing/YSP/sonify/band_detect.py)

- `detect_band_columns(df: pd.DataFrame) -> list[str]`:
  - Regex patterns to match band/channel columns broadly: `Band_\d+_bc`, `Band_\d+`, `Channel_\d+`, `band\d+`, etc. (case-insensitive)
  - **Exclude** columns containing `std`, `sdt`, `max`, `min`, `err` (case-insensitive) — these are housekeeping/noise columns
  - Return sorted list of matched column names (sorted by extracted numeric index)
  - **[FIXED]** Also return the extracted numeric band index per matched column (e.g. as a parallel list or `dict[str, int]`), not just the column names. Component 5's wavelength mode needs the actual band number to look up wavelengths, and Component 4's rebinning needs to keep band order and band identity in sync — both downstream consumers need this, so detection is the right place to produce it once rather than re-parsing column names later.
- `confirm_with_user(detected_columns: list[str], skip_confirm: bool = False) -> bool`:
  - Print detected count and column names
  - If `skip_confirm`, return `True`
  - Otherwise prompt `Proceed with these N channels? [y/n]`

---

### Component 4: Preprocessing — `sonify/preprocess.py`

#### [NEW] [preprocess.py](file:///d:/Freelancing/YSP/sonify/preprocess.py)

- `sort_by_row_order(df: pd.DataFrame) -> pd.DataFrame`:
  - If `row_num` column exists, sort by it ascending
  - Elif `depth` column exists, sort by it descending (instrument descending)
  - Else keep original order
- `clean(df: pd.DataFrame, band_cols: list[str]) -> np.ndarray`:
  - Extract the band columns as a 2D numpy array (rows × channels)
  - Replace NaN with 0
  - Clip negative values to 0 (background subtraction artifacts)
  - **[FIXED]** This is now the **single, canonical place** negative-clipping happens in the whole pipeline. `mapping.scale_values()` (Component 5) no longer re-clips — see that section for why duplicating it was a problem.
- `rebin(matrix: np.ndarray, n_bins: int) -> np.ndarray`:
  - If `n_bins >= matrix.shape[1]`, return as-is
  - Split columns into `n_bins` contiguous groups (as equal-width as possible using `np.array_split`)
  - Average each group per row → result is (rows × n_bins)
- **[FIXED] [NEW]** `rebin_wavelengths(wavelengths: np.ndarray, n_bins: int) -> np.ndarray`:
  - Same grouping logic as `rebin()` (`np.array_split` into `n_bins` contiguous groups), but applied to the 1D array of per-band wavelength centers, averaging each group.
  - **Must use the identical grouping boundaries as `rebin()`** — extract the grouping into a small shared helper (e.g. `_contiguous_groups(n_items, n_bins)`) that both `rebin()` and `rebin_wavelengths()` call, so the data matrix and the wavelength array are always collapsed in lockstep. Without this, rebinned channel 3's frequency could be computed from the wrong slice of the original spectrum.

---

### Component 5: Mapping — `sonify/mapping.py`

#### [NEW] [mapping.py](file:///d:/Freelancing/YSP/sonify/mapping.py)

- `scale_values(matrix: np.ndarray, mode: str) -> np.ndarray`:
  - **[FIXED]** Assumes input is already non-negative (guaranteed by `preprocess.clean()` upstream — no re-clipping here). Docstring should state this precondition explicitly so nobody re-adds the clip later out of caution and reintroduces the duplication.
  - `"linear"`: return as-is
  - `"log10"`: `np.log10(matrix + epsilon)` where epsilon = 1e-10
  - `"ln"`: `np.log(matrix + epsilon)`
- **[FIXED]** `normalize_per_channel(matrix: np.ndarray) -> np.ndarray` (renamed from `normalize_global`):
  - **Per-channel** min/max normalization: for each channel (column), compute that channel's own min and max across all rows in the loaded slice, then scale that column to `[0, 1]` independently of every other channel.
  - Guard divide-by-zero: if a channel is constant (max == min), set that channel's output to 0 everywhere.
  - **Resolved design decision** (previously inconsistent across two sections of this plan — see rationale below): this is per-channel, not a single global min/max shared across all 32 channels.
- `assign_frequencies(n_channels: int, min_freq: float, max_freq: float, mode: str = "index", wavelengths: np.ndarray | None = None) -> np.ndarray`:
  - **index mode**: `freqs[i] = min_freq * (max_freq / min_freq) ** (i / (n - 1))` for i in 0..n-1
  - **wavelength mode**: use the wavelength values (already rebinned to length `n_channels` via `preprocess.rebin_wavelengths()` if rebinning was applied) to interpolate into the log-frequency window. Map wavelength range [λ_min, λ_max] → [min_freq, max_freq] via linear interpolation on wavelength, log-placement in Hz.
  - **[FIXED]** Caller (the CLI runner, Component 9) is responsible for passing a `wavelengths` array whose length already matches `n_channels` post-rebinning — `assign_frequencies` itself does no rebinning and should assert `len(wavelengths) == n_channels` defensively.
  - Returns 1D array of length n_channels
- `load_wavelength_table(path: str) -> dict[int, float]`:
  - Load the watson CSV, return mapping of band number → wavelength center (nm)

> [!IMPORTANT]
> **Normalization decision (resolved):** going with **per-channel** normalization. Fluorescence intensity varies by orders of magnitude between bands as an inherent property of the spectrum — a few bands run bright, others sit near-zero for most of the descent. Global normalization (one shared min/max across all channels and rows) would leave those weak bands almost silent for the entire piece, so in practice you'd only hear two or three loud channels instead of a 32-channel sonification. Per-channel normalization reveals the depth-structure *within* each band, which is the actual point of sonifying a multi-channel dataset. This also matches the spec's intent: "normalize each channel's scaled values... using the min/max across the whole loaded slice" — "each channel's" was the operative phrase, and "whole loaded slice" was contrasting against per-row normalization (§4.3 step 3), not against per-channel. If a true global/relative-loudness mode is wanted later, expose it as an additional `normalize_global()` function and a `--normalize {per-channel,global}` flag rather than overloading this one.

---

### Component 6: Synthesis — `sonify/synth.py`

#### [NEW] [synth.py](file:///d:/Freelancing/YSP/sonify/synth.py)

- `synthesize(amplitude_matrix: np.ndarray, freqs: np.ndarray, seconds_per_row: float, sample_rate: int) -> np.ndarray`:
  - For each row (segment):
    - Compute `segment_samples = round(seconds_per_row * sample_rate)`
    - For each channel: generate `amplitude * sin(2π * freq * t + phase_offset)` where `phase_offset` carries over from previous segment
    - Apply short fade-in/fade-out envelope (raised-cosine) at segment boundaries per channel
    - Sum all channel oscillators → mono segment
  - Concatenate all segments → full waveform
  - Apply master gain: `waveform *= volume`
  - Peak-normalize: `waveform /= max(abs(waveform))` (guard divide-by-zero)
  - Return float64 array in [-1, 1]

Key implementation details:
- `t` array for each segment: `np.arange(segment_samples) / sample_rate`
- Phase update: `phase_offset += 2 * pi * freq * segment_samples / sample_rate`; take modulo 2π
- **[FIXED]** Fade envelope: raised-cosine, but length is now capped relative to the segment itself: `fade_samples = min(round(0.010 * sample_rate), segment_samples // 4)`. At the default 10 rows/sec this is still the full 10ms. But at `--playback-speed 40` (a value the spec's own manual-tuning step asks you to test), each segment is only ~25ms long; an uncapped 10ms-in + 10ms-out fade would leave just 5ms of full-amplitude tone per row, so most of what you'd hear is ramp rather than signal. The cap guarantees at least half of every segment stays at full amplitude regardless of playback speed.

---

### Component 7: Playback — `sonify/playback.py`

#### [NEW] [playback.py](file:///d:/Freelancing/YSP/sonify/playback.py)

- `play(waveform: np.ndarray, sample_rate: int)`:
  - Try `import sounddevice as sd`, then `sd.play(waveform, sample_rate); sd.wait()`
  - **[FIXED]** If `sounddevice` import or playback fails, fall back to `import simpleaudio as sa`: convert waveform to int16 (same conversion as `export.py`), `sa.play_buffer(...)`, then `.wait_done()`.
  - Only if **both** fail, print an error message with install instructions for either library (`pip install sounddevice` or `pip install simpleaudio`) rather than silently doing nothing — this matters because some sandboxed/Antigravity environments may not have audio drivers available for `sounddevice` specifically.

---

### Component 8: WAV Export — `sonify/export.py`

#### [NEW] [export.py](file:///d:/Freelancing/YSP/sonify/export.py)

- `export_wav(waveform: np.ndarray, sample_rate: int, path: str)`:
  - Convert float [-1, 1] → int16: `(waveform * 32767).clip(-32768, 32767).astype(np.int16)`
  - Write via `scipy.io.wavfile.write(path, sample_rate, data)`
  - Print confirmation message with file path

---

### Component 9: CLI Runner — `scripts/run_sonify.py`

#### [NEW] [run_sonify.py](file:///d:/Freelancing/YSP/scripts/run_sonify.py)

- `argparse` setup wiring all CLI parameters from §5
- Pipeline execution order:
  1. Parse args → build `SonificationConfig` → validate
  2. `load_csv(input_path)`
  3. `detect_band_columns(df)` → `confirm_with_user(cols, skip_confirm=config.yes)`
  4. `sort_by_row_order(df)`
  5. Slice rows: `df.iloc[row_start:row_end]`
  6. `clean(df, band_cols)` → numpy matrix (negative-clipping happens here, once)
  7. `rebin(matrix, n_bins)` if n_bins specified
  8. **[FIXED]** If `freq_mode == "wavelength"`: load wavelength table, build the per-detected-band wavelength array (using the band indices from Component 3), then if rebinning was applied in step 7, also call `rebin_wavelengths(wavelength_array, n_bins)` so the wavelength array shrinks to the same length and grouping as the rebinned data matrix
  9. `scale_values(matrix, config.scale)`
  10. `normalize_per_channel(matrix)`
  11. `assign_frequencies(n_channels, min_freq, max_freq, freq_mode, wavelengths)` — `wavelengths` here is the (possibly rebinned) array from step 8, or `None` in index mode
  12. `synthesize(amplitude_matrix, freqs, 1.0/playback_speed, sample_rate)`
  13. If `output` path specified: `export_wav(waveform, sample_rate, output)`
  14. Else: `play(waveform, sample_rate)`
- Dataset-specific defaults (borehole file path, wavelength table path) live here, not in the engine

---

### Component 10: Unit Tests

#### [NEW] [test_band_detect.py](file:///d:/Freelancing/YSP/tests/test_band_detect.py)
- Test with synthetic CSV: columns like `Band_1_bc`, `Band_2_bc`, `Band_1_std1_MAX_SDT`, `row_num`, `depth`
- Verify correct columns detected, `_std1_MAX_SDT` excluded
- **[FIXED]** Verify the returned band-index mapping is correct (e.g. `Band_3_bc` → index `3`), since Components 4/5/9 now depend on it

#### [NEW] [test_preprocess.py](file:///d:/Freelancing/YSP/tests/test_preprocess.py)
- Test `clean()`: NaN → 0, negative → 0
- Test `rebin()`: 8 channels → 4 bins produces correct averaged values
- **[FIXED]** Test `rebin_wavelengths()` on the same 8-channel synthetic case, with known wavelength values, and assert its grouping boundaries exactly match `rebin()`'s for the same `n_bins` (e.g. by checking both used the same `_contiguous_groups` output)

#### [NEW] [test_mapping.py](file:///d:/Freelancing/YSP/tests/test_mapping.py)
- Test all 3 scale modes on data with zeros (no longer test negatives here, since `scale_values` now assumes pre-clipped input — test negative-clipping in `test_preprocess.py` instead)
- Test `assign_frequencies`: correct count, within range, strictly increasing, log-spaced (constant ratio) in index mode
- **[FIXED]** Test `assign_frequencies` in wavelength mode with a rebinned wavelength array, confirming it raises/asserts clearly if array length doesn't match `n_channels`
- Test `normalize_per_channel`: output in [0, 1] per column independently; confirm a low-intensity channel still reaches 1.0 at its own peak (this is the behavior that per-channel normalization is for for — assert it explicitly so a future refactor can't silently flip back to global without a test failing)

#### [NEW] [test_synth.py](file:///d:/Freelancing/YSP/tests/test_synth.py)
- Test waveform length matches `n_rows * seconds_per_row * sample_rate` (±1 sample for rounding)
- Test no sample exceeds [-1, 1] after normalization
- **[FIXED]** Test fade-length capping: with a very high playback speed (short `seconds_per_row`) producing `segment_samples` smaller than `2 * round(0.010 * sample_rate)`, assert the actual fade applied is `segment_samples // 4` on each side, not the uncapped 10ms value
- Test .wav export round-trip: write → read back, check sample rate matches

---

### Component 11: Package Init & README

#### [MODIFY] [__init__.py](file:///d:/Freelancing/YSP/sonify/__init__.py)
- Add version string and public API imports (use `normalize_per_channel`, not the old `normalize_global` name, in any exported symbol list)

#### [MODIFY] [README.md](file:///d:/Freelancing/YSP/README.md)
- Usage instructions, CLI examples, manual tuning notes
- **[FIXED]** Document the per-channel normalization decision and rationale (one sentence) so future contributors don't have to dig through this plan to know why weak channels are still audible

---

## Verification Plan

### Automated Tests
```bash
python -m pytest tests/ -v
```

### Manual Verification
1. Run on full borehole dataset at playback speeds 5, 10, 20, 40 rows/sec
2. Listen for clicks/pops at segment boundaries
3. **[FIXED]** At 40 rows/sec specifically, confirm tones are still audibly present (not just fade ramps) — this is the scenario the fade-cap fix targets
4. Compare `--scale linear` vs `log10` vs `ln`
5. Compare `--freq-mode index` vs `wavelength`, including with `--n-bins` set lower than the detected channel count, to confirm wavelength mode still sounds coherent after rebinning (this is the scenario the wavelength-rebinning fix targets)
6. Confirm a known weak channel (one that's near-zero most of the descent in the raw data) is still audibly present at its peak, not drowned out — this is the scenario the per-channel normalization decision targets
7. Verify `.wav` export plays correctly in an external player
8. If testing in an environment without working audio output, confirm the `sounddevice` → `simpleaudio` fallback in `playback.py` is reachable (e.g. by temporarily forcing the primary import to fail) rather than only ever exercising the `--output` / `.wav` export path