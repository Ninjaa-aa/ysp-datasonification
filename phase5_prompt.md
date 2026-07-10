# Phase 5 — Display Improvements, Auto-Gain, Sustain, and Sound Enhancement

## Step 0: Read the codebase before writing anything

Read every file before proposing or implementing anything:

```
sonify/config.py
sonify/data_io.py
sonify/band_detect.py
sonify/preprocess.py
sonify/mapping.py
sonify/synth.py
sonify/playback.py
sonify/export.py
sonify/visualize.py
sonify/video_export.py
scripts/run_sonify.py
tests/test_band_detect.py
tests/test_preprocess.py
tests/test_mapping.py
tests/test_synth.py
tests/test_visualize.py
```

After reading, write a brief summary of the current pipeline and visual layer before
touching any code. Confirm you understand the trail display, per-channel normalization,
and the existing scale modes (linear/log10/ln) before proceeding.

---

## Project background

BMSIS YSP "Sounds of Deep Ice Fluorescence" — open-source generic Python toolkit for
sonifying any multi-channel tabular dataset. Test case: 32-channel UV fluorescence scan
down a Greenland ice borehole. The pipeline is across Phases 1-4: generic CSV load,
auto band detection, preprocessing, per-channel amplitude mapping, additive sine
synthesis with phase-continuity, synchronized visual display with trail and minimap,
video export.

These changes come directly from feedback by Dr. Michael Malaska (project supervisor)
after reviewing the Phase 4 video outputs. Implement every item below exactly as
specified — do not skip or defer any of them.

**Critical constraint (unchanged):** `sonify/` engine stays generic and dataset-agnostic.
No hardcoded channel counts or column names anywhere in the engine.

---

## This prompt covers four areas, implement in this order

1. **Display fixes** — marker size, square markers, intensity scale/colorbar
2. **Auto-gain** — global gain control with multiple reference modes
3. **Interactive playback speed** — prompt at runtime if not set via CLI
4. **Sound enhancements** — sustain component, harmonic overtones for bell/chime quality

Do each area completely, run all tests, confirm passing, then move to the next.

---

## Part 1: Display Fixes

Dr. Malaska's exact requests: "Possible to make pixels larger? Possible to make pixels
square? Possible to add scale somewhere?"

### 1a. Larger, configurable marker size

Add `--marker-size INT` CLI argument (default: `120`; the current default is around 36,
which was too small). This maps directly to matplotlib's `s=` scatter parameter.

In `sonify/visualize.py`:
- Add `marker_size: int = 120` parameter to `render_frame()` and `render_all_frames()`
- In circles mode, `marker_size` controls the BASE (minimum visible) size, and the
  maximum radius still scales with intensity — so `size = marker_size + (marker_size * 3 * amplitude)`
  giving a 1x-4x size range. This makes the size variation actually visible at the larger
  base size.
- In dots mode, `marker_size` is a fixed scatter `s=` value for all dots.

Add `marker_size: int = 120` to `SonificationConfig`. Validate: `marker_size >= 10`.

### 1b. Square markers

Add `--marker-shape {circle,square}` CLI argument (default: `square` — Dr. Malaska
specifically asked for square, so make it the new default).

In `sonify/visualize.py`:
- `circle` → matplotlib scatter marker `'o'` (current behavior)
- `square` → matplotlib scatter marker `'s'`
- This applies to both dots mode and circles mode. In circles mode with square markers,
  the size still scales with intensity (squares of varying size), the shape just changes
  from circle to square.

Add `marker_shape: Literal["circle", "square"] = "square"` to `SonificationConfig`.

### 1c. Intensity scale / colorbar

Add a colorbar to the right side of the MAIN panel (not the minimap panel) showing what
the colors mean. Label it "Fluorescence Intensity (normalized)".

In `sonify/visualize.py`, `render_frame()`:
- Add a `ScalarMappable` with the same colormap and `vmin=0, vmax=1` used for the dots
- Call `plt.colorbar(mappable, ax=ax, fraction=0.02, pad=0.01)` to add a thin colorbar
  on the right edge of the main panel
- Label the colorbar: `cbar.set_label("Intensity", color="white", fontsize=8)`
- Style the colorbar tick labels white to match the dark background
- The colorbar must not overlap with the minimap panel — position it inside the main
  panel's axes only

Add `show_colorbar: bool = True` to `SonificationConfig` and `--show-colorbar` /
`--no-colorbar` flags (default on). Some users may want a cleaner frame without it.

### 1d. Tests for display changes

Add to `tests/test_visualize.py`:
- `test_square_marker_differs_from_circle`: frames rendered with `marker_shape="square"`
  and `marker_shape="circle"` produce different pixel arrays
- `test_marker_size_affects_frame`: frames rendered with `marker_size=40` and
  `marker_size=200` produce different pixel arrays
- `test_colorbar_changes_frame`: frame with `show_colorbar=True` differs from
  `show_colorbar=False`

---

## Part 2: Auto-Gain

Dr. Malaska's exact request: "We may be amplifying the noise just before the first signal.
Add an auto-gain function that sets the volume intensity. The code would scan the entire
band dataset and identify the global maximum value. User options to set scales to various
intensities relative to the maximum signal."

### The problem with the current approach

Phase 1 uses `normalize_per_channel()` — each channel is independently normalized to
[0, 1] using its own min/max. This means a channel that is pure noise (all values near
zero) still gets amplified to full range, because its own tiny max becomes 1.0. This is
exactly the noise amplification Dr. Malaska identified.

### New approach: global gain normalization

Replace the per-channel normalization step in the pipeline with a configurable global
gain function. The reference point is computed from the **entire band matrix** (all
channels, all rows), not per-channel. The user chooses what statistical measure becomes
the reference.

**Keep `normalize_per_channel()` in `mapping.py` for backward compatibility** (tests
depend on it). Add new functions alongside it.

### New `gain_mode` parameter

Add to `SonificationConfig`:

```python
gain_mode: Literal[
    "max_linear",       # global max = amplitude 1.0 (LINEAR response) [DEFAULT]
    "max_log",          # global max = amplitude 1.0 (LOG response)
    "pct90_linear",     # 90th percentile = amplitude 1.0, clip above (linear)
    "pct90_log",        # 90th percentile = amplitude 1.0, clip above (log)
    "median_linear",    # median value = amplitude 0.5 (linear)
    "median_log",       # median value = amplitude 0.5 (log)
    "mean_linear",      # mean value = amplitude 0.5 (linear)
    "mean_log",         # mean value = amplitude 0.5 (log)
] = "max_linear"
```

CLI: `--gain-mode {max_linear,max_log,pct90_linear,pct90_log,median_linear,median_log,mean_linear,mean_log}`

### Implementation in `sonify/mapping.py`

Add these functions:

```python
def compute_gain_reference(matrix: np.ndarray, gain_mode: str) -> tuple[float, float]:
    """Scan the entire band matrix and compute the gain reference point.
    
    Returns (reference_value, scale_factor) where:
    - reference_value: the statistical anchor (max, 90th pct, median, mean)
    - scale_factor: what amplitude that reference maps to (1.0 for max modes, 0.5 for midpoint modes)
    
    Prints a summary line so the user can see what the auto-gain detected:
      [GAIN] Global max: 326.4  Median: 42.1  Mean: 58.3  Reference: 326.4 (max_linear)
    """

def apply_global_gain(
    matrix: np.ndarray,
    gain_mode: str,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Apply global gain normalization to the full matrix.
    
    For LINEAR modes:
        - Compute reference value from gain_mode
        - If midpoint mode (median/mean): amplitude = (value / reference) * 0.5
          Values above reference get amplitudes above 0.5, clipped to 1.0
        - If max mode: amplitude = value / reference, clipped to 1.0
    
    For LOG modes:
        - Apply log10 transform first (value + epsilon)
        - Then apply the same reference logic on the log-transformed values
        - The log transform compresses dynamic range so quiet structure is audible
          but the GAIN REFERENCE still anchors the loudest point
    
    Returns np.ndarray in [0, 1], same shape as input.
    Always clips output to [0, 1] — no value can exceed 1.0.
    """
```

### Pipeline change in `run_sonify.py`

Replace the current `normalize_per_channel(matrix)` call with `apply_global_gain(matrix, config.gain_mode)`.

The `--scale` parameter (linear/log10/ln) that currently controls pre-processing still
runs BEFORE `apply_global_gain()`, unchanged. Think of it as two independent steps:
- `scale_values()` — transforms the raw values (log compression of the raw numbers)
- `apply_global_gain()` — normalizes to [0,1] using a global reference point

Both can be set independently. The default combination is `--scale log10 --gain-mode max_linear`.

### Print the gain summary to the user

When the pipeline runs, after computing the gain reference, print:

```
[GAIN]  Scanning 4000 rows x 32 channels...
[GAIN]  Global max: 326.4   90th pct: 184.2   Median: 42.1   Mean: 58.3
[GAIN]  Mode: max_linear  ->  reference = 326.4 (maps to amplitude 1.0)
```

This directly addresses Dr. Malaska's concern — the user can see what the gain function
found and judge whether the reference makes sense for their dataset.

### Tests for auto-gain

Add to `tests/test_mapping.py`:
- `test_gain_max_linear_clips_to_one`: output max is 1.0, all values in [0,1]
- `test_gain_median_linear_midpoint`: median of output is approximately 0.5
- `test_gain_mean_linear_midpoint`: mean of output is approximately 0.5
- `test_gain_pct90_clips_outliers`: values above 90th percentile are clipped to 1.0
- `test_gain_log_modes_compress_range`: log mode output has smaller std than linear mode
  on the same data (confirms dynamic range compression)
- `test_gain_noisy_channel_stays_quiet`: a channel whose max is 5% of the global max
  stays near-silent in max_linear mode (this is the specific noise problem Dr. Malaska
  identified — verify the fix works)

---

## Part 3: Interactive Playback Speed

Dr. Malaska's exact request: "User input to set playback speed."

If `--playback-speed` is not provided on the CLI, instead of using the default silently,
prompt the user interactively:

```
Playback speed not set. How many rows per second?
  Suggested values:
    [1]   1  rows/sec  ->  400.0s total  (very slow, meditative)
    [2]   5  rows/sec  ->   80.0s total  (slow)
    [3]  10  rows/sec  ->   40.0s total  (default, recommended)
    [4]  20  rows/sec  ->   20.0s total  (fast)
    [5]  40  rows/sec  ->   10.0s total  (very fast)
    [c]  Custom value

Enter choice [1-5/c]: 
```

If the user picks `c`, prompt: `Enter rows per second (e.g. 15): `

Calculate the "total" duration based on the number of rows being rendered so the user
can make an informed choice.

If `--yes` flag is set (non-interactive mode), use the default of 10 rows/sec without
prompting.

Implement this as a `prompt_playback_speed(n_rows: int, skip: bool) -> float` function
at the top of `run_sonify.py` (not in the engine — CLI layer only).

No new tests needed for this (it's an interactive prompt — test interactivity manually).

---

## Part 4: Sound Enhancements

### 4a. Sustain component

Dr. Malaska's exact request: "Add a sustain component."

Currently, amplitude transitions between rows are abrupt (even with the fade envelope,
the amplitude TARGET changes instantaneously from one row to the next). Sustain means
the sound from the previous row lingers into the next row with a gradual decay, so
consecutive rows overlap and blend rather than cutting cleanly.

**Implementation in `sonify/synth.py`:**

Add `sustain: float = 0.3` parameter to `synthesize()` (range 0.0 to 1.0, where 0.0
is the current behavior and 1.0 is maximum sustain).

When `sustain > 0`, for each channel in each row:
- The target amplitude for the new row is `new_amp`
- The amplitude the oscillator actually starts at is a blend:
  `effective_start_amp = (1 - sustain) * new_amp + sustain * prev_amp`
- Then smoothly interpolate from `effective_start_amp` to `new_amp` across the segment
  using a linear ramp (not a step): `np.linspace(effective_start_amp, new_amp, segment_samples)`
- This means the channel doesn't immediately jump to its new amplitude; it slides there
  across the duration of the row

The effect: loud channels fade into the next row instead of cutting off, creating a
smoother, more connected sound. The phase-continuity from Phase 1 is unchanged — the
frequency remains continuous; only the amplitude envelope changes.

Add `sustain: float = 0.3` to `SonificationConfig`. Validate: `0.0 <= sustain <= 1.0`.
CLI: `--sustain FLOAT` (default 0.3). Document that 0.0 = no sustain (Phase 1 behavior),
0.3 = light sustain (new default), 1.0 = maximum overlap.

**Tests in `tests/test_synth.py`:**
- `test_sustain_zero_matches_no_sustain`: with `sustain=0.0`, output is identical to
  current behavior
- `test_sustain_smooths_amplitude_transitions`: given two consecutive rows with very
  different amplitudes, the output waveform with `sustain=0.5` has a smaller max
  amplitude jump at the boundary than with `sustain=0.0`

### 4b. Harmonic overtones for bell/chime quality

Dr. Malaska's exact request: "Other adjustments to sound to make more pleasant: tinkling
wind chimes or something."

The current synthesis uses pure sine waves (single frequency per channel per row). Pure
sines sound electronic and flat. Bells and wind chimes have a characteristic timbre from
harmonic overtones — the fundamental frequency plus additional partials at integer
multiples (2x, 3x, 4x) with decreasing amplitude.

**Implementation in `sonify/synth.py`:**

Add `timbre: Literal["sine", "bell", "chime"] = "sine"` parameter to `synthesize()`.

The three modes:

- **`sine` (default, Phase 1 behavior):** pure sine at fundamental frequency only.
  No change to existing code path.

- **`bell`:** fundamental + 3 harmonic partials. For each channel oscillator, instead
  of one sine, generate 4 and sum them:
  ```
  partial 1 (fundamental):  amplitude * 1.00 * sin(2π * freq * t + φ)
  partial 2 (octave):       amplitude * 0.50 * sin(2π * freq * 2 * t + φ2)
  partial 3:                amplitude * 0.25 * sin(2π * freq * 3 * t + φ3)
  partial 4:                amplitude * 0.12 * sin(2π * freq * 4 * t + φ4)
  ```
  Each partial has its own independent phase tracker (so 4 phase values per channel,
  not 1). Amplitudes sum to ~1.87 of the base amplitude, so divide the total by 1.87
  to keep the output in the same amplitude range as sine mode.

- **`chime`:** same as bell but with slightly inharmonic partials (not exact integer
  multiples), giving a metallic, shimmery quality:
  ```
  partial 1 (fundamental):  amplitude * 1.00 * sin(2π * freq * 1.000 * t + φ)
  partial 2:                amplitude * 0.50 * sin(2π * freq * 2.756 * t + φ2)
  partial 3:                amplitude * 0.25 * sin(2π * freq * 5.404 * t + φ3)
  partial 4:                amplitude * 0.12 * sin(2π * freq * 8.933 * t + φ4)
  ```
  The inharmonic ratios (2.756, 5.404, 8.933) are derived from the known overtone
  spectrum of a tubular bell. They produce the characteristic "shimmering" quality
  that makes real bells and chimes sound like they sustain and sparkle rather than
  sound electronic. Same phase-tracking and normalization as bell mode.

Add `timbre: Literal["sine", "bell", "chime"] = "chime"` to `SonificationConfig`.
Note: default is `chime` (not `sine`) because this is what Dr. Malaska asked for as the
target sound, and users who want the original pure-sine behavior can pass `--timbre sine`.
CLI: `--timbre {sine,bell,chime}`.

**Performance note:** Bell and chime modes do 4x the oscillator work per channel. For 32
channels at 4000 rows, this is still fast (numpy vectorized sine is cheap). At 2048
channels it becomes relevant — add a comment noting this. No optimization needed for
Phase 5, but document the tradeoff.

**Tests in `tests/test_synth.py`:**
- `test_timbre_sine_unchanged`: `timbre="sine"` output matches existing synthesis
  behavior exactly
- `test_timbre_bell_differs_from_sine`: `timbre="bell"` and `timbre="sine"` produce
  different waveforms on the same input
- `test_timbre_chime_differs_from_bell`: `timbre="chime"` and `timbre="bell"` produce
  different waveforms (confirms inharmonic ratios are actually different from integer
  multiples)
- `test_timbre_amplitude_range`: both bell and chime outputs remain in [-1, 1] after
  normalization

---

## Test summary

After all four parts are done:

```bash
python -m pytest tests/ -v
```

Expected counts:
- 39 Phase 1 tests (unchanged)
- 6 Phase 2 tests (unchanged)
- 3 trail tests from Phase 3/4 (unchanged)
- 2 parameter mapping tests from Phase 3/4 (unchanged)
- 3 new display tests (Part 1)
- 6 new auto-gain tests (Part 2)
- 2 new sustain tests (Part 4a)
- 4 new timbre tests (Part 4b)

**Total: 65 tests minimum, all passing.**

If any Phase 1-4 tests break due to the auto-gain change (the normalization step changed),
fix the tests to reflect the new behavior and document what changed and why. Do not
silently delete failing tests.

---

## Verification commands

Run all of these and report actual observed results in the walkthrough:

```bash
# Confirm square markers, larger size, colorbar visible
py scripts/run_sonify.py --yes --row-end 50 \
    --marker-shape square --marker-size 150 --show-colorbar \
    --output-name display_test \
    --video-output outputs/display_test.mp4 \
    --frame-width 640 --frame-height 360

# Auto-gain: compare all modes on the same 200-row slice
# Run each and report what [GAIN] printed for global max, median, mean
py scripts/run_sonify.py --yes --row-end 200 --gain-mode max_linear --output-name gain_max_lin --video-output outputs/gain_max_lin.mp4
py scripts/run_sonify.py --yes --row-end 200 --gain-mode max_log --output-name gain_max_log --video-output outputs/gain_max_log.mp4
py scripts/run_sonify.py --yes --row-end 200 --gain-mode pct90_linear --output-name gain_pct90 --video-output outputs/gain_pct90.mp4
py scripts/run_sonify.py --yes --row-end 200 --gain-mode median_linear --output-name gain_median --video-output outputs/gain_median.mp4
py scripts/run_sonify.py --yes --row-end 200 --gain-mode mean_linear --output-name gain_mean --video-output outputs/gain_mean.mp4

# Timbre comparison — same data, three timbres
py scripts/run_sonify.py --yes --row-end 100 --timbre sine --output outputs/timbre_sine.wav
py scripts/run_sonify.py --yes --row-end 100 --timbre bell --output outputs/timbre_bell.wav
py scripts/run_sonify.py --yes --row-end 100 --timbre chime --output outputs/timbre_chime.wav

# Sustain comparison
py scripts/run_sonify.py --yes --row-end 100 --sustain 0.0 --output outputs/sustain_off.wav
py scripts/run_sonify.py --yes --row-end 100 --sustain 0.3 --output outputs/sustain_03.wav
py scripts/run_sonify.py --yes --row-end 100 --sustain 0.7 --output outputs/sustain_07.wav

# Full V2 run — all new features combined
py scripts/run_sonify.py --yes --row-end 200 \
    --marker-shape square --marker-size 150 --show-colorbar \
    --gain-mode max_log \
    --timbre chime --sustain 0.3 \
    --trail-rows 5 --show-minimap \
    --output-name v2_full \
    --video-output outputs/v2_full.mp4

# Interactive playback speed prompt (no --yes, no --playback-speed)
py scripts/run_sonify.py --row-end 50 --output outputs/interactive_test.wav
# Manually select option [3] (10 rows/sec) and confirm it works
```

---

## Walkthrough format when done

- What was built / modified (table, same format as prior phases)
- Key design decisions: especially the auto-gain mode design and how the timbre
  overtone ratios were chosen
- The `[GAIN]` printout values for the borehole dataset (global max, 90th pct, median,
  mean) — this directly addresses Dr. Malaska's noise concern and is important context
- Test counts: X prior + Y new = Z total, all passing
- Actual observed results for every verification command:
  - Does the display look visually better (larger, square, colorbar present)?
  - Does `max_log` mode reveal more quiet structure than `max_linear`?
  - Is the chime timbre audibly more pleasant than sine?
  - Does sustain 0.7 sound noticeably smoother than sustain 0.0?
- Render timing for `v2_full.mp4` at 640×360 and at default 1280×720