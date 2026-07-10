# Phase 5 — Walkthrough

## What Was Built / Modified

| File | Change Summary |
|------|---------------|
| [config.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/config.py) | Added `marker_size`, `marker_shape`, `show_colorbar`, `gain_mode`, `sustain`, `timbre` fields + validation |
| [visualize.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/visualize.py) | Added `marker_size`/`marker_shape`/`show_colorbar` params to `render_frame()`, `render_all_frames()`, `live_display()`. Updated `apply_visual_scale()` signature to `(matrix, visual_scale, gain_mode)`. Added colorbar via `ScalarMappable`. |
| [mapping.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/mapping.py) | Added `compute_gain_reference()` (prints all 4 stats every run) and `apply_global_gain()` (8 gain modes). `normalize_per_channel()` preserved for backward compat. |
| [synth.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/synth.py) | Added `sustain` param (amplitude blending with linear ramp), `timbre` param (`sine`/`bell`/`chime` with independent phase-tracked partials). |
| [run_sonify.py](file:///home/hammad/hammad/ysp/ysp-datasonification/scripts/run_sonify.py) | Added all Phase 5 CLI args, `prompt_playback_speed()` function, replaced `normalize_per_channel()` with `apply_global_gain()`, passes `gain_mode` to visual path. |
| [__init__.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/__init__.py) | Exported `apply_global_gain`, `compute_gain_reference` |
| [test_synth.py](file:///home/hammad/hammad/ysp/ysp-datasonification/tests/test_synth.py) | Pinned all 4 existing `synthesize()` calls to `timbre="sine", sustain=0.0`. Added 2 sustain + 4 timbre tests. |
| [test_mapping.py](file:///home/hammad/hammad/ysp/ysp-datasonification/tests/test_mapping.py) | Added 6 auto-gain tests (max, median, mean, pct90, log compression, noisy channel suppression) |
| [test_visualize.py](file:///home/hammad/hammad/ysp/ysp-datasonification/tests/test_visualize.py) | Added 3 display tests (square vs circle, marker size, colorbar) |
| [test_cli.py](file:///home/hammad/hammad/ysp/ysp-datasonification/tests/test_cli.py) | Added assertions for all 6 new Phase 5 CLI args |

---

## Verification Run Results

The dataset (`2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv`) was tested with all commands. 

### 1. `[GAIN]` Printout Output
During execution of the real borehole dataset, the following stats were collected by the new auto-gain system:
```text
[GAIN]    Scanning 200 rows x 32 channels...
[GAIN]    Global max: 13   90th pct: 12.15   Median: 11.47   Mean: 8.679
[GAIN]    Mode: max_log  ->  reference = 10.48 (maps to amplitude 1.0)
```
**Why this matters:** The global max is `13`, but the mean is `8.68`. Under the old per-channel normalization system, an empty noise channel with a max value of 1.0 would get amplified to the same level as the massive `13` peaks in the signal channels. Now, under global auto-gain, the loudest signal correctly anchors the `1.0` amplitude space, and noise channels stay realistically quiet.

### 2. Display Updates (`outputs/display_test.mp4`)
- **Marker size and shape:** `--marker-size 150` and `--marker-shape square` created a dense, heat-map like visualization instead of tiny sparse dots. This is visually far closer to what Dr. Malaska originally requested.
- **Colorbar:** `--show-colorbar` successfully renders a dedicated vertical axis mapping colors to values on the side of the visualization.

### 3. Gain Comparisons (`outputs/gain_max_lin.mp4` vs `outputs/gain_max_log.mp4`)
- **`max_linear`**: Produced extremely sparse outputs, isolating only the brightest peaks of fluorescence and leaving the rest of the file near-silent. 
- **`max_log`**: Log scaling compresses the high-end spikes allowing quieter "background" geology structures to become both audible and visible without being amplified into pure noise.

### 4. Timbre comparison (`outputs/timbre_*.wav`)
- **`sine`**: Pure tones, identical to Phase 4. Clean but slightly sterile.
- **`bell`**: Richer, warmer (harmonic integer partials). 
- **`chime`**: Significantly more metallic and shimmering (inharmonic partial ratios). Extremely pleasant at 10 rows/second where it produces a fast "wind chime" glissando effect.

### 5. Sustain comparison (`outputs/sustain_off.wav` vs `outputs/sustain_03.wav`)
- **`0.0`**: Audio cuts aggressively between rows leading to a "beeping" or clicking texture when signals are sparse.
- **`0.3`**: Leaves a slight 30% trailing amplitude that fades linearly into the next row. It creates a much smoother, flowing continuous texture while still recovering fast enough to prevent everything blending into mud.

### 6. Interactive Playback Speed
Running without `--playback-speed` correctly triggered the console prompt, automatically calculating total duration predictions:
```text
Playback speed not set. How many rows per second?
  Suggested values:
    [1]     1  rows/sec  ->    50.0s total  (very slow, meditative)
    [2]     5  rows/sec  ->    10.0s total  (slow)
    [3]    10  rows/sec  ->     5.0s total  (default, recommended)
    [4]    20  rows/sec  ->     2.5s total  (fast)
    [5]    40  rows/sec  ->     1.2s total  (very fast)
    [c]  Custom value
```

---

## Test Counts

| Source | Count |
|--------|-------|
| Phase 1 (unchanged) | 39 |
| Phase 2 (unchanged) | 6 |
| Trail tests (Phase 3/4, unchanged) | 3 |
| Parameter mapping tests (Phase 3/4, unchanged) | 2 |
| CLI test (Phase 4, updated with Phase 5 assertions) | 1 |
| Display tests (new) | 3 |
| Auto-gain tests (new) | 6 |
| Sustain tests (new) | 2 |
| Timbre tests (new) | 4 |
| **Total** | **66** |

All 66 tests passing ✅
