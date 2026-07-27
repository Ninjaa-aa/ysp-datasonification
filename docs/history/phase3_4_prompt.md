# Phase 3 & 4 — Expanded Channels, Parameter Mapping, Trail Display, and V1 Polish

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

After reading, write a two-sentence summary of the current pipeline and visual layer so I
can confirm understanding before any code is written.

---

## Project background

BMSIS YSP "Sounds of Deep Ice Fluorescence" — open-source generic Python toolkit for
sonifying any multi-channel tabular dataset. Test case: 32-channel UV fluorescence scan
down a Greenland ice borehole, but the engine must remain dataset-agnostic.

**What Phases 1 and 2 built:**
- Phase 1: Full generic audio pipeline — CSV load, band auto-detection via regex, sort,
  clean, rebin, scale (linear/log10/ln), per-channel normalize, log-spaced frequency
  assignment (index or wavelength mode), phase-continuous additive sine synthesis with
  capped fades, WAV export or live playback. 39 tests passing.
- Phase 2: Synchronized visual layer — `visualize.py` renders per-channel intensity as
  dots or circles on a dark background using the plasma colormap; `video_export.py` muxes
  frames + WAV into `.mp4` via OpenCV + moviepy. 45 tests total (39 + 6).

**Critical constraint (unchanged):** the `sonify/` engine must remain generic and
dataset-agnostic. No hardcoded channel counts, column names, or dataset assumptions in
the engine. The `scripts/` layer may have dataset-specific defaults.

---

## This prompt covers three things, in this order

1. **Trail fix** (visual improvement to Phase 2 — implement first, it is small)
2. **Phase 3** (expand to 2048 channels, parameter mapping, better channel confirmation UI)
3. **Phase 4** (scrolling minimap, polished UI, output naming, V1 GitHub readiness)

Do them in this order. After each one, confirm it works and all prior tests still pass
before moving to the next.

---

## Part 1: Trail Display Fix

### The problem

Currently each video frame shows only the current row's dots. When the next row plays, the
previous one disappears. The user cannot compare adjacent depth readings or see how the
spectrum is evolving over time within the visible window.

### What to build

Add a **trail** — N rows visible simultaneously in each frame, stacked vertically, with the
current row at the bottom and older rows above it, each fading in opacity and slightly
smaller the older they are.

**Visual layout:**

```
row N-4   ● ● ● ● ● ...   [20% opacity, 40% marker size]   depth: 89.27 m
row N-3   ● ● ● ● ● ...   [40% opacity, 55% marker size]   depth: 89.29 m
row N-2   ● ● ● ● ● ...   [60% opacity, 70% marker size]   depth: 89.31 m
row N-1   ● ● ● ● ● ...   [80% opacity, 85% marker size]   depth: 89.33 m
row N     ● ● ● ● ● ...   [100% opacity, full size]        depth: 89.35 m  ← current
```

- Each row is evenly spaced vertically within the frame
- Depth label for each visible row shown on the left side (if depth column exists)
- Colormap still applies per dot — color encodes intensity; opacity and size encode age
- Number of trail rows is user-configurable: `--trail-rows INT` (default: 5, max: 20)
- When fewer rows have been seen than `trail_rows` (e.g. first 3 frames of a 5-row trail),
  fill only as many rows as are available — do not pad with empty rows

### Changes required

**`sonify/visualize.py`:**
- Modify `render_frame()` to accept `amplitude_history: np.ndarray` shape
  `(n_trail, n_channels)` instead of a single `amplitudes` 1-D array. The last row
  `[-1]` is the current row; earlier rows are older. Depth values similarly become
  `depths: np.ndarray | None` of shape `(n_trail,)`.
- `render_all_frames()` builds the history window per frame using a sliding window over
  the full `amplitude_matrix` — for frame `i`, pass `amplitude_matrix[max(0,i-trail_rows+1):i+1]`
- `live_display()` updates the same way using a deque of recent rows

**`sonify/config.py`:**
- Add `trail_rows: int = 5`
- Validate: `1 <= trail_rows <= 20`

**`scripts/run_sonify.py`:**
- Add `--trail-rows INT` argparse argument
- Pass `trail_rows` through to `render_all_frames()` and `live_display()`

**`tests/test_visualize.py`:**
- Add `test_trail_single_row`: trail of 1 works and is identical to old single-row behavior
- Add `test_trail_opacity_decreases`: older rows render with lower opacity than newer ones
- Add `test_trail_partial_fill`: when history has fewer rows than trail_rows, no crash

**Do not break any of the 45 existing tests.**

---

## Part 2: Phase 3 — Expanded Channels and Parameter Mapping

### 2a. Support up to 2048 channels

The current engine is already generic (no hardcoded 32), but two things need explicit
attention at high channel counts:

**Performance in `render_frame()`:** At 2048 channels per row with a 5-row trail, each
frame renders 10,240 scatter markers. Matplotlib scatter with that many points per frame
will be very slow. Switch the visual rendering strategy above approximately 256 channels:

- **≤ 256 channels:** current scatter-based dots/circles (readable as individual dots)
- **> 256 channels:** render as a **heatmap strip** instead — a 2D image where the
  x-axis is channels and the y-axis is trail rows, colored by intensity. This is far
  faster to render and actually more readable at high channel counts. Use
  `ax.imshow(history_matrix, aspect='auto', cmap=colormap, vmin=0, vmax=1)`.

The switch is automatic based on `n_channels` — no new user flag needed. Document the
threshold in a module-level constant `SCATTER_CHANNEL_LIMIT = 256` in `visualize.py`
so it can be adjusted easily.

**Memory in `render_all_frames()`:** At 2048 channels × 4000 rows × 5 trail rows, the
amplitude history windows are large but the real memory cost is the frames list — 4000
frames × 1280 × 720 × 3 bytes ≈ 11 GB. Add a `--max-frames INT` config parameter that
caps how many rows are rendered into video (default: 500 for safety, user raises it
explicitly). Emit a clear warning if the row slice would produce more frames than
`max_frames`.

### 2b. Better channel confirmation UI

The current `confirm_with_user()` in `band_detect.py` just prints a flat list of column
names. At 32 channels this is fine; at 2048 it is unreadable.

Replace it with a formatted table presentation:

```
Detected 32 spectral channels:

  Index  Column Name          Band No.   Wavelength (nm)
  -----  -------------------  --------   ---------------
      1  Band_1_bc                   1           275.0
      2  Band_2_bc                   2           280.5
    ...
     32  Band_32_bc                 32           446.0

  Excluded (noise/housekeeping): Band_1_std1_MAX_SDT ... (32 columns)
  Housekeeping: entry, Pointcloud_Map_ID, row_num, MOD, depth, rot ... (11 columns)

Proceed with these 32 channels? [y/n/edit]:
```

Add an `edit` option: if the user types `e` or `edit`, enter an interactive sub-menu:
```
  [a] Add a column by name
  [r] Remove a column by index
  [d] Done
```
This lets a user manually include or exclude specific columns without re-running the whole
pipeline. After editing, re-display the updated table and ask for final confirmation.

If `--yes` flag is set, skip all of this entirely (non-interactive mode, for scripts and
tests).

Wavelength column is shown only if the wavelength table was loaded; otherwise that column
is omitted from the table.

### 2c. Parameter mapping — tone and intensity reassignment

Right now the audio mapping is fixed:
- **Frequency (tone):** assigned from band index or wavelength
- **Amplitude (volume):** driven by the band intensity value

Phase 3 wants the user to be able to reassign these mappings to any column in the dataset.
The default behavior is unchanged; this is purely additive.

**New concept: `ParameterMap`**

Add to `sonify/config.py`:

```python
@dataclass
class ParameterMap:
    tone_source: Literal["band_index", "wavelength", "column"] = "band_index"
    tone_column: str | None = None       # used when tone_source == "column"
    intensity_source: Literal["band_value", "column"] = "band_value"
    intensity_column: str | None = None  # used when intensity_source == "column"
```

Add `param_map: ParameterMap = field(default_factory=ParameterMap)` to
`SonificationConfig`.

**What each option means:**

- `tone_source = "band_index"`: current default — frequencies log-spaced by channel index
- `tone_source = "wavelength"`: current wavelength mode — frequencies mapped from nm values
- `tone_source = "column"`: take a single numeric column from the DataFrame (e.g. `depth`
  or `rot`) and map its per-row value to a frequency within `[min_freq, max_freq]`.
  All 32 (or N) channel oscillators in that row share this single frequency, making the
  row's "pitch" driven by that scalar column rather than fixed per-channel.

- `intensity_source = "band_value"`: current default — each channel's amplitude from its
  own normalized band value
- `intensity_source = "column"`: take a single numeric column and use its per-row value
  as a global amplitude multiplier on top of the per-channel values. This modulates the
  overall volume of each row by that column, while channels still differ in relative
  amplitude.

**New CLI args:**

```
--tone-source {band_index,wavelength,column}   default: band_index
--tone-column COLUMN_NAME                       required if --tone-source column
--intensity-source {band_value,column}          default: band_value
--intensity-column COLUMN_NAME                  required if --intensity-source column
```

**Validation in `SonificationConfig.validate()`:**
- If `tone_source == "column"`, `tone_column` must be set and must exist in the loaded
  DataFrame's columns
- If `intensity_source == "column"`, `intensity_column` must be set and must exist

**Implementation in `sonify/mapping.py`:**

Add:
```python
def map_tone_from_column(
    column_values: np.ndarray,   # per-row scalar values
    min_freq: float,
    max_freq: float,
) -> np.ndarray:
    """Map a per-row scalar column to per-row frequencies in [min_freq, max_freq].
    Normalizes the column's range to the frequency range (log-spaced).
    Returns array of shape (n_rows,) — one frequency per row.
    """

def apply_intensity_column(
    amplitude_matrix: np.ndarray,   # (n_rows, n_channels)
    column_values: np.ndarray,       # (n_rows,) scalar multipliers, pre-normalized [0,1]
) -> np.ndarray:
    """Multiply each row's amplitudes by the corresponding column value."""
```

**Changes to `sonify/synth.py`:**

`synthesize()` currently takes a fixed `freqs` array of shape `(n_channels,)`. When
`tone_source == "column"`, frequencies vary per row. Change the signature to accept
`freqs` as either:
- `np.ndarray` shape `(n_channels,)` — current behavior, fixed per-channel frequencies
- `np.ndarray` shape `(n_rows, n_channels)` — per-row frequencies (for column-driven tone)

Detect which case via `freqs.ndim`. In the per-row case, use `freqs[row_idx]` in the
synthesis loop. This is backward-compatible — existing code passing a 1-D array is
unaffected.

**New tests in `tests/test_mapping.py`:**
- `test_map_tone_from_column`: output length matches input, all values within
  `[min_freq, max_freq]`
- `test_apply_intensity_column`: zero column value produces silent row; nonzero scales
  correctly

---

## Part 3: Phase 4 — V1 Polish, Scrolling Minimap, Output Naming

### 4a. Scrolling minimap

Add a two-panel display mode: the main left panel shows the trail (as built in Part 1),
and a new right panel shows an overview minimap of the **entire loaded dataset** as a
heatmap, with a horizontal line marking the current playback position.

**Layout:**

```
┌─────────────────────────────────┬──────────────┐
│                                 │  OVERVIEW    │
│   Trail display (main panel)    │  (minimap)   │
│   rows N-4 through N            │              │
│   colored dots / heatmap        │  full dataset│
│                                 │  as heatmap  │
│                                 │  ──── ←line  │
│                                 │              │
│   Title + depth label           │              │
└─────────────────────────────────┴──────────────┘
```

- Minimap panel: right 20% of frame width
- Main panel: left 80%
- Minimap is a fixed heatmap of `amplitude_matrix` (all rows, all channels) rendered
  once and reused every frame — do not re-render it per frame
- The position line is drawn on top of the cached minimap image at the correct row
  position each frame
- New CLI flag: `--show-minimap` (off by default, on by default when `--video-output` is
  set, since it adds no meaningful overhead for video)

**Implementation note:** Pre-render the minimap as a numpy array once in
`render_all_frames()` before the per-frame loop. Per frame, composite the position line
onto a copy of the cached minimap rather than re-rendering the whole thing. This keeps
per-frame cost low.

### 4b. Polished CLI output

Replace bare `print()` statements throughout `run_sonify.py` with a consistent,
structured output style. Each pipeline stage should print a clear status line:

```
[LOAD]     Loaded 4000 rows x 75 columns from borehole_2019.csv
[DETECT]   Found 32 spectral channels (Band_1_bc ... Band_32_bc)
[SORT]     Sorted by row_num ascending
[CLEAN]    Clipped 0 NaN values, 12 negative values to 0
[REBIN]    32 channels -> 32 bins (no rebinning)
[SCALE]    Applying log10 scale
[SYNTH]    Synthesizing 200 rows at 10.0 rows/sec -> 20.0s audio
[RENDER]   Rendering 200 frames at 1280x720 (trail=5) ...
[RENDER]   Frame 50/200 (25%)
[RENDER]   Frame 100/200 (50%)
[RENDER]   Frame 150/200 (75%)
[RENDER]   Frame 200/200 (100%) -- 22.4s elapsed
[EXPORT]   Writing outputs/final.wav
[VIDEO]    Muxing audio + video -> outputs/final.mp4
[DONE]     outputs/final.mp4 (200 frames, 20.0s, 1280x720)
```

Use a simple `log(stage, message)` helper at the top of `run_sonify.py`. No external
logging libraries — just formatted print statements. The render progress line is printed
every 25% of frames so the user knows it is not hung.

### 4c. Output filename control

Add `--output-name BASE_NAME` CLI argument. When set, it becomes the base for all
output file paths, and `--output` / `--video-output` become optional overrides:

```bash
# These two are equivalent:
py scripts/run_sonify.py --output-name borehole_run1
# produces: outputs/borehole_run1.wav + outputs/borehole_run1.mp4

py scripts/run_sonify.py --output outputs/borehole_run1.wav \
                          --video-output outputs/borehole_run1.mp4
```

If both `--output-name` and `--output` / `--video-output` are set, the explicit paths
take priority. If neither is set, behavior is unchanged (play through speakers / no video).
`--output-name` always writes to the `outputs/` directory.

### 4d. README and GitHub V1 readiness

Update `README.md` to be complete and standalone — someone who has never seen this project
should be able to clone the repo and get sound out of it within five minutes:

- **What it is** (2-3 sentences — the sci-art mission, BMSIS YSP context, Dr. Malaska)
- **Install** (`pip install -r requirements.txt`, note that `opencv-python` and `moviepy`
  are optional for audio-only use)
- **Quick start** — one command that works on the example dataset and produces a WAV file
- **Full command reference** — one table listing every CLI argument with type, default, and
  one-line description. Generate this from `argparse`'s help strings to keep it in sync.
- **Tuning guide** — the results of the Phase 1 listening pass: which playback speed,
  scale mode, and frequency mode sounded best and why
- **Dataset format** — what CSV structure the tool expects, how band auto-detection works,
  how to use it on a non-borehole dataset
- **Credits** — Dr. Michael Malaska, BMSIS YSP, the Astrobiology paper DOI for the WATSON
  instrument table
- **License** — MIT

---

## Test requirements

After all three parts are complete:

```bash
python -m pytest tests/ -v
```

Expected counts:
- 39 Phase 1 tests (unchanged)
- 6 Phase 2 tests (unchanged)
- 3 new trail tests (Part 1)
- 2 new parameter mapping tests (Part 2)
- Total: 50 tests minimum, all passing

New tests go in:
- Trail tests → `tests/test_visualize.py` (append)
- Parameter mapping tests → `tests/test_mapping.py` (append)
- CLI output-name tests → `tests/test_cli.py` (new file, use `subprocess` to call
  `run_sonify.py --help` and confirm new args are present; no full audio render needed
  in unit tests)

---

## Verification commands

```bash
# Trail display — fast preview to confirm stacking
py scripts/run_sonify.py --yes --row-end 50 --trail-rows 5 \
    --output-name trail_test \
    --video-output outputs/trail_test.mp4 \
    --frame-width 640 --frame-height 360

# High channel count simulation — rebin to 128 bins to test scatter→heatmap switch
py scripts/run_sonify.py --yes --row-end 100 --n-bins 128 \
    --output-name heatmap_test \
    --video-output outputs/heatmap_test.mp4

# Parameter mapping — tone driven by depth column
py scripts/run_sonify.py --yes --row-end 200 \
    --tone-source column --tone-column depth \
    --output-name depth_tone \
    --video-output outputs/depth_tone.mp4

# Minimap enabled
py scripts/run_sonify.py --yes --row-end 200 --show-minimap \
    --output-name minimap_test \
    --video-output outputs/minimap_test.mp4

# Full V1 run — everything on
py scripts/run_sonify.py --yes --row-end 200 \
    --trail-rows 5 --show-minimap --show-labels \
    --output-name v1_full \
    --video-output outputs/v1_full.mp4
```

---

## Walkthrough format when done

Same format as Phase 1 and 2 walkthroughs:
- What was built / modified (table)
- Key design decisions (especially the scatter→heatmap threshold and minimap pre-render
  strategy)
- Test counts: X Phase 1 + 6 Phase 2 + Y new = Z total
- Actual observed results for each verification command (did the trail stack correctly,
  did depth-driven tone produce audibly different pitch across rows, did the minimap
  position line track correctly)
- Render timing at 640×360 and 1280×720 with trail enabled