# Phase 2 — Visual Display Implementation Plan (Revised)

## Phase 1 Summary (Codebase Understanding)

Phase 1 built a complete, generic, dataset-agnostic audio sonification pipeline. The flow is:
**load CSV** → **auto-detect band columns** via regex (excluding noise/housekeeping columns) →
**sort rows** by `row_num` or `depth` → **slice** rows → **clean** (NaN→0, negative→0) →
**rebin** (merge adjacent channels via mean) → **scale** (linear/log10/ln with epsilon) →
**normalize per-channel** to [0,1] → **assign frequencies** (log-spaced by index or wavelength)
→ **synthesize** (phase-continuous additive sine with raised-cosine fades) → **export WAV**
or **play through speakers**. All parameters live in `SonificationConfig` dataclass with
validation. The CLI runner in `scripts/run_sonify.py` wires everything together. 39 tests pass
across 4 test files.

**Revision note:** this version fixes four issues in the prior draft: the pipeline sequencing
made WAV export happen before synthesis (impossible); the Agg backend was set at module level
(breaks live display in the same process); the 1920×1080 default resolution would make
frame rendering take hours on the full dataset; and the depth-label test conflated two
independent parameters into one test. `moviepy` is also version-pinned. Changes are marked
**[FIXED]** inline.

---

## Key Integration Points

The Phase 2 visual layer taps into the pipeline at two points:

1. **After step 6 (clean + rebin), before step 7 (audio scale)** — save a copy of the
   cleaned, rebinned matrix here as `clean_matrix`. This is the pre-scaled data that the visual
   path applies its own `visual_scale` to independently of the audio `scale` parameter.
2. **After step 12 (synthesize) and after WAV export** — both the frames list and the WAV
   file on disk must exist before `export_video()` is called. Frame rendering and synthesis
   are independent of each other and can run in either order, but both must complete before
   muxing.

Critical shared data:
- `clean_matrix` — cleaned + rebinned, pre-audio-scale `(n_rows, n_channels)` array. Visual
  path applies `visual_scale` + `normalize_per_channel()` to this independently.
- `depth` column values from the sorted/sliced DataFrame (if column exists) for depth labels.
- `wavelengths_array` computed during frequency assignment for optional channel labels.
- `config.playback_speed` — both audio row-rate and video FPS.

---

## Corrected Pipeline Order

**[FIXED]** The prior draft placed video muxing inside the visual block before synthesis, which
is self-contradictory (the WAV file cannot exist before synthesis happens). The correct
execution order is:

```
Steps 1–6:   load → detect → sort → slice → clean → rebin
             → save clean_matrix = matrix.copy() here
Steps 7–11:  scale (audio) → normalize → assign frequencies → synthesize
             → waveform now exists
Step 12a:    export WAV to disk (required before muxing can happen)
Step 12b:    render all visual frames (independent of audio, can run in parallel in future)
Step 12c:    export_video(frames, wav_path, output_path, fps) — both inputs now exist
```

Frame rendering and audio synthesis are fully independent — they both only need `clean_matrix`
and the config, not each other. What is strictly required is that both the WAV file and the
frames are ready before `export_video()` is called. This ordering must be reflected exactly
in `run_sonify.py`.

---

## Proposed Changes

### Component 1: Config Layer

#### [MODIFY] [config.py](file:///d:/Freelancing/YSP/sonify/config.py)

Add 9 new fields to `SonificationConfig`:

```python
visual_mode: Literal["dots", "circles"] = "dots"
visual_scale: Literal["linear", "log10", "ln"] = "log10"
colormap: str = "plasma"
show_labels: bool = False
video_output: Optional[str] = None
live_display: bool = False
video_title: str = "Sounds of Deep Ice Fluorescence"
frame_width: int = 1280    # [FIXED] default 1280x720, not 1920x1080 — see §performance note
frame_height: int = 720
```

Add validation in `validate()`:
- If `video_output` is set: path must end in `.mp4` or `.avi`; parent directory must exist and
  be writable
- `visual_mode` must be one of `"dots"`, `"circles"`
- `visual_scale` must be one of `"linear"`, `"log10"`, `"ln"`
- `frame_width` and `frame_height` must both be positive integers

---

### Component 2: Visual Rendering Engine

#### [NEW] [visualize.py](file:///d:/Freelancing/YSP/sonify/visualize.py)

Core visual rendering module. Fully generic — no hardcoded band count or column names.

**[FIXED] Backend handling:**
Do NOT call `matplotlib.use('Agg')` at module level or at import time.
`matplotlib.use()` must be called before pyplot is imported, and once set it cannot be changed
in the same process. Setting it at module level would silently kill the live display mode's
ability to open an interactive window regardless of what it tries later.

Instead, set the backend conditionally inside `render_all_frames()` only:

```python
def render_all_frames(...) -> list[np.ndarray]:
    import matplotlib
    matplotlib.use("Agg")          # safe here: called before pyplot import in this path
    import matplotlib.pyplot as plt
    ...
```

`live_display()` must NOT call `matplotlib.use()` at all — it relies on whatever interactive
backend the user's system provides (Qt5Agg, TkAgg, etc.), which matplotlib selects
automatically. If `render_all_frames()` has already been called in the same process before
`live_display()` is reached, the Agg backend will already be set and the animation window
cannot open. The CLI runner must therefore ensure that `render_all_frames()` and
`live_display()` are never both called in the same process invocation — the spec already
requires this ("if both `--live-display` and `--video-output` are set, only video is
generated"), so enforce it as the first check in the CLI's visual branch.

**[FIXED] Performance note — frame resolution:**
Matplotlib's Agg renderer takes roughly 100–500ms per 1280×720 frame depending on the
system, with time scaling roughly with pixel count. At 1920×1080 (the prior draft's default)
this is ~4x more pixels, meaning the full 4000-row dataset would take 7–33 minutes just for
frame generation before synthesis or muxing begins. The default is lowered to **1280×720**,
which is still true HD and is a standard streaming/export resolution. Expose
`--frame-width` / `--frame-height` CLI args so users can trade resolution for speed:
use 640×360 for fast preview runs, 1920×1080 for final export. Document these tradeoffs
explicitly in the README.

**Public API:**

```python
def apply_visual_scale(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Apply visual-specific scaling (linear/log10/ln) then re-normalize to [0,1].

    Delegates to mapping.scale_values() + mapping.normalize_per_channel()
    so no scaling logic is duplicated. Input matrix must be pre-clipped
    (guaranteed by preprocess.clean() upstream).
    Precondition: matrix values are non-negative.
    """

def render_frame(
    amplitudes: np.ndarray,              # 1-D, length n_channels, values in [0,1]
    mode: str = "dots",                  # "dots" or "circles"
    colormap: str = "plasma",
    depth: float | None = None,          # current depth (m) for label
    wavelengths: np.ndarray | None = None,
    show_labels: bool = False,
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1280,
    fig_height: int = 720,
) -> np.ndarray:
    """Render one frame as an RGB numpy array (H, W, 3), dtype uint8.
    Caller is responsible for setting the matplotlib backend before calling this.
    """

def render_all_frames(
    amplitude_matrix: np.ndarray,        # (n_rows, n_channels), already visual-scaled + normalized
    mode: str = "dots",
    colormap: str = "plasma",
    depths: np.ndarray | None = None,
    wavelengths: np.ndarray | None = None,
    show_labels: bool = False,
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1280,
    fig_height: int = 720,
) -> list[np.ndarray]:
    """Render all frames. Sets Agg backend internally. Returns list of RGB arrays."""

def live_display(
    amplitude_matrix: np.ndarray,
    waveform: np.ndarray,
    sample_rate: int,
    playback_speed: float,
    mode: str = "dots",
    colormap: str = "plasma",
    depths: np.ndarray | None = None,
    wavelengths: np.ndarray | None = None,
    show_labels: bool = False,
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1280,
    fig_height: int = 720,
) -> None:
    """Show matplotlib FuncAnimation synchronized with audio playback.
    Does NOT set matplotlib backend — relies on system interactive backend.
    Must not be called in the same process as render_all_frames().
    """
```

**Frame layout (both modes):**
- Dark background (`#0A0A0A`)
- Channels laid out left-to-right in band/index order, evenly spaced horizontally, centered
  vertically
- Title at top center; depth label at bottom-left (shown whenever `depth` is not None,
  independent of `show_labels`)
- **Dots mode:** fixed-size scatter markers, color from `colormap(intensity)`
- **Circles mode:** circle radius proportional to intensity (min visible radius to max radius),
  color from `colormap(intensity)`
- **Channel labels** (band index or wavelength nm) below each dot/circle:
  only when `show_labels=True` (independent parameter from depth label)

---

### Component 3: Video Export

#### [NEW] [video_export.py](file:///d:/Freelancing/YSP/sonify/video_export.py)

**Lazy imports:** `cv2` and `moviepy` imported inside functions, not at module level.
This keeps them fully optional — users without these packages can still use audio-only.

**[FIXED] moviepy version pin:** `moviepy>=1.0,<2.0`. The v1.x and v2.x APIs are completely
different (v2 changed `VideoFileClip`, `AudioFileClip`, `CompositeAudioClip` and more).
Without a pin, `pip install moviepy` on a fresh environment may pull v2 and break video export
silently. Pin v1 until v2 compatibility is explicitly tested and documented.

**Public API:**

```python
def export_video(
    frames: list[np.ndarray],   # RGB frames (H, W, 3), uint8
    wav_path: str,              # path to WAV already on disk — must exist before this is called
    output_path: str,           # final .mp4 or .avi
    fps: float,                 # = playback_speed (rows/sec = frames/sec)
) -> None:
```

**Implementation steps inside `export_video()`:**
1. Import `cv2` (opencv-python); convert each RGB frame to BGR (`frame[:, :, ::-1]`) —
   OpenCV's `VideoWriter` expects BGR, not RGB.
2. Write all BGR frames to a temporary silent video file using `cv2.VideoWriter`.
3. Import `moviepy.editor`; use `VideoFileClip` + `AudioFileClip` to combine the silent
   video with `wav_path` into the final output file.
4. Delete the temporary silent video file.
5. Print confirmation with the output path and file size.

**Temporary WAV handling:**
When `--video-output` is set but `--output` (audio WAV) is not, the pipeline must still
produce a WAV on disk for muxing. The CLI runner generates the WAV to a `tempfile`-managed
path, passes it to `export_video()`, then deletes it after muxing completes. If `--output`
is also set, reuse that path directly (no temp file needed, no cleanup required).

---

### Component 4: Live Display (Optional, Secondary)

Implemented as `live_display()` inside `sonify/visualize.py` (see Component 2 for signature).

- Uses `matplotlib.animation.FuncAnimation` with `interval = 1000 / playback_speed` ms
- Starts audio playback via `sounddevice` in a background thread in parallel with the animation
- Documented as best-effort sync (matplotlib animation timing is not sample-accurate)
- **Must never be called in the same process as `render_all_frames()`** due to the Agg
  backend conflict described in Component 2
- The CLI runner enforces this: `--video-output` takes priority; `--live-display` is ignored
  if `--video-output` is also set

---

### Component 5: CLI Integration

#### [MODIFY] [run_sonify.py](file:///d:/Freelancing/YSP/scripts/run_sonify.py)

**New argparse arguments:**

```
--visual-mode {dots,circles}       default: dots
--visual-scale {linear,log10,ln}   default: log10
--colormap COLORMAP                default: plasma
--show-labels                      flag, off by default
--video-output PATH                path for .mp4 or .avi output
--live-display                     flag, off by default
--video-title TITLE                default: "Sounds of Deep Ice Fluorescence"
--frame-width INT                  default: 1280
--frame-height INT                 default: 720
```

**[FIXED] Corrected pipeline in run_sonify.py:**

```python
# Steps 1-6: load, detect, sort, slice, clean, rebin
df = load_csv(config.input_path)
band_cols, band_indices = detect_band_columns(df)
confirm_with_user(band_cols, skip_confirm=config.yes)
df = sort_by_row_order(df)
df = df.iloc[config.row_start:config.row_end]
matrix = clean(df, band_cols)
matrix = rebin(matrix, config.n_bins) if config.n_bins else matrix

# Save copy BEFORE audio scaling so visual path can scale independently
clean_matrix = matrix.copy()

# Steps 7-11: audio path
matrix = scale_values(matrix, config.scale)
matrix = normalize_per_channel(matrix)
freqs = assign_frequencies(n_channels, config.min_freq, config.max_freq,
                           config.freq_mode, wavelengths_array)
waveform = synthesize(matrix, freqs, 1.0 / config.playback_speed, config.sample_rate)

# Step 12a: WAV export (must happen before video muxing)
if config.video_output or config.output:
    wav_path = config.output or tempfile.mktemp(suffix=".wav")
    export_wav(waveform, config.sample_rate, wav_path)
    temp_wav = (config.output is None)  # track whether we own this file

# Step 12b: visual frames (independent of audio, needs clean_matrix)
if config.video_output:  # video takes priority over live_display
    from sonify.visualize import apply_visual_scale, render_all_frames
    visual_matrix = apply_visual_scale(clean_matrix, config.visual_scale)
    depths = df["depth"].values if "depth" in df.columns else None
    frames = render_all_frames(
        visual_matrix,
        mode=config.visual_mode,
        colormap=config.colormap,
        depths=depths,
        wavelengths=wavelengths_array,
        show_labels=config.show_labels,
        title=config.video_title,
        fig_width=config.frame_width,
        fig_height=config.frame_height,
    )
    # Step 12c: mux — both frames and wav_path now exist
    from sonify.video_export import export_video
    export_video(frames, wav_path, config.video_output, fps=config.playback_speed)
    if temp_wav:
        os.remove(wav_path)

elif config.live_display:
    from sonify.visualize import apply_visual_scale, live_display
    visual_matrix = apply_visual_scale(clean_matrix, config.visual_scale)
    depths = df["depth"].values if "depth" in df.columns else None
    live_display(visual_matrix, waveform, config.sample_rate, config.playback_speed,
                 mode=config.visual_mode, colormap=config.colormap,
                 depths=depths, wavelengths=wavelengths_array,
                 show_labels=config.show_labels, title=config.video_title,
                 fig_width=config.frame_width, fig_height=config.frame_height)

elif config.output:
    pass  # WAV already written above

else:
    play(waveform, config.sample_rate)
```

#### [MODIFY] [__init__.py](file:///d:/Freelancing/YSP/sonify/__init__.py)

Do NOT import `visualize` or `video_export` at module level. They have optional dependencies
(`matplotlib`, `opencv-python`, `moviepy`) and are imported lazily inside the CLI runner only
when the relevant flags are set. Keep all Phase 1 imports unchanged.

---

### Component 6: Dependencies

#### [MODIFY] [requirements.txt](file:///d:/Freelancing/YSP/requirements.txt)

```
matplotlib
opencv-python
moviepy>=1.0,<2.0
```

---

### Component 7: Tests

#### [NEW] [test_visualize.py](file:///d:/Freelancing/YSP/tests/test_visualize.py)

**[FIXED]** 6 tests (prior draft had 5; depth-label test is now split into two separate tests
because depth label and channel labels are independent parameters that should be verified
independently):

| Test | What it verifies |
|------|-----------------|
| `test_frame_shape` | Given a (1×N) amplitude matrix, `render_frame()` returns an RGB array of the expected pixel dimensions (H=720, W=1280 for default resolution) |
| `test_dots_mode_vs_circles_mode` | Both modes run on synthetic data without error; output arrays are not identical (confirms modes produce visually different results) |
| `test_visual_scale_modes` | All three `apply_visual_scale()` modes (linear, log10, ln) run without error on data containing zeros; output is in [0,1] for all three |
| `test_colormap_applied` | Rendered frame with nonzero intensities is not all-black (confirms colormap lookup is actually being applied) |
| `test_depth_label_changes_frame` | Frame rendered with a `depth` value differs from frame rendered with `depth=None` — depth label is shown whenever depth is provided, independent of `show_labels` |
| `test_channel_labels_change_frame` | Frame rendered with `show_labels=True` differs from `show_labels=False` — channel labels are a separate parameter from depth label, verified independently |

All tests use synthetic data, mock the matplotlib backend where needed, and write no actual
video files to disk.

---

## Execution Order

1. `config.py` — add 9 new fields and validation
2. `sonify/visualize.py` — rendering engine with correct backend handling
3. `sonify/video_export.py` — video muxing with lazy imports and moviepy pin
4. `scripts/run_sonify.py` — corrected pipeline order and new CLI args
5. `requirements.txt` — add dependencies with version pin
6. `tests/test_visualize.py` — 6 new tests

---

## Verification Plan

### Automated Tests

```bash
python -m pytest tests/ -v
# Expected: 39 Phase 1 tests + 6 Phase 2 tests = 45 total
```

### Manual Verification

```bash
# Fast preview (640x360, 50 rows)
python scripts/run_sonify.py --yes --row-end 50 --playback-speed 10 \
    --output outputs/preview.wav \
    --video-output outputs/preview.mp4 \
    --frame-width 640 --frame-height 360

# Full HD final export (first 200 rows — manageable render time)
python scripts/run_sonify.py --yes --row-end 200 --playback-speed 10 \
    --output outputs/final.wav \
    --video-output outputs/final.mp4 \
    --visual-mode dots --colormap plasma --show-labels

# Circles mode comparison
python scripts/run_sonify.py --yes --row-end 200 \
    --video-output outputs/circles.mp4 --visual-mode circles

# Test live display (separate process — do not combine with video export run)
python scripts/run_sonify.py --yes --row-end 50 --live-display
```

**What to inspect:**
1. `preview.mp4` plays without error and audio is synchronized with visual frames
2. Dot/circle brightness and size match perceived audio loudness (a loud moment in the audio
   should correspond to bright, large markers in that frame)
3. Depth label updates every frame as the instrument descends
4. Channel labels are visible when `--show-labels` is set, absent otherwise
5. `--frame-width 640 --frame-height 360` preview renders in reasonable time (under 2 minutes
   for 50 frames); 1280×720 at 200 frames should complete in under 5 minutes on a modern
   machine — document actual timings observed in the final walkthrough
6. Verify no Phase 1 functionality is broken: run an audio-only command without any visual
   flags and confirm it still works exactly as before