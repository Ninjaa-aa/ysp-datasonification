# Phase 2 — Visual Display Implementation Plan

## Phase 1 Summary (Codebase Understanding)

Phase 1 built a complete, generic, dataset-agnostic audio sonification pipeline. The flow is: **load CSV** → **auto-detect band columns** via regex (excluding noise/housekeeping columns) → **sort rows** by `row_num` or `depth` → **slice** rows → **clean** (NaN→0, negative→0) → **rebin** (merge adjacent channels via mean) → **scale** (linear/log10/ln with epsilon) → **normalize per-channel** to [0,1] → **assign frequencies** (log-spaced by index or wavelength) → **synthesize** (phase-continuous additive sine with raised-cosine fades) → **export WAV** or **play through speakers**. All parameters live in `SonificationConfig` dataclass with validation. The CLI runner in `scripts/run_sonify.py` wires everything together. 39 tests pass across 4 test files.

---

## Key Integration Points

The Phase 2 visual layer needs to tap into the pipeline **after step 10 (normalize)** and **before step 12 (synthesize)**. The critical data flow is:

1. **Normalized amplitude matrix** — `matrix` after `normalize_per_channel()` at [run_sonify.py:158](file:///d:/Freelancing/YSP/scripts/run_sonify.py#L158). This is the `(n_rows, n_channels)` array in [0,1] that drives both audio amplitudes AND visual intensity. The spec explicitly requires sharing this same array.
2. **Depth values** — extracted from the sorted/sliced DataFrame's `depth` column (if it exists) for the depth label overlay.
3. **Wavelength values** — the `wavelengths_array` computed at [run_sonify.py:142-151](file:///d:/Freelancing/YSP/scripts/run_sonify.py#L142-L151) for optional channel labels.
4. **Playback speed** — `config.playback_speed` determines both audio row-rate and video FPS.

> [!IMPORTANT]
> The visual layer applies its own `visual_scale` (linear/log10/ln) **independently** from the audio `scale`. This means we need the **cleaned + rebinned matrix BEFORE audio scaling** so the visual layer can apply its own scaling + normalization. Currently the pipeline scales in-place at line 155. We'll need to save a copy of the pre-scaled matrix for the visual path.

---

## Proposed Changes

### Component 1: Config Layer

#### [MODIFY] [config.py](file:///d:/Freelancing/YSP/sonify/config.py)

Add 7 new fields to `SonificationConfig`:

```python
visual_mode: Literal["dots", "circles"] = "dots"
visual_scale: Literal["linear", "log10", "ln"] = "log10"
colormap: str = "plasma"
show_labels: bool = False
video_output: Optional[str] = None
live_display: bool = False
video_title: str = "Sounds of Deep Ice Fluorescence"
```

Add validation in `validate()`:
- If `video_output` is set, check the path ends in `.mp4` or `.avi`
- If `video_output` is set, check the parent directory exists and is writable
- Validate `visual_mode` is one of `"dots"`, `"circles"`
- Validate `visual_scale` is one of `"linear"`, `"log10"`, `"ln"`

---

### Component 2: Visual Rendering Engine

#### [NEW] [visualize.py](file:///d:/Freelancing/YSP/sonify/visualize.py)

The core visual rendering module. Fully generic — no hardcoded band count or column names.

**Key design decisions:**

- Uses `matplotlib` with the `Agg` backend for frame rendering (no GUI needed for video export)
- Each frame is rendered as a matplotlib figure → RGB numpy array via `fig.canvas.tostring_rgb()`
- Frame dimensions: configurable, default 1920×1080 (Full HD)

**Public API:**

```python
def apply_visual_scale(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Apply visual-specific scaling (linear/log10/ln) then re-normalize to [0,1].
    
    Uses mapping.scale_values() + mapping.normalize_per_channel() 
    so scaling logic is not duplicated.
    """

def render_frame(
    amplitudes: np.ndarray,          # 1-D array, length n_channels, values in [0,1]
    mode: str = "dots",              # "dots" or "circles"  
    colormap: str = "plasma",        # any matplotlib colormap name
    depth: float | None = None,      # current depth value for label
    wavelengths: np.ndarray | None = None,  # per-channel wavelengths for labels
    show_labels: bool = False,       # whether to show channel labels
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1920,
    fig_height: int = 1080,
) -> np.ndarray:
    """Render one frame as an RGB numpy array (H, W, 3), dtype uint8."""

def render_all_frames(
    amplitude_matrix: np.ndarray,    # (n_rows, n_channels), already visual-scaled
    mode: str = "dots",
    colormap: str = "plasma",
    depths: np.ndarray | None = None,
    wavelengths: np.ndarray | None = None,
    show_labels: bool = False,
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1920,
    fig_height: int = 1080,
) -> list[np.ndarray]:
    """Render all frames. Returns list of RGB arrays."""
```

**Frame layout (both modes):**
- Dark background (#0A0A0A or black)
- Channels laid out left-to-right, evenly spaced horizontally, centered vertically
- Title at top center, depth label at bottom-left
- **Dots mode**: Fixed-size circles (scatter plot markers), color from colormap(intensity)
- **Circles mode**: Circle radius proportional to intensity (scaled from a min visible radius to a max radius), color from colormap(intensity)
- Channel labels (band index or wavelength in nm) below each dot/circle when `show_labels=True`

---

### Component 3: Video Export

#### [NEW] [video_export.py](file:///d:/Freelancing/YSP/sonify/video_export.py)

**Lazy imports**: `cv2` and `moviepy` are imported inside functions, not at module level. This keeps them optional — users without these packages can still use audio-only.

**Public API:**

```python
def export_video(
    frames: list[np.ndarray],    # list of RGB frames (H, W, 3), uint8
    wav_path: str,               # path to the WAV file (already exported)
    output_path: str,            # final .mp4 or .avi path
    fps: float,                  # frames per second = playback_speed
) -> None:
    """Write frames as video, then mux with WAV audio to produce final output."""
```

**Implementation approach:**
1. Write frames to a temporary video file using `cv2.VideoWriter` (note: OpenCV uses BGR, so we convert RGB→BGR)
2. Use `moviepy` to combine the silent video with the WAV audio into the final `.mp4`/`.avi`
3. Clean up the temporary silent video file

---

### Component 4: Live Display (Optional, Secondary)

Implemented inside [visualize.py](file:///d:/Freelancing/YSP/sonify/visualize.py) as an additional function:

```python
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
) -> None:
    """Show matplotlib FuncAnimation synchronized with audio playback."""
```

- Uses `matplotlib.animation.FuncAnimation` with interval = `1000 / playback_speed` ms
- Starts audio playback via `sounddevice` in parallel
- Documented as best-effort sync (video export gives better sync)
- If both `--live-display` and `--video-output` are set, only video is generated (per spec)

---

### Component 5: CLI Integration

#### [MODIFY] [run_sonify.py](file:///d:/Freelancing/YSP/scripts/run_sonify.py)

**New argparse arguments** (added to `build_parser()`):

```
--visual-mode {dots,circles}       default: dots
--visual-scale {linear,log10,ln}   default: log10
--colormap COLORMAP                default: plasma
--show-labels                      flag, off by default
--video-output PATH                path for .mp4 output
--live-display                     flag, off by default
--video-title TITLE                default: "Sounds of Deep Ice Fluorescence"
```

**Pipeline modifications** (after step 10, normalize):

```python
# ── Visual pipeline (Phase 2) ────────────────────────────────
if config.video_output or config.live_display:
    # Save pre-scaled clean matrix for visual path
    # Apply visual-specific scaling
    from sonify.visualize import apply_visual_scale, render_all_frames
    visual_matrix = apply_visual_scale(clean_matrix, config.visual_scale)
    
    # Extract depth values if available
    depths = df["depth"].values if "depth" in df.columns else None
    
    # Render all frames
    frames = render_all_frames(
        visual_matrix, mode=config.visual_mode,
        colormap=config.colormap, depths=depths,
        wavelengths=wavelengths_array,
        show_labels=config.show_labels,
        title=config.video_title,
    )
    
    if config.video_output:
        # Export WAV first (needed for muxing)
        wav_path = config.output or config.video_output.rsplit('.', 1)[0] + '.wav'
        from sonify.video_export import export_video
        export_video(frames, wav_path, config.video_output, fps=config.playback_speed)
    elif config.live_display:
        from sonify.visualize import live_display
        live_display(visual_matrix, waveform, config.sample_rate, ...)
```

> [!IMPORTANT]
> **Data flow for separate audio/visual scaling**: We need to preserve the cleaned+rebinned matrix (before audio `scale_values()`) so the visual path can apply `visual_scale` independently. I'll save a reference (`clean_matrix = matrix.copy()`) at line ~154 before the audio scaling happens.

#### [MODIFY] [__init__.py](file:///d:/Freelancing/YSP/sonify/__init__.py)

- Do NOT import `visualize` or `video_export` at module level (they have optional dependencies)
- Keep Phase 1 imports unchanged

---

### Component 6: Dependencies

#### [MODIFY] [requirements.txt](file:///d:/Freelancing/YSP/requirements.txt)

Add:
```
matplotlib
opencv-python
moviepy
```

---

### Component 7: Tests

#### [NEW] [test_visualize.py](file:///d:/Freelancing/YSP/tests/test_visualize.py)

5 tests as specified:

| Test | What it verifies |
|------|-----------------|
| `test_frame_shape` | Given a (1×N) amplitude matrix, `render_frame()` returns an RGB array of expected pixel dimensions |
| `test_dots_mode_vs_circles_mode` | Both modes run on synthetic data; outputs differ (modes are actually different) |
| `test_visual_scale_modes` | All three visual scale modes (linear, log10, ln) run without error on data containing zeros |
| `test_colormap_applied` | Rendered frame is not all-black when intensities are nonzero |
| `test_depth_label_present` | When depth values supplied + `show_labels=True`, frame differs from `show_labels=False` |

All tests use synthetic data, no real display, no actual video files written.

---

## Execution Order

1. **Config** — Add fields + validation to `SonificationConfig`
2. **Visualize** — Build `sonify/visualize.py` (render engine)
3. **Video Export** — Build `sonify/video_export.py` (video muxing)
4. **CLI** — Wire new args and pipeline steps into `run_sonify.py`
5. **Dependencies** — Update `requirements.txt`
6. **Tests** — Write `tests/test_visualize.py`
7. **Verify** — Run all tests (39 Phase 1 + 5 Phase 2 = 44 expected)

---

## Verification Plan

### Automated Tests
```bash
# Run all tests — must see 39 Phase 1 tests pass + 5 new Phase 2 tests
python -m pytest tests/ -v
```

### Manual Verification
- Generate a sample video:
  ```bash
  python scripts/run_sonify.py --input data/raw/...csv --yes --output outputs/test.wav --video-output outputs/test.mp4 --playback-speed 10 --visual-mode dots --colormap plasma
  ```
- Verify the `.mp4` file plays correctly and audio is synchronized with visual frames
- Check that dot/circle intensity visually matches audio dynamics
- Test both `dots` and `circles` modes
- Test `--show-labels` flag

---

## Open Questions

> [!IMPORTANT]
> **Frame resolution**: The spec doesn't specify frame dimensions. I'll default to **1920×1080** (Full HD) which is standard for video. Should I use a different resolution?

> [!NOTE]
> **Temporary WAV for video muxing**: When `--video-output` is set but `--output` (WAV) is not, the pipeline still needs a WAV file for muxing audio into video. I'll generate a temporary WAV, mux it, then clean up. If `--output` IS set, I'll reuse that WAV file directly.
