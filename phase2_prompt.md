# Phase 2 — Visual Display for Borehole Sonification Toolkit

## Step 0: Read the codebase before writing anything

Before proposing or implementing anything, read every file in this project. Start with:

```
sonify/config.py
sonify/data_io.py
sonify/band_detect.py
sonify/preprocess.py
sonify/mapping.py
sonify/synth.py
sonify/playback.py
sonify/export.py
scripts/run_sonify.py
tests/test_band_detect.py
tests/test_preprocess.py
tests/test_mapping.py
tests/test_synth.py
```

Then read `PHASE1_SPEC.md` for the original design rationale. Do not write a single line of code until you have read all of these. After reading, produce a one-paragraph summary of what Phase 1 built and how the pipeline flows, so I can confirm you have understood it correctly before Phase 2 begins.

---

## Project background (for context)

This is the BMSIS YSP "Sounds of Deep Ice Fluorescence" project under Dr. Michael Malaska. The goal of the overall project is an open-source, generic Python toolkit that can sonify any multi-channel tabular dataset. The test case is a 32-channel fluorescence spectral scan taken during descent down a Greenland ice borehole (Summit Station, July 4, 2019). Each row of the CSV is one depth point; each of the 32 Band columns is a fluorescence intensity at a different UV wavelength (275–446 nm, evenly spaced).

Phase 1 built the complete audio pipeline:
- Generic CSV loading and auto-detection of band/channel columns via regex
- Preprocessing: NaN/negative cleaning, row sorting, rebinning (N channels → M bins by contiguous-group averaging)
- Mapping: three intensity-scaling modes (linear, log10, ln), per-channel normalization to [0,1], log-spaced frequency assignment (index mode or wavelength mode using the WATSON band table)
- Synthesis: phase-continuous additive sine synthesis with capped raised-cosine fades to prevent clicking at row boundaries
- Output: WAV export (`scipy.io.wavfile`) and live playback (`sounddevice` → `simpleaudio` fallback)
- CLI: `scripts/run_sonify.py` with `argparse`, all parameters in `SonificationConfig` dataclass with validation
- 39 unit tests, all passing

The pipeline runs as: load → detect bands → sort rows → slice → clean → rebin → scale → normalize → assign frequencies → synthesize → export/play.

**Critical design constraint carried from Phase 1:** this toolkit must remain generic and dataset-agnostic. Do not hardcode 32 bands, these specific column names, or this specific CSV anywhere in the engine (`sonify/`). The `scripts/` layer can have dataset-specific defaults, but the engine must work on any multi-channel CSV.

---

## Phase 2 goal

Add a synchronized visual display to the sonification. As audio plays row by row, a corresponding visual frame renders for each row, showing per-channel intensity as either colored dots or colored circles (with size proportional to intensity). The final deliverable is either a live animated display during playback, a video output file (`.mp4` or `.avi`) synchronized with the WAV audio, or both.

---

## What Phase 2 must implement

### 1. Visual display module — `sonify/visualize.py` (new file)

The engine module. Must be fully generic (no hardcoded band count or column names).

**Frame rendering:**
- Each frame corresponds to one row of data (same row the audio is playing)
- Display the channels laid out left to right in band/index order
- Two display modes, user-selectable:
  - `dots` — fixed-size dots, color encodes intensity (colormap, e.g. viridis or plasma)
  - `circles` — circle radius scales with intensity (normalized [0,1] from the same per-channel normalized amplitude matrix Phase 1 already produces); color can be fixed or also intensity-mapped
- Intensity values fed to the renderer must be the **same normalized amplitude matrix** already computed by `mapping.normalize_per_channel()` in Phase 1 — do not re-normalize separately in the visual layer; share the same data array the audio uses so audio and video are always driven by identical values
- Background: dark (black or very dark grey) — fluorescence data on a dark background is both more readable and more aesthetically appropriate for a sci-art project
- Channel labels: optionally show band index or wavelength (nm) below each dot/circle if the wavelength table was loaded; off by default, enabled via a flag
- Depth label: show current depth (m) in the frame if a depth column exists in the data, so the viewer knows where in the borehole the current frame is
- Title: show a configurable title string in the frame (default: "Sounds of Deep Ice Fluorescence")

**Intensity scaling in the visual layer:**
- The phase 2 spec explicitly asks for log10 and log(e) scaling factors for the visual display, separate from the audio scaling already in Phase 1
- Add a `visual_scale: Literal["linear", "log10", "ln"] = "log10"` parameter that applies only to the visual rendering (the amplitude values fed to dot color / circle radius), independent of `--scale` which controls audio
- This means audio and visual can use different scaling modes simultaneously, which is useful: e.g. linear audio with log10 visual coloring

**Colormap:**
- Default: `plasma` (perceptually uniform, dark-background-friendly, visually striking for a sci-art piece)
- User-selectable via `--colormap` (any valid matplotlib colormap name)
- Map normalized [0,1] intensity to colormap RGBA for dot/circle color

### 2. Video export — `sonify/video_export.py` (new file)

- `export_video(frames, waveform, sample_rate, output_path, fps)` where `frames` is a list/array of rendered RGB frames (one per row) and `fps` is derived from `playback_speed` (rows per second = frames per second)
- Use `opencv-python` (`cv2`) to write the video, then use `moviepy` or `ffmpeg-python` to mux the WAV audio track into the video file, producing a single synchronized `.mp4`
- Frame rate = `playback_speed` (rows per second), so a 10 rps sonification produces a 10 fps video — this keeps audio and video in perfect sync by construction
- Output path controlled by `--video-output` CLI arg (separate from `--output` which is the audio-only WAV)
- If video output is not requested, this module is never imported (keep it optional so users without `opencv-python` can still use the audio-only pipeline)

### 3. Live animated display (optional, secondary to video export)

- If `--live-display` flag is set and no `--video-output` is requested, show a matplotlib `FuncAnimation` window that advances one frame per row as audio plays
- This is best-effort: live sync between matplotlib animation and sounddevice playback is inherently imperfect, so document that video export + an external player gives better sync than the live display
- If both `--live-display` and `--video-output` are set, generate the video and skip the live display

### 4. Config additions — `sonify/config.py` (modify)

Add to `SonificationConfig`:
- `visual_mode: Literal["dots", "circles"] = "dots"`
- `visual_scale: Literal["linear", "log10", "ln"] = "log10"`
- `colormap: str = "plasma"`
- `show_labels: bool = False`
- `video_output: str | None = None`
- `live_display: bool = False`
- `video_title: str = "Sounds of Deep Ice Fluorescence"`

Add validation: if `video_output` is set, check the path ends in `.mp4` or `.avi` and the directory is writable.

### 5. CLI additions — `scripts/run_sonify.py` (modify)

Add new argparse arguments:
```
--visual-mode {dots,circles}       default: dots
--visual-scale {linear,log10,ln}   default: log10
--colormap COLORMAP                default: plasma
--show-labels                      flag, off by default
--video-output PATH                path for .mp4 output
--live-display                     flag, off by default
--video-title TITLE                default: "Sounds of Deep Ice Fluorescence"
```

Pipeline additions after the existing Phase 1 steps:
- After step 11 (synthesize): if `video_output` or `live_display` is requested, render all frames via `visualize.py` (pass the normalized amplitude matrix, depth values if available, wavelengths if loaded)
- If `video_output`: call `video_export.export_video(frames, waveform, sample_rate, video_output, fps=playback_speed)`
- If `live_display`: run matplotlib animation
- Existing audio-only path (`--output` WAV or live speaker playback) is unchanged

### 6. Required libraries (add to `requirements.txt`)

```
matplotlib
opencv-python
moviepy
```

`matplotlib` is needed for colormap lookups and the optional live display. `opencv-python` and `moviepy` are needed for video export only; guard their import inside `video_export.py` so the rest of the pipeline works if they are not installed.

---

## Tests to add (`tests/test_visualize.py`)

Write new tests for the visual layer. Do not use a real display or write actual video files in the tests; mock/patch as needed.

- `test_frame_shape`: given a (1 × N) amplitude matrix, `render_frame()` returns an RGB array of the expected pixel dimensions
- `test_dots_mode_vs_circles_mode`: both modes run without error on synthetic data; outputs are not identical (confirming the modes actually differ)
- `test_visual_scale_modes`: all three visual scale modes (`linear`, `log10`, `ln`) run without error on data containing zeros
- `test_colormap_applied`: rendered frame is not all-black when intensities are nonzero (confirms colormap is being applied)
- `test_depth_label_present`: when depth values are supplied and `show_labels=True`, the rendered frame differs from one with `show_labels=False` (pixel-level check)

---

## Constraints and things NOT to do

- Do not break any of the 39 existing Phase 1 tests. Run `pytest tests/` after implementing Phase 2 and confirm all 39 still pass plus the new visual tests.
- Do not re-normalize the amplitude matrix inside `visualize.py`. Use the same matrix the audio pipeline already produced.
- Do not hardcode 32 channels, specific band names, or this dataset's column layout anywhere in `sonify/visualize.py` or `sonify/video_export.py`.
- Do not make video export a required dependency of the audio pipeline. A user who only wants WAV output should not need `opencv-python` or `moviepy` installed.
- Keep `visual_scale` and `scale` (audio scaling) as fully independent parameters. They can be set to different values simultaneously.

---

## Deliverable format

When done, provide a walkthrough in the same format as the Phase 1 walkthrough:
- What was built (module table)
- What was modified
- New tests added and counts
- End-to-end commands to generate a sample video and verify sync
- What to manually inspect (watch the video, check dot/circle intensity matches audio dynamics)