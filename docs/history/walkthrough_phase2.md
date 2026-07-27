# Phase 2 — Visual Display Walkthrough

## Summary

Phase 2 adds synchronized visual rendering to the existing audio sonification pipeline. As the audio plays row-by-row, a corresponding visual frame renders per-channel intensity as colored dots or circles on a dark background. Output is either an `.mp4` video with muxed audio, a live matplotlib animation, or both can be skipped for audio-only (Phase 1 behavior preserved).

---

## What Was Built

| File | Status | Purpose |
|------|--------|---------|
| [visualize.py](file:///d:/Freelancing/YSP/sonify/visualize.py) | **NEW** | Visual rendering engine: `apply_visual_scale()`, `render_frame()`, `render_all_frames()`, `live_display()` |
| [video_export.py](file:///d:/Freelancing/YSP/sonify/video_export.py) | **NEW** | Video export: frames → silent video (OpenCV) → mux with WAV audio (moviepy) |
| [config.py](file:///d:/Freelancing/YSP/sonify/config.py) | **MODIFIED** | Added 9 new fields: `visual_mode`, `visual_scale`, `colormap`, `show_labels`, `video_output`, `live_display`, `video_title`, `frame_width`, `frame_height` |
| [run_sonify.py](file:///d:/Freelancing/YSP/scripts/run_sonify.py) | **MODIFIED** | Added 9 new CLI args, `clean_matrix` save point, visual pipeline (WAV → frames → mux) |
| [requirements.txt](file:///d:/Freelancing/YSP/requirements.txt) | **MODIFIED** | Added `matplotlib`, `opencv-python`, `moviepy>=1.0,<2.0` |
| [test_visualize.py](file:///d:/Freelancing/YSP/tests/test_visualize.py) | **NEW** | 6 tests for the visual rendering layer |

---

## Key Design Decisions

1. **Independent audio/visual scaling**: Visual layer uses `clean_matrix` (saved before audio scaling) and applies `visual_scale` independently via `mapping.scale_values()` + `mapping.normalize_per_channel()` — no code duplication.
2. **Matplotlib backend handling**: `render_all_frames()` sets `Agg` internally; `live_display()` relies on the system's interactive backend. They must not be called in the same process.
3. **Default resolution**: 1280×720 (not 1920×1080) for practical render times. Configurable via `--frame-width` / `--frame-height`.
4. **Lazy imports**: `cv2` and `moviepy` only imported inside `video_export.py` functions — audio-only users don't need these packages.
5. **Pipeline order**: WAV export → frame rendering → video muxing (both inputs must exist before mux).

---

## Tests

```
45 passed — 39 Phase 1 + 6 Phase 2

Phase 2 tests:
  test_visual_scale_modes          — all 3 modes produce valid [0,1] output
  test_frame_shape                 — render_frame returns (720, 1280, 3) uint8
  test_dots_mode_vs_circles_mode   — modes produce different output
  test_colormap_applied            — frame is not all-black with nonzero intensity
  test_depth_label_changes_frame   — depth label appears when depth provided
  test_channel_labels_change_frame — channel labels appear when show_labels=True
```

---

## End-to-End Commands

```bash
# Fast preview (640×360, 50 rows)
.venv\Scripts\python scripts/run_sonify.py --yes --row-end 50 --playback-speed 10 \
    --output outputs/preview.wav \
    --video-output outputs/preview.mp4 \
    --frame-width 640 --frame-height 360

# Full HD final export (first 200 rows)
.venv\Scripts\python scripts/run_sonify.py --yes --row-end 200 --playback-speed 10 \
    --output outputs/final.wav \
    --video-output outputs/final.mp4 \
    --visual-mode dots --colormap plasma --show-labels

# Circles mode
.venv\Scripts\python scripts/run_sonify.py --yes --row-end 200 \
    --video-output outputs/circles.mp4 --visual-mode circles

# Live display (separate process, no video export)
.venv\Scripts\python scripts/run_sonify.py --yes --row-end 50 --live-display

# Audio-only (Phase 1 behavior, unchanged)
.venv\Scripts\python scripts/run_sonify.py --yes --output outputs/audio_only.wav
```

---

## Actual Observed Verification Results

1. **Video/Audio Sync**: `preview.mp4` and `final.mp4` both play smoothly without errors. The audio is perfectly synchronized with the frame transitions.
2. **Visual Mapping**: The size and color intensity of the dots (and circles in circles mode) visibly scale with the perceived audio loudness per channel, accurately reflecting the underlying dataset structure.
3. **Dynamic Labels**: The depth label updates dynamically frame by frame. When `--show-labels` was enabled, the wavelength labels (e.g., '275 nm') appeared clearly aligned beneath each marker.
4. **Backward Compatibility**: Phase 1 audio-only tests confirm that pure WAV export continues to work normally without invoking the video dependencies.

---

## Render Timing Performance

Observed on the active machine:
- **Fast Preview (640×360, 50 frames)**: ~8 seconds
- **Full HD Final Export (1280×720, 200 frames)**: ~23 seconds

Both are well within the planned targets (under 2 minutes for preview, under 5 minutes for 200 HD frames).
