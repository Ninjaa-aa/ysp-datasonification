#!/usr/bin/env python
"""
CLI entry point for the borehole sonification toolkit (Phases 1–4).

This script wires together the generic ``sonify`` engine with
dataset-specific defaults for the ice borehole fluorescence example.
The engine itself is dataset-agnostic.

Usage
-----
    py scripts/run_sonify.py --input data/raw/...csv --yes --output outputs/test.wav
    py scripts/run_sonify.py --yes --video-output outputs/preview.mp4
    py scripts/run_sonify.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

# Ensure the project root is on sys.path so ``sonify`` package imports work
# when running ``python scripts/run_sonify.py`` from the project root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from sonify.config import SonificationConfig, ParameterMap
from sonify.data_io import load_csv
from sonify.band_detect import detect_band_columns, confirm_with_user
from sonify.preprocess import sort_by_row_order, clean, rebin, rebin_wavelengths
from sonify.mapping import (
    scale_values,
    normalize_per_channel,
    assign_frequencies,
    load_wavelength_table,
    map_tone_from_column,
    apply_intensity_column,
)
from sonify.synth import synthesize
from sonify.playback import play
from sonify.export import export_wav


# ── Structured logging helper ─────────────────────────────────────────────

def log(stage: str, message: str) -> None:
    """Print a pipeline stage message with consistent formatting."""
    print(f"[{stage:<8s}] {message}")


# ── Dataset-specific defaults ─────────────────────────────────────────────
_DEFAULT_INPUT = os.path.join(
    _PROJECT_ROOT,
    "data", "raw",
    "2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv",
)
_DEFAULT_WAVELENGTH = os.path.join(
    _PROJECT_ROOT, "data", "reference", "watson_band_wavelengths.csv"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sonify a multi-channel CSV dataset (Phases 1-4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Phase 1 arguments ───────────────────────────────────────────────
    p.add_argument(
        "--input", default=_DEFAULT_INPUT,
        help="Path to the input CSV file",
    )
    p.add_argument("--row-start", type=int, default=None, help="First row index to include (0-based)")
    p.add_argument("--row-end", type=int, default=None, help="Last row index (exclusive)")
    p.add_argument("--n-bins", type=int, default=None, help="Number of output frequency bins (channels); default = detected count")
    p.add_argument("--min-freq", type=float, default=150.0, help="Lowest tone frequency in Hz")
    p.add_argument("--max-freq", type=float, default=2500.0, help="Highest tone frequency in Hz")
    p.add_argument("--playback-speed", type=float, default=10.0, help="Rows per second")
    p.add_argument("--volume", type=float, default=0.8, help="Master gain 0.0-1.0")
    p.add_argument("--scale", choices=["linear", "log10", "ln"], default="log10", help="Intensity scaling mode (audio)")
    p.add_argument("--freq-mode", choices=["index", "wavelength"], default="index", help="Frequency assignment mode")
    p.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    p.add_argument("--output", type=str, default=None, help="Output .wav path (omit to play through speakers)")
    p.add_argument("--yes", action="store_true", help="Skip interactive band-confirmation prompt")
    p.add_argument(
        "--wavelength-path", default=_DEFAULT_WAVELENGTH,
        help="Path to band-number-to-wavelength CSV (for --freq-mode wavelength)",
    )

    # ── Phase 2 visual arguments ─────────────────────────────────────────
    p.add_argument("--visual-mode", choices=["dots", "circles"], default="dots",
                   help="Visual display mode")
    p.add_argument("--visual-scale", choices=["linear", "log10", "ln"], default="log10",
                   help="Intensity scaling mode for visual display (independent of --scale)")
    p.add_argument("--colormap", type=str, default="plasma",
                   help="Matplotlib colormap name for visual display")
    p.add_argument("--show-labels", action="store_true",
                   help="Show channel index/wavelength labels below each dot/circle")
    p.add_argument("--video-output", type=str, default=None,
                   help="Output video path (.mp4 or .avi) with synchronized audio")
    p.add_argument("--live-display", action="store_true",
                   help="Show live matplotlib animation during playback")
    p.add_argument("--video-title", type=str,
                   default="Sounds of Deep Ice Fluorescence",
                   help="Title string displayed in visual frames")
    p.add_argument("--frame-width", type=int, default=1280,
                   help="Visual frame width in pixels")
    p.add_argument("--frame-height", type=int, default=720,
                   help="Visual frame height in pixels")

    # ── Phase 3 arguments ────────────────────────────────────────────────
    p.add_argument("--trail-rows", type=int, default=5,
                   help="Number of trail rows visible simultaneously (1-20)")
    p.add_argument("--max-frames", type=int, default=500,
                   help="Maximum frames to render into video (safety cap)")
    p.add_argument("--tone-source",
                   choices=["band_index", "wavelength", "column"],
                   default="band_index",
                   help="What drives the pitch of each row")
    p.add_argument("--tone-column", type=str, default=None,
                   help="Column name for --tone-source column")
    p.add_argument("--intensity-source",
                   choices=["band_value", "column"],
                   default="band_value",
                   help="What drives the volume of each row")
    p.add_argument("--intensity-column", type=str, default=None,
                   help="Column name for --intensity-source column")

    # ── Phase 4 arguments ────────────────────────────────────────────────
    p.add_argument("--show-minimap", action="store_true",
                   help="Show overview minimap panel in video frames")
    p.add_argument("--output-name", type=str, default=None,
                   help="Base name for output files (produces outputs/NAME.wav + .mp4)")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── Output name resolution ────────────────────────────────────────────
    # --output-name sets defaults; explicit --output / --video-output override.
    if args.output_name:
        if args.output is None:
            args.output = os.path.join("outputs", f"{args.output_name}.wav")
        if args.video_output is None:
            args.video_output = os.path.join("outputs", f"{args.output_name}.mp4")

    # ── 1. Build and validate config ──────────────────────────────────────
    param_map = ParameterMap(
        tone_source=args.tone_source,
        tone_column=args.tone_column,
        intensity_source=args.intensity_source,
        intensity_column=args.intensity_column,
    )

    config = SonificationConfig(
        input_path=args.input,
        row_start=args.row_start,
        row_end=args.row_end,
        n_bins=args.n_bins,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        playback_speed=args.playback_speed,
        volume=args.volume,
        scale=args.scale,
        freq_mode=args.freq_mode,
        sample_rate=args.sample_rate,
        output=args.output,
        yes=args.yes,
        wavelength_path=args.wavelength_path,
        # Phase 2 visual parameters
        visual_mode=args.visual_mode,
        visual_scale=args.visual_scale,
        colormap=args.colormap,
        show_labels=args.show_labels,
        video_output=args.video_output,
        live_display=args.live_display,
        video_title=args.video_title,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        # Phase 3+4
        trail_rows=args.trail_rows,
        max_frames=args.max_frames,
        param_map=param_map,
        show_minimap=args.show_minimap,
        output_name=args.output_name,
    )
    config.validate()

    # ── 2. Load CSV ───────────────────────────────────────────────────────
    log("LOAD", f"Loading {config.input_path}")
    df = load_csv(config.input_path)
    log("LOAD", f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # ── 3. Load wavelength table (for display and frequency mapping) ──────
    wl_table = None
    try:
        wl_table = load_wavelength_table(config.wavelength_path)
    except Exception:
        pass  # wavelength table is optional

    # ── 4. Detect and confirm band columns ────────────────────────────────
    band_cols, band_indices = detect_band_columns(df)
    confirmed_cols = confirm_with_user(
        band_cols,
        band_indices,
        skip_confirm=config.yes,
        wavelength_table=wl_table,
        all_columns=list(df.columns),
        df=df,
    )
    if not confirmed_cols:
        log("DETECT", "Aborted by user.")
        sys.exit(1)

    band_cols = confirmed_cols
    log("DETECT", f"Found {len(band_cols)} spectral channels "
        f"({band_cols[0]} ... {band_cols[-1]})")

    # ── 5. Sort by row order ──────────────────────────────────────────────
    df = sort_by_row_order(df)
    log("SORT", "Sorted by row_num ascending" if "row_num" in df.columns
        else "Sorted by depth descending" if "depth" in df.columns
        else "Original order preserved")

    # ── 6. Slice rows ─────────────────────────────────────────────────────
    start = config.row_start or 0
    end = config.row_end or len(df)
    df = df.iloc[start:end].reset_index(drop=True)

    # ── 7. Clean (NaN→0, negative→0) ─────────────────────────────────────
    matrix = clean(df, band_cols)
    n_nan = int(np.isnan(df[band_cols].to_numpy()).sum())
    n_neg = int((df[band_cols].to_numpy() < 0).sum())
    log("CLEAN", f"Clipped {n_nan} NaN values, {n_neg} negative values to 0")
    n_channels_original = matrix.shape[1]

    # ── 8. Rebin if requested ─────────────────────────────────────────────
    n_bins = config.n_bins or n_channels_original
    did_rebin = n_bins < n_channels_original
    if did_rebin:
        matrix = rebin(matrix, n_bins)
        log("REBIN", f"{n_channels_original} channels -> {n_bins} bins")
    else:
        log("REBIN", f"{n_channels_original} channels -> {n_bins} bins (no rebinning)")

    n_channels = matrix.shape[1]

    # ── Save clean matrix BEFORE audio scaling (for visual path) ──────────
    want_visual = config.video_output or config.live_display
    if want_visual:
        clean_matrix = matrix.copy()

    # ── 9. Prepare wavelengths (if wavelength mode) ───────────────────────
    wavelengths_array = None
    if config.freq_mode == "wavelength" or config.param_map.tone_source == "wavelength":
        if wl_table is None:
            wl_table = load_wavelength_table(config.wavelength_path)
        # Build per-detected-band wavelength array using band indices
        wl_per_band = np.array(
            [wl_table.get(idx, 0.0) for idx in band_indices], dtype=np.float64
        )
        if did_rebin:
            wl_per_band = rebin_wavelengths(wl_per_band, n_bins)
        wavelengths_array = wl_per_band

    # ── 10. Scale values (audio) ──────────────────────────────────────────
    matrix = scale_values(matrix, config.scale)
    log("SCALE", f"Applying {config.scale} scale")

    # ── 11. Normalize per-channel ─────────────────────────────────────────
    matrix = normalize_per_channel(matrix)

    # ── 12. Assign frequencies ────────────────────────────────────────────
    pm = config.param_map
    if pm.tone_source == "column":
        # Column-driven tone: per-row frequencies
        if pm.tone_column not in df.columns:
            raise ValueError(
                f"tone_column '{pm.tone_column}' not found in dataset columns"
            )
        col_values = df[pm.tone_column].to_numpy(dtype=np.float64)
        row_freqs = map_tone_from_column(col_values, config.min_freq, config.max_freq)
        # Broadcast to (n_rows, n_channels) — all channels share the same freq per row
        freqs = np.tile(row_freqs[:, np.newaxis], (1, n_channels))
        log("SYNTH", f"Tone from column '{pm.tone_column}': "
            f"{row_freqs.min():.1f}-{row_freqs.max():.1f} Hz")
    else:
        # Standard per-channel frequency assignment
        freq_mode = "wavelength" if pm.tone_source == "wavelength" else config.freq_mode
        freqs = assign_frequencies(
            n_channels, config.min_freq, config.max_freq,
            mode=freq_mode, wavelengths=wavelengths_array,
        )
        log("SYNTH", f"Frequencies: {freqs[0]:.1f} Hz - {freqs[-1]:.1f} Hz "
            f"({n_channels} channels)")

    # ── 13. Apply intensity column modulation (if configured) ─────────────
    if pm.intensity_source == "column":
        if pm.intensity_column not in df.columns:
            raise ValueError(
                f"intensity_column '{pm.intensity_column}' not found in dataset columns"
            )
        intensity_values = df[pm.intensity_column].to_numpy(dtype=np.float64)
        matrix = apply_intensity_column(matrix, intensity_values)
        log("SYNTH", f"Intensity modulated by column '{pm.intensity_column}'")

    # ── 14. Synthesize ────────────────────────────────────────────────────
    seconds_per_row = 1.0 / config.playback_speed
    duration_s = len(matrix) * seconds_per_row
    log("SYNTH", f"Synthesizing {len(matrix)} rows at {config.playback_speed} "
        f"rows/sec -> {duration_s:.1f}s audio")
    waveform = synthesize(
        matrix, freqs, seconds_per_row, config.sample_rate, config.volume,
    )
    duration = len(waveform) / config.sample_rate

    # ── 15. Export / play / video ─────────────────────────────────────────
    temp_wav = False
    wav_path = None

    if config.video_output:
        # ── Step 15a: Export WAV to disk (needed for video muxing) ─────
        if config.output:
            wav_path = config.output
        else:
            # Create a temporary WAV for muxing, will clean up after
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            temp_wav = True
        export_wav(waveform, config.sample_rate, wav_path)
        log("EXPORT", f"Writing {wav_path}")

        # ── Step 15b: Render visual frames ────────────────────────────
        from sonify.visualize import apply_visual_scale, render_all_frames

        visual_matrix = apply_visual_scale(clean_matrix, config.visual_scale)
        depths = df["depth"].values if "depth" in df.columns else None

        render_start = time.time()
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
            trail_rows=config.trail_rows,
            max_frames=config.max_frames,
            show_minimap=config.show_minimap,
        )
        render_elapsed = time.time() - render_start
        log("RENDER", f"Rendered {len(frames)} frames -- {render_elapsed:.1f}s elapsed")

        # ── Step 15c: Mux audio + video ──────────────────────────────
        from sonify.video_export import export_video

        log("VIDEO", f"Muxing audio + video -> {config.video_output}")
        export_video(frames, wav_path, config.video_output,
                     fps=config.playback_speed)

        # Clean up temp WAV if we created one
        if temp_wav and os.path.isfile(wav_path):
            os.remove(wav_path)

        log("DONE", f"{config.video_output} ({len(frames)} frames, "
            f"{duration:.1f}s, {config.frame_width}x{config.frame_height})")

    elif config.live_display:
        # ── Live animated display (best-effort sync) ──────────────────
        from sonify.visualize import apply_visual_scale, live_display

        visual_matrix = apply_visual_scale(clean_matrix, config.visual_scale)
        depths = df["depth"].values if "depth" in df.columns else None

        # Also export WAV if requested
        if config.output:
            export_wav(waveform, config.sample_rate, config.output)
            log("EXPORT", f"Writing {config.output}")

        live_display(
            visual_matrix, waveform, config.sample_rate,
            config.playback_speed,
            mode=config.visual_mode,
            colormap=config.colormap,
            depths=depths,
            wavelengths=wavelengths_array,
            show_labels=config.show_labels,
            title=config.video_title,
            fig_width=config.frame_width,
            fig_height=config.frame_height,
            trail_rows=config.trail_rows,
        )

    elif config.output:
        # ── Audio-only WAV export (Phase 1 behavior) ──────────────────
        export_wav(waveform, config.sample_rate, config.output)
        log("EXPORT", f"Writing {config.output}")
        log("DONE", f"{config.output} ({duration:.1f}s)")

    else:
        # ── Audio-only speaker playback (Phase 1 behavior) ────────────
        play(waveform, config.sample_rate)


if __name__ == "__main__":
    main()
