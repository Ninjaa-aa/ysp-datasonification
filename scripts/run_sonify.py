#!/usr/bin/env python
"""
CLI entry point for the borehole sonification toolkit (Phase 1 + Phase 2).

This script wires together the generic ``sonify`` engine with
dataset-specific defaults for the ice borehole fluorescence example.
The engine itself is dataset-agnostic.

Phase 2 additions: visual display modes (dots/circles), video export
with synchronized audio, and optional live animated display.

Usage
-----
    python scripts/run_sonify.py --input data/raw/...csv --yes --output outputs/test.wav
    python scripts/run_sonify.py --yes --video-output outputs/preview.mp4
    python scripts/run_sonify.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# Ensure the project root is on sys.path so ``sonify`` package imports work
# when running ``python scripts/run_sonify.py`` from the project root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from sonify.config import SonificationConfig
from sonify.data_io import load_csv
from sonify.band_detect import detect_band_columns, confirm_with_user
from sonify.preprocess import sort_by_row_order, clean, rebin, rebin_wavelengths
from sonify.mapping import (
    scale_values,
    normalize_per_channel,
    assign_frequencies,
    load_wavelength_table,
)
from sonify.synth import synthesize
from sonify.playback import play
from sonify.export import export_wav


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
        description="Sonify a multi-channel CSV dataset (Phase 1 + Phase 2).",
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
    p.add_argument("--volume", type=float, default=0.8, help="Master gain 0.0–1.0")
    p.add_argument("--scale", choices=["linear", "log10", "ln"], default="log10", help="Intensity scaling mode (audio)")
    p.add_argument("--freq-mode", choices=["index", "wavelength"], default="index", help="Frequency assignment mode")
    p.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    p.add_argument("--output", type=str, default=None, help="Output .wav path (omit to play through speakers)")
    p.add_argument("--yes", action="store_true", help="Skip interactive band-confirmation prompt")
    p.add_argument(
        "--wavelength-path", default=_DEFAULT_WAVELENGTH,
        help="Path to band-number → wavelength CSV (for --freq-mode wavelength)",
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

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── 1. Build and validate config ──────────────────────────────────────
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
    )
    config.validate()
    print(f"Config validated: {config.scale} scale, {config.playback_speed} rows/s, "
          f"{config.min_freq}-{config.max_freq} Hz")

    # ── 2. Load CSV ───────────────────────────────────────────────────────
    print(f"Loading: {config.input_path}")
    df = load_csv(config.input_path)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # ── 3. Detect and confirm band columns ────────────────────────────────
    band_cols, band_indices = detect_band_columns(df)
    if not confirm_with_user(band_cols, skip_confirm=config.yes):
        print("Aborted by user.")
        sys.exit(1)

    # ── 4. Sort by row order ──────────────────────────────────────────────
    df = sort_by_row_order(df)

    # ── 5. Slice rows ─────────────────────────────────────────────────────
    start = config.row_start or 0
    end = config.row_end or len(df)
    df = df.iloc[start:end].reset_index(drop=True)
    print(f"Using rows {start}-{end} ({len(df)} rows)")

    # ── 6. Clean (NaN→0, negative→0) ─────────────────────────────────────
    matrix = clean(df, band_cols)
    n_channels_original = matrix.shape[1]

    # ── 7. Rebin if requested ─────────────────────────────────────────────
    n_bins = config.n_bins or n_channels_original
    did_rebin = n_bins < n_channels_original
    if did_rebin:
        matrix = rebin(matrix, n_bins)
        print(f"Rebinned {n_channels_original} -> {n_bins} channels")

    n_channels = matrix.shape[1]

    # ── Phase 2: Save clean matrix BEFORE audio scaling ───────────────
    # Visual path applies visual_scale independently on this copy.
    want_visual = config.video_output or config.live_display
    if want_visual:
        clean_matrix = matrix.copy()

    # ── 8. Prepare wavelengths (if wavelength mode) ───────────────────────
    wavelengths_array = None
    if config.freq_mode == "wavelength":
        wl_table = load_wavelength_table(config.wavelength_path)
        # Build per-detected-band wavelength array using band indices
        wl_per_band = np.array(
            [wl_table.get(idx, 0.0) for idx in band_indices], dtype=np.float64
        )
        if did_rebin:
            wl_per_band = rebin_wavelengths(wl_per_band, n_bins)
        wavelengths_array = wl_per_band
        print(f"Wavelength mode: {wavelengths_array[0]:.1f}-{wavelengths_array[-1]:.1f} nm")

    # ── 9. Scale values (audio) ───────────────────────────────────────────
    matrix = scale_values(matrix, config.scale)

    # ── 10. Normalize per-channel ─────────────────────────────────────────
    matrix = normalize_per_channel(matrix)

    # ── 11. Assign frequencies ────────────────────────────────────────────
    freqs = assign_frequencies(
        n_channels, config.min_freq, config.max_freq,
        mode=config.freq_mode, wavelengths=wavelengths_array,
    )
    print(f"Frequencies: {freqs[0]:.1f} Hz - {freqs[-1]:.1f} Hz ({n_channels} channels)")

    # ── 12. Synthesize ────────────────────────────────────────────────────
    seconds_per_row = 1.0 / config.playback_speed
    print(f"Synthesizing {len(matrix)} rows x {seconds_per_row*1000:.0f}ms/row ...")
    waveform = synthesize(
        matrix, freqs, seconds_per_row, config.sample_rate, config.volume,
    )
    duration = len(waveform) / config.sample_rate
    print(f"Waveform: {len(waveform)} samples ({duration:.1f}s)")

    # ── 13. Export / play / video ──────────────────────────────────────────
    #
    # Pipeline order for video: WAV export → render frames → mux video.
    # Both the WAV file and the frames must exist before muxing.
    #
    temp_wav = False
    wav_path = None

    if config.video_output:
        # ── Step 13a: Export WAV to disk (needed for video muxing) ─────
        if config.output:
            wav_path = config.output
        else:
            # Create a temporary WAV for muxing, will clean up after
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            temp_wav = True
        export_wav(waveform, config.sample_rate, wav_path)

        # ── Step 13b: Render visual frames ────────────────────────────
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

        # ── Step 13c: Mux audio + video ──────────────────────────────
        from sonify.video_export import export_video

        export_video(frames, wav_path, config.video_output,
                     fps=config.playback_speed)

        # Clean up temp WAV if we created one
        if temp_wav and os.path.isfile(wav_path):
            os.remove(wav_path)
            print(f"Cleaned up temporary WAV: {wav_path}")

    elif config.live_display:
        # ── Live animated display (best-effort sync) ──────────────────
        from sonify.visualize import apply_visual_scale, live_display

        visual_matrix = apply_visual_scale(clean_matrix, config.visual_scale)
        depths = df["depth"].values if "depth" in df.columns else None

        # Also export WAV if requested
        if config.output:
            export_wav(waveform, config.sample_rate, config.output)

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
        )

    elif config.output:
        # ── Audio-only WAV export (Phase 1 behavior) ──────────────────
        export_wav(waveform, config.sample_rate, config.output)

    else:
        # ── Audio-only speaker playback (Phase 1 behavior) ────────────
        play(waveform, config.sample_rate)


if __name__ == "__main__":
    main()

