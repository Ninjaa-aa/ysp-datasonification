#!/usr/bin/env python
"""
CLI entry point for the borehole sonification toolkit (Phase 1).

This script wires together the generic ``sonify`` engine with
dataset-specific defaults for the ice borehole fluorescence example.
The engine itself is dataset-agnostic.

Usage
-----
    python scripts/run_sonify.py --input data/raw/...csv --yes --output outputs/test.wav
    python scripts/run_sonify.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

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
        description="Sonify a multi-channel CSV dataset (Phase 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    p.add_argument("--scale", choices=["linear", "log10", "ln"], default="log10", help="Intensity scaling mode")
    p.add_argument("--freq-mode", choices=["index", "wavelength"], default="index", help="Frequency assignment mode")
    p.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    p.add_argument("--output", type=str, default=None, help="Output .wav path (omit to play through speakers)")
    p.add_argument("--yes", action="store_true", help="Skip interactive band-confirmation prompt")
    p.add_argument(
        "--wavelength-path", default=_DEFAULT_WAVELENGTH,
        help="Path to band-number → wavelength CSV (for --freq-mode wavelength)",
    )
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

    # ── 9. Scale values ───────────────────────────────────────────────────
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

    # ── 13/14. Export or play ─────────────────────────────────────────────
    if config.output:
        export_wav(waveform, config.sample_rate, config.output)
    else:
        play(waveform, config.sample_rate)


if __name__ == "__main__":
    main()
