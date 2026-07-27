#!/usr/bin/env python
"""
CLI entry point for the borehole sonification toolkit (Phases 1–5 + Sound Quality).

This script wires together the generic ``sonify`` engine with
dataset-specific defaults for the ice borehole fluorescence example.
The engine itself is dataset-agnostic.

Usage
-----
    py scripts/run_sonify.py --input data/raw/...csv --yes --output outputs/test.wav
    py scripts/run_sonify.py --yes --video-output outputs/preview.mp4
    py scripts/run_sonify.py --preset chime --yes --output outputs/chime.wav
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
    apply_global_gain,
    assign_frequencies,
    assign_frequencies_pentatonic,
    assign_frequencies_lambda_max,
    load_wavelength_table,
    map_tone_from_column,
    apply_intensity_column,
    smooth_amplitude_matrix,
)
from sonify.events import (
    apply_trigger,
    row_trigger_mask,
    summarize_events,
    threshold_for_target_tones,
)
from sonify.synth import synthesize
from sonify.playback import play
from sonify.export import export_wav


# ── Structured logging helper ─────────────────────────────────────────────

def log(stage: str, message: str) -> None:
    """Print a pipeline stage message with consistent formatting."""
    print(f"[{stage:<8s}] {message}")


# ── Preset definitions ────────────────────────────────────────────────────

PRESETS = {
    "none": {},

    "chime": {
        # Target: wind chime aesthetic, the primary Dr. Malaska goal
        "n_bins":             8,
        "freq_mode":          "pentatonic",
        "pentatonic_root":    220.0,
        "pentatonic_octaves": 3,
        "timbre_partition":   True,
        "adsr_shape":         "tight",
        "gain_mode":          "max_log",
        "scale":              "log10",
        "smoothing":          0.0,
        "playback_speed":     5.0,
    },

    "ambient": {
        # Target: slow, meditative, full dataset playback
        "n_bins":             6,
        "freq_mode":          "pentatonic",
        "pentatonic_root":    110.0,
        "pentatonic_octaves": 4,
        "timbre":             "bell",
        "timbre_partition":   True,
        "adsr_shape":         "slow",
        "gain_mode":          "median_log",
        "scale":              "log10",
        "smoothing":          0.7,
    },

    "scientific": {
        # Target: data fidelity over aesthetics, all channels, no musical processing
        "n_bins":             None,  # no rebinning, use all detected channels
        "freq_mode":          "index",
        "timbre":             "sine",
        "timbre_partition":   False,
        "adsr_shape":         "natural",
        "gain_mode":          "max_linear",
        "scale":              "linear",
        "smoothing":          0.0,
    },

    "event": {
        # Target: Dr. Malaska's two-function design (2026-07-24) end to end.
        # A linear trigger picks which peaks sound; log intensity sets how loud;
        # each event is pitched by its dominant emission wavelength.
        # ~25 tones matches his "probably above 10" guidance.
        "n_bins":             None,  # keep all channels so lambda-max is exact
        "target_tones":       25,
        "trigger_type":       "linear",
        "tone_source":        "lambda_max",
        "freq_mode":          "pentatonic",
        "pentatonic_root":    220.0,
        "pentatonic_octaves": 3,
        "timbre":             "chime",
        "timbre_partition":   False,  # single voice; partitioning needs >= 3 channels
        "adsr_shape":         "natural",
        "gain_mode":          "max_log",
        "scale":              "log10",
        "smoothing":          0.0,
        # 25 rows/s puts the full descent at ~2.7 min with a median ~8.6 s
        # between events. At 10 rows/s the median gap is 21.6 s (max 53 s),
        # which reads as dead air rather than sparse wind chimes.
        "playback_speed":     25.0,
        "reverb_tail_ms":     1200.0,  # carries each strike into the gaps
    },
}


# Global defaults for sentinel-valued args that are not filled by preset
GLOBAL_DEFAULTS = {
    "freq_mode":          "index",
    "timbre":             "sine",
    "timbre_partition":   True,
    "adsr_shape":         "natural",
    "gain_mode":          "max_linear",
    "scale":              "log10",
    "smoothing":          0.3,
    "threshold":          0.0,
    "trigger_type":       "linear",
    "tone_source":        "band_index",
    "reverb_tail_ms":     0.0,
}


def apply_preset(args: argparse.Namespace, n_detected_channels: int) -> None:
    """Fill None values on args namespace from preset definition.

    Called AFTER band detection (so n_detected_channels is known) and BEFORE
    SonificationConfig is built. Only sets values where args attribute is None
    (i.e., user did not explicitly provide that argument).
    """
    preset_name = args.preset or "none"
    preset = PRESETS.get(preset_name, {})

    for key, value in preset.items():
        if getattr(args, key, None) is None:
            if key == "n_bins" and value is None:
                setattr(args, key, n_detected_channels)  # no rebinning
            else:
                setattr(args, key, value)

    if preset_name != "none":
        applied = {k: getattr(args, k) for k in preset}
        parts = ", ".join(f"{k}={v}" for k, v in applied.items())
        log("PRESET", f"{preset_name}: {parts}")


def apply_global_defaults(
    args: argparse.Namespace, n_detected_channels: int
) -> None:
    """Fill remaining None values with global defaults after preset application."""
    for key, value in GLOBAL_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    # n_bins: default to detected count if still None
    if args.n_bins is None:
        args.n_bins = n_detected_channels


# ── Interactive playback speed prompt (Dr. Malaska, 2026-07-09) ───────────

def prompt_playback_speed(n_rows: int, skip: bool) -> float:
    """Prompt the user to choose a playback speed interactively.

    Parameters
    ----------
    n_rows : int
        Number of rows that will be rendered (for duration calculation).
    skip : bool
        If True (--yes flag), return the default without prompting.

    Returns
    -------
    float
        Chosen rows per second.
    """
    default = 10.0
    if skip:
        return default

    options = [
        (1, 1.0),
        (2, 5.0),
        (3, 10.0),
        (4, 20.0),
        (5, 40.0),
    ]

    print("\nPlayback speed not set. How many rows per second?")
    print("  Suggested values:")
    for num, speed in options:
        total = n_rows / speed
        label = {1: "very slow, meditative", 2: "slow", 3: "default, recommended",
                 4: "fast", 5: "very fast"}[num]
        print(f"    [{num}]  {speed:>4.0f}  rows/sec  ->  {total:>6.1f}s total  ({label})")
    print("    [c]  Custom value")

    while True:
        choice = input("\nEnter choice [1-5/c]: ").strip().lower()
        if choice in ("1", "2", "3", "4", "5"):
            speed = dict(options)[int(choice)]
            log("SPEED", f"Playback speed set to {speed} rows/sec")
            return speed
        elif choice == "c":
            while True:
                try:
                    custom = float(input("Enter rows per second (e.g. 15): ").strip())
                    if custom > 0:
                        log("SPEED", f"Playback speed set to {custom} rows/sec")
                        return custom
                    print("Value must be positive.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print("Please enter 1-5 or 'c'.")


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
        description="Sonify a multi-channel CSV dataset (Phases 1-5 + Sound Quality).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Flags are grouped so --help reads as a menu rather than a flat list of 50.
    g_core   = p.add_argument_group("Core", "Input, row range, channels")
    g_audio  = p.add_argument_group("Audio", "Pitch, loudness, timbre, envelope")
    g_trig   = p.add_argument_group(
        "Trigger", "Which measurements sound at all (independent of how loud)")
    g_map    = p.add_argument_group("Mapping", "What data drives pitch and volume")
    g_visual = p.add_argument_group("Visual", "Display and video rendering")
    g_out    = p.add_argument_group("Output", "Where results are written")

    g_core.add_argument(
        "--input", default=_DEFAULT_INPUT,
        help="Path to the input CSV file",
    )
    g_core.add_argument("--row-start", type=int, default=None, help="First row index to include (0-based)")
    g_core.add_argument("--row-end", type=int, default=None, help="Last row index (exclusive)")
    # n_bins: default=None (sentinel for preset override detection)
    g_core.add_argument("--n-bins", type=int, default=None, help="Number of output frequency bins (channels); default = detected count")
    g_trig.add_argument("--threshold", type=float, default=None,
                   help="Trigger threshold: rows with no band above this stay silent, "
                        "applied before --scale (0.0 = off, every row sounds)")
    g_trig.add_argument("--trigger-type", choices=["linear", "log"], default=None,
                   help="Trigger comparison domain (Dr. Malaska recommends linear)")
    g_trig.add_argument("--target-tones", type=int, default=None,
                   help="Solve --threshold from the data to yield ~N sounding events "
                        "(Dr. Malaska: 'probably above 10')")
    g_audio.add_argument("--min-freq", type=float, default=150.0, help="Lowest tone frequency in Hz")
    g_audio.add_argument("--max-freq", type=float, default=2500.0, help="Highest tone frequency in Hz")
    g_audio.add_argument("--playback-speed", type=float, default=None,
                   help="Rows per second (if omitted, prompts interactively)")
    g_audio.add_argument("--volume", type=float, default=0.8, help="Master gain 0.0-1.0")
    # scale: default=None (sentinel for preset)
    g_audio.add_argument("--scale", choices=["linear", "log10", "ln"], default=None, help="Intensity scaling mode (audio)")
    # freq_mode: default=None (sentinel for preset)
    g_audio.add_argument("--freq-mode", choices=["index", "wavelength", "pentatonic"], default=None, help="Frequency assignment mode")
    g_audio.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    g_out.add_argument("--output", type=str, default=None, help="Output .wav path (omit to play through speakers)")
    g_core.add_argument("--yes", action="store_true", help="Skip interactive prompts (band confirmation, playback speed)")
    g_map.add_argument(
        "--wavelength-path", default=_DEFAULT_WAVELENGTH,
        help="Path to band-number-to-wavelength CSV (for --freq-mode wavelength)",
    )

    # ── Pentatonic arguments ─────────────────────────────────────────────
    g_audio.add_argument("--pentatonic-root", type=float, default=None,
                   help="Root note in Hz for pentatonic mode (default: 220.0 = A3)")
    g_audio.add_argument("--pentatonic-octaves", type=int, default=None,
                   help="Number of octaves to span in pentatonic mode (default: 3)")

    # ── Visual arguments ─────────────────────────────────────────────────
    g_visual.add_argument("--visual-mode", choices=["dots", "circles"], default="dots",
                   help="Visual display mode")
    g_visual.add_argument("--visual-scale", choices=["linear", "log10", "ln"], default="log10",
                   help="Intensity scaling mode for visual display (independent of --scale)")
    g_visual.add_argument("--colormap", type=str, default="plasma",
                   help="Matplotlib colormap name for visual display")
    g_visual.add_argument("--show-labels", action="store_true",
                   help="Show channel index/wavelength labels below each dot/circle")
    g_out.add_argument("--video-output", type=str, default=None,
                   help="Output video path (.mp4 or .avi) with synchronized audio")
    g_visual.add_argument("--live-display", action="store_true",
                   help="Show live matplotlib animation during playback")
    g_visual.add_argument("--video-title", type=str,
                   default="Sounds of Deep Ice Fluorescence",
                   help="Title string displayed in visual frames")
    g_visual.add_argument("--frame-width", type=int, default=1280,
                   help="Visual frame width in pixels")
    g_visual.add_argument("--frame-height", type=int, default=720,
                   help="Visual frame height in pixels")

    # ── Trail / parameter-mapping arguments ──────────────────────────────
    g_visual.add_argument("--trail-rows", type=int, default=5,
                   help="Number of trail rows visible simultaneously (1-20)")
    g_visual.add_argument("--max-frames", type=int, default=500,
                   help="Maximum frames to render into video (safety cap)")
    g_map.add_argument("--tone-source",
                   choices=["band_index", "wavelength", "lambda_max", "column"],
                   default=None,
                   help="What drives the pitch. 'lambda_max' pitches each row by its "
                        "dominant emission wavelength and collapses it to one voice")
    g_map.add_argument("--tone-column", type=str, default=None,
                   help="Column name for --tone-source column")
    g_map.add_argument("--intensity-source",
                   choices=["band_value", "column"],
                   default="band_value",
                   help="What drives the volume of each row")
    g_map.add_argument("--intensity-column", type=str, default=None,
                   help="Column name for --intensity-source column")

    # ── Minimap / output-naming arguments ────────────────────────────────
    g_visual.add_argument("--show-minimap", action="store_true",
                   help="Show overview minimap panel in video frames")
    g_out.add_argument("--output-name", type=str, default=None,
                   help="Base name for output files (produces outputs/NAME.wav + .mp4)")

    # ── Display arguments (Dr. Malaska, 2026-07-09) ──────────────────────
    g_visual.add_argument("--marker-size", type=int, default=120,
                   help="Marker size for scatter plots (matplotlib s= parameter)")
    g_visual.add_argument("--marker-shape", choices=["circle", "square"], default="square",
                   help="Marker shape for scatter plots")
    g_visual.add_argument("--show-colorbar", action="store_true", default=True,
                   help="Show intensity colorbar (default: on)")
    g_visual.add_argument("--no-colorbar", action="store_false", dest="show_colorbar",
                   help="Hide intensity colorbar")

    # ── Auto-gain arguments (Dr. Malaska, 2026-07-09) ────────────────────
    # gain_mode: default=None (sentinel for preset)
    g_audio.add_argument("--gain-mode",
                   choices=["max_linear", "max_log", "pct90_linear", "pct90_log",
                            "median_linear", "median_log", "mean_linear", "mean_log"],
                   default=None,
                   help="Global gain normalization mode")

    # ── Sound-character arguments ────────────────────────────────────────
    # timbre: default=None (sentinel for preset)
    g_audio.add_argument("--timbre", choices=["sine", "bell", "chime"], default=None,
                   help="Synthesis timbre (sine=pure, bell=harmonic partials, chime=inharmonic)")

    # ── Sound quality arguments ──────────────────────────────────────────
    g_core.add_argument("--preset", choices=["none", "chime", "ambient", "scientific", "event"],
                   default="none",
                   help="Preset configuration (overrides individual settings not explicitly provided)")
    # adsr_shape: default=None (sentinel for preset)
    g_audio.add_argument("--adsr-shape", choices=["tight", "natural", "slow"], default=None,
                   help="ADSR envelope shape (tight=plucked, natural=bell, slow=pad)")
    # smoothing: default=None (sentinel for preset)
    g_audio.add_argument("--smoothing", type=float, default=None,
                   help="Temporal amplitude smoothing 0.0-1.0 (0=off, 0.3=light, 0.7=heavy)")
    g_audio.add_argument("--reverb-tail-ms", type=float, default=None,
                   help="Decaying tail so notes ring out into silence (0=off, "
                        "600-1500 suits sparse event mode)")
    # timbre_partition: default=None (sentinel for preset)
    g_audio.add_argument("--timbre-partition", action="store_true", default=None, dest="timbre_partition",
                   help="Enable timbral partitioning (different timbres per spectral group)")
    g_audio.add_argument("--no-timbre-partition", action="store_false", dest="timbre_partition",
                   help="Disable timbral partitioning (all channels use same timbre)")

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

    # ── Handle playback speed sentinel ────────────────────────────────────
    # playback_speed is None when user didn't set it on CLI.
    # We will resolve it after preset application, before building config.
    playback_speed_from_cli = args.playback_speed  # None or float

    # Record which trigger settings came from the command line, before presets
    # fill in the blanks.  An explicit --threshold must beat a preset's
    # --target-tones, otherwise `--preset event --threshold 400` silently
    # renders the preset's tone count instead of the threshold asked for.
    threshold_from_cli = args.threshold is not None
    target_tones_from_cli = args.target_tones is not None

    # ── 1. Load CSV ───────────────────────────────────────────────────────
    input_path = args.input
    log("LOAD", f"Loading {input_path}")
    df = load_csv(input_path)
    log("LOAD", f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # ── 2. Load wavelength table (for display and frequency mapping) ──────
    wl_table = None
    try:
        wl_table = load_wavelength_table(args.wavelength_path)
    except Exception:
        pass  # wavelength table is optional

    # ── 3. Detect and confirm band columns ────────────────────────────────
    band_cols, band_indices = detect_band_columns(df)
    confirmed_cols = confirm_with_user(
        band_cols,
        band_indices,
        skip_confirm=args.yes,
        wavelength_table=wl_table,
        all_columns=list(df.columns),
        df=df,
    )
    if not confirmed_cols:
        log("DETECT", "Aborted by user.")
        sys.exit(1)

    band_cols = confirmed_cols
    n_detected = len(band_cols)
    log("DETECT", f"Found {n_detected} spectral channels "
        f"({band_cols[0]} ... {band_cols[-1]})")

    # ── Density warning (before preset application) ───────────────────────
    if (args.timbre in ("bell", "chime")
            and args.n_bins is None
            and not args.yes
            and n_detected > 16):
        n_partials = 4 if args.timbre in ("bell", "chime") else 1
        log("WARN", f'Timbre "{args.timbre}" with {n_detected} channels = '
            f'{n_detected * n_partials} simultaneous partials. Sounds like noise.')
        log("WARN", "Recommended: --n-bins 8 --freq-mode pentatonic, or use --preset chime")

    # ── 4. Apply preset and global defaults ───────────────────────────────
    apply_preset(args, n_detected)

    # An explicitly-given --threshold wins over a preset-supplied --target-tones.
    if threshold_from_cli and not target_tones_from_cli:
        args.target_tones = None
    apply_global_defaults(args, n_detected)

    # Resolve playback speed if still None after preset
    needs_playback_speed_prompt = (args.playback_speed is None)
    if args.playback_speed is None:
        args.playback_speed = 10.0  # temporary safe default for config validation

    # Fill pentatonic defaults if still None
    if args.pentatonic_root is None:
        args.pentatonic_root = 220.0
    if args.pentatonic_octaves is None:
        args.pentatonic_octaves = 3

    # ── 5. Build and validate config ──────────────────────────────────────
    param_map = ParameterMap(
        tone_source=args.tone_source,
        tone_column=args.tone_column,
        intensity_source=args.intensity_source,
        intensity_column=args.intensity_column,
    )

    # lambda_max needs the raw per-band spectrum to locate the peak, so
    # rebinning would blur exactly the information it depends on.
    if args.tone_source == "lambda_max" and args.n_bins != n_detected:
        log("FREQ", f"lambda_max tone: using all {n_detected} channels "
            f"(ignoring --n-bins {args.n_bins})")
        args.n_bins = n_detected

    config = SonificationConfig(
        input_path=args.input,
        row_start=args.row_start,
        row_end=args.row_end,
        n_bins=args.n_bins,
        threshold=args.threshold,
        trigger_type=args.trigger_type,
        target_tones=args.target_tones,
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
        pentatonic_root=args.pentatonic_root,
        pentatonic_octaves=args.pentatonic_octaves,
        # Visual parameters
        visual_mode=args.visual_mode,
        visual_scale=args.visual_scale,
        colormap=args.colormap,
        show_labels=args.show_labels,
        video_output=args.video_output,
        live_display=args.live_display,
        video_title=args.video_title,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        # Trail, minimap, parameter mapping
        trail_rows=args.trail_rows,
        max_frames=args.max_frames,
        param_map=param_map,
        show_minimap=args.show_minimap,
        output_name=args.output_name,
        # Display / gain / sound
        marker_size=args.marker_size,
        marker_shape=args.marker_shape,
        show_colorbar=args.show_colorbar,
        gain_mode=args.gain_mode,
        timbre=args.timbre,
        # Sound quality
        adsr_shape=args.adsr_shape,
        timbre_partition=args.timbre_partition,
        smoothing=args.smoothing,
        reverb_tail_ms=args.reverb_tail_ms,
    )
    config.validate()

    # ── 6. Sort by row order ──────────────────────────────────────────────
    df = sort_by_row_order(df)
    log("SORT", "Sorted by row_num ascending" if "row_num" in df.columns
        else "Sorted by depth descending" if "depth" in df.columns
        else "Original order preserved")

    # ── 7. Slice rows ─────────────────────────────────────────────────────
    start = config.row_start or 0
    end = config.row_end or len(df)
    df = df.iloc[start:end].reset_index(drop=True)

    # ── 7b. Interactive playback speed prompt ─────────────────────────────
    # Now that we know n_rows, resolve playback speed if neither CLI nor preset set it.
    # This MUST happen before synthesis but after row slicing.
    if needs_playback_speed_prompt:
        config.playback_speed = prompt_playback_speed(len(df), config.yes)

    # ── 8. Clean (NaN→0, negative→0) ─────────────────────────────────────
    matrix = clean(df, band_cols)
    n_nan = int(np.isnan(df[band_cols].to_numpy()).sum())
    n_neg = int((df[band_cols].to_numpy() < 0).sum())
    log("CLEAN", f"Clipped {n_nan} NaN values, {n_neg} negative values to 0")
    n_channels_original = matrix.shape[1]

    # ── 9. Rebin if requested ─────────────────────────────────────────────
    n_bins = config.n_bins or n_channels_original
    did_rebin = n_bins < n_channels_original
    if did_rebin:
        matrix = rebin(matrix, n_bins)
        log("REBIN", f"{n_channels_original} channels -> {n_bins} bins")
    else:
        log("REBIN", f"{n_channels_original} channels -> {n_bins} bins (no rebinning)")

    n_channels = matrix.shape[1]

    # ── 9b. Trigger gate — Dr. Malaska's "which peaks should have sound" ───
    # Function A of his two-function design (2026-07-24): a linear-domain
    # threshold decides *whether* a row sounds, entirely separately from
    # --scale, which decides *how loud* it is.  Applied before the audio and
    # visual matrices diverge so the video and WAV agree on what is silent.
    if config.target_tones is not None:
        solved = threshold_for_target_tones(matrix, config.target_tones)
        log("TRIGGER", f"Target {config.target_tones} tones -> "
            f"threshold {solved:.1f}")
        config.threshold = solved

    if config.threshold > 0:
        mask = row_trigger_mask(matrix, config.threshold, config.trigger_type)
        events = summarize_events(matrix, mask)
        matrix = apply_trigger(matrix, config.threshold, config.trigger_type)
        log("TRIGGER", f"{config.trigger_type} threshold {config.threshold:g}: "
            f"{int(mask.sum())} of {len(mask)} rows sound, "
            f"in {len(events)} event cluster(s)")
    else:
        log("TRIGGER", "Trigger gate: off (threshold=0.0, every row sounds)")

    # ── Save clean matrix BEFORE audio scaling (for visual path) ──────────
    want_visual = config.video_output or config.live_display
    if want_visual:
        clean_matrix = matrix.copy()

    # ── 10. Prepare wavelengths (if any wavelength-driven mode) ───────────
    wavelengths_array = None
    if (config.freq_mode == "wavelength"
            or config.param_map.tone_source in ("wavelength", "lambda_max")):
        if wl_table is None:
            wl_table = load_wavelength_table(config.wavelength_path)
        # Build per-detected-band wavelength array using band indices
        wl_per_band = np.array(
            [wl_table.get(idx, 0.0) for idx in band_indices], dtype=np.float64
        )
        if did_rebin:
            wl_per_band = rebin_wavelengths(wl_per_band, n_bins)
        wavelengths_array = wl_per_band

    # ── 10b. Lambda-max collapse (Phase 3 default: tone=lambda max) ───────
    # Reduce each row to one voice pitched by its dominant emission
    # wavelength, so an event reads as a single identifiable note rather than
    # a chord of every channel at once.  Done before scaling so the peak
    # intensity flows through the normal gain path as the note's loudness.
    lambda_max_freqs = None
    if config.param_map.tone_source == "lambda_max":
        scale_freqs = None
        if config.freq_mode == "pentatonic":
            scale_freqs = assign_frequencies_pentatonic(
                5 * config.pentatonic_octaves,
                root_hz=config.pentatonic_root,
                n_octaves=config.pentatonic_octaves,
            )
        lambda_max_freqs, peak_values = assign_frequencies_lambda_max(
            matrix, wavelengths_array, config.min_freq, config.max_freq,
            scale_freqs=scale_freqs,
        )
        # Collapse to a single voice. The visual path already saved the full
        # spectral matrix above, so the video still shows all channels.
        matrix = peak_values[:, np.newaxis]
        n_channels = 1
        n_distinct = len(np.unique(np.round(lambda_max_freqs, 1)))
        log("FREQ", f"Lambda-max tone: {n_distinct} distinct pitches, "
            f"{lambda_max_freqs.min():.1f}-{lambda_max_freqs.max():.1f} Hz"
            + (" (snapped to pentatonic)" if scale_freqs is not None else ""))

    # ── 11. Scale values (audio) ──────────────────────────────────────────
    matrix = scale_values(matrix, config.scale)
    log("SCALE", f"Applying {config.scale} scale")

    # ── 12. Global gain normalization ─────────────────────────────────────
    matrix = apply_global_gain(matrix, config.gain_mode)

    # ── 12b. Temporal amplitude smoothing ─────────────────────────────────
    if config.smoothing > 0:
        sigma = config.smoothing * 10.0
        log("SMOOTH", f"Temporal smoothing: sigma={sigma:.1f} rows (smoothing={config.smoothing})")
        matrix = smooth_amplitude_matrix(matrix, config.smoothing)
    else:
        log("SMOOTH", "Temporal smoothing: off (smoothing=0.0)")

    # ── 13. Assign frequencies ────────────────────────────────────────────
    pm = config.param_map
    if pm.tone_source == "lambda_max":
        # One voice per row, already pitched in step 10b.
        freqs = lambda_max_freqs[:, np.newaxis]
        log("SYNTH", f"Frequencies: {freqs.min():.1f} Hz - {freqs.max():.1f} Hz "
            f"(1 voice, lambda-max driven)")
    elif pm.tone_source == "column":
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
    elif config.freq_mode == "pentatonic":
        # Pentatonic mode: musically-constrained frequencies
        freqs = assign_frequencies(
            n_channels, config.min_freq, config.max_freq,
            mode="pentatonic",
            pentatonic_root=config.pentatonic_root,
            pentatonic_octaves=config.pentatonic_octaves,
        )
        log("FREQ", f"Pentatonic mode: {config.pentatonic_root:.1f} Hz root, "
            f"{config.pentatonic_octaves} octaves -> {n_channels} notes selected")
        log("FREQ", "min-freq / max-freq ignored in pentatonic mode")
        log("SYNTH", f"Frequencies: {freqs[0]:.1f} Hz - {freqs[-1]:.1f} Hz "
            f"({n_channels} channels, pentatonic)")
    else:
        # Standard per-channel frequency assignment
        freq_mode = "wavelength" if pm.tone_source == "wavelength" else config.freq_mode
        freqs = assign_frequencies(
            n_channels, config.min_freq, config.max_freq,
            mode=freq_mode, wavelengths=wavelengths_array,
        )
        log("SYNTH", f"Frequencies: {freqs[0]:.1f} Hz - {freqs[-1]:.1f} Hz "
            f"({n_channels} channels)")

    # ── 14. Apply intensity column modulation (if configured) ─────────────
    if pm.intensity_source == "column":
        if pm.intensity_column not in df.columns:
            raise ValueError(
                f"intensity_column '{pm.intensity_column}' not found in dataset columns"
            )
        intensity_values = df[pm.intensity_column].to_numpy(dtype=np.float64)
        matrix = apply_intensity_column(matrix, intensity_values)
        log("SYNTH", f"Intensity modulated by column '{pm.intensity_column}'")

    # ── 15. Synthesize ────────────────────────────────────────────────────
    seconds_per_row = 1.0 / config.playback_speed
    duration_s = len(matrix) * seconds_per_row
    partition_str = "partition" if config.timbre_partition else f"timbre={config.timbre}"
    log("SYNTH", f"Synthesizing {len(matrix)} rows at {config.playback_speed} "
        f"rows/sec -> {duration_s:.1f}s audio "
        f"(adsr={config.adsr_shape}, {partition_str})")
    if config.reverb_tail_ms > 0:
        log("SYNTH", f"Reverb tail: {config.reverb_tail_ms:.0f}ms decay")
    waveform = synthesize(
        matrix, freqs, seconds_per_row, config.sample_rate, config.volume,
        timbre=config.timbre,
        adsr_shape=config.adsr_shape,
        timbre_partition=config.timbre_partition,
        reverb_tail_ms=config.reverb_tail_ms,
    )
    duration = len(waveform) / config.sample_rate

    # ── 16. Export / play / video ─────────────────────────────────────────
    temp_wav = False
    wav_path = None

    if config.video_output:
        # ── Step 16a: Export WAV to disk (needed for video muxing) ─────
        if config.output:
            wav_path = config.output
        else:
            # Create a temporary WAV for muxing, will clean up after
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            temp_wav = True
        export_wav(waveform, config.sample_rate, wav_path)
        log("EXPORT", f"Writing {wav_path}")

        # ── Step 16b: Render visual frames ────────────────────────────
        from sonify.visualize import apply_visual_scale, render_all_frames

        visual_matrix = apply_visual_scale(
            clean_matrix, config.visual_scale, gain_mode=config.gain_mode
        )
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
            marker_size=config.marker_size,
            marker_shape=config.marker_shape,
            show_colorbar=config.show_colorbar,
        )
        render_elapsed = time.time() - render_start
        log("RENDER", f"Rendered {len(frames)} frames -- {render_elapsed:.1f}s elapsed")

        # ── Step 16c: Mux audio + video ──────────────────────────────
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

        visual_matrix = apply_visual_scale(
            clean_matrix, config.visual_scale, gain_mode=config.gain_mode
        )
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
            marker_size=config.marker_size,
            marker_shape=config.marker_shape,
            show_colorbar=config.show_colorbar,
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
