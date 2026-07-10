"""
SonificationConfig — all user-tunable parameters with validation.

Every CLI parameter from Phases 1–4 lives here as a typed field.  The CLI
runner (scripts/run_sonify.py) builds one of these from argparse, calls
validate(), then passes it down the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ParameterMap:
    """Controls which data column drives tone (pitch) and intensity (volume).

    Defaults reproduce the Phase 1/2 behavior: tone from band index,
    intensity from each band's own value.
    """
    tone_source: Literal["band_index", "wavelength", "column"] = "band_index"
    tone_column: str | None = None       # used when tone_source == "column"
    intensity_source: Literal["band_value", "column"] = "band_value"
    intensity_column: str | None = None  # used when intensity_source == "column"


@dataclass
class SonificationConfig:
    """Configuration for the sonification pipeline."""

    input_path: str = ""
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    n_bins: Optional[int] = None  # defaults to detected channel count at runtime
    min_freq: float = 150.0
    max_freq: float = 2500.0
    playback_speed: float = 10.0  # rows per second
    volume: float = 0.8  # master gain, 0.0–1.0
    scale: Literal["linear", "log10", "ln"] = "log10"
    freq_mode: Literal["index", "wavelength"] = "index"
    sample_rate: int = 44100
    output: Optional[str] = None  # path for .wav; None = play through speakers
    yes: bool = False  # skip interactive band-confirmation prompt
    wavelength_path: Optional[str] = None  # path to wavelength reference CSV

    # ── Phase 2: visual display parameters ────────────────────────────────
    visual_mode: Literal["dots", "circles"] = "dots"
    visual_scale: Literal["linear", "log10", "ln"] = "log10"
    colormap: str = "plasma"
    show_labels: bool = False
    video_output: Optional[str] = None  # path for .mp4/.avi output
    live_display: bool = False
    video_title: str = "Sounds of Deep Ice Fluorescence"
    frame_width: int = 1280
    frame_height: int = 720

    # ── Phase 3: trail display, channels, parameter mapping ───────────────
    trail_rows: int = 5  # number of rows visible simultaneously in trail
    max_frames: int = 500  # safety cap for render_all_frames()
    param_map: ParameterMap = field(default_factory=ParameterMap)

    # ── Phase 4: minimap, output naming ───────────────────────────────────
    show_minimap: bool = False
    output_name: Optional[str] = None  # base name for output files

    # ── Phase 5: display improvements ─────────────────────────────────────
    marker_size: int = 120  # matplotlib scatter s= parameter
    marker_shape: Literal["circle", "square"] = "square"  # Dr. Malaska requested square
    show_colorbar: bool = True  # intensity scale colorbar

    # ── Phase 5: auto-gain ────────────────────────────────────────────────
    gain_mode: Literal[
        "max_linear",       # global max = amplitude 1.0 (LINEAR response) [DEFAULT]
        "max_log",          # global max = amplitude 1.0 (LOG response)
        "pct90_linear",     # 90th percentile = amplitude 1.0, clip above (linear)
        "pct90_log",        # 90th percentile = amplitude 1.0, clip above (log)
        "median_linear",    # median value = amplitude 0.5 (linear)
        "median_log",       # median value = amplitude 0.5 (log)
        "mean_linear",      # mean value = amplitude 0.5 (linear)
        "mean_log",         # mean value = amplitude 0.5 (log)
    ] = "max_linear"

    # ── Phase 5: sound enhancements ───────────────────────────────────────
    sustain: float = 0.3  # amplitude sustain blend (0.0 = no sustain, 1.0 = max)
    timbre: Literal["sine", "bell", "chime"] = "chime"  # Dr. Malaska: wind chimes

    def validate(self) -> None:
        """Validate all parameters, raising ValueError on bad input."""
        if not self.input_path:
            raise ValueError("input_path must be specified")

        if self.min_freq <= 0:
            raise ValueError(f"min_freq must be positive, got {self.min_freq}")
        if self.max_freq <= 0:
            raise ValueError(f"max_freq must be positive, got {self.max_freq}")
        if self.max_freq <= self.min_freq:
            raise ValueError(
                f"max_freq ({self.max_freq}) must be greater than "
                f"min_freq ({self.min_freq})"
            )

        if self.playback_speed <= 0:
            raise ValueError(
                f"playback_speed must be positive, got {self.playback_speed}"
            )

        if not (0.0 <= self.volume <= 1.0):
            raise ValueError(f"volume must be in [0.0, 1.0], got {self.volume}")

        if self.scale not in ("linear", "log10", "ln"):
            raise ValueError(
                f"scale must be 'linear', 'log10', or 'ln', got '{self.scale}'"
            )

        if self.freq_mode not in ("index", "wavelength"):
            raise ValueError(
                f"freq_mode must be 'index' or 'wavelength', got '{self.freq_mode}'"
            )

        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

        if self.n_bins is not None and self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")

        if self.row_start is not None and self.row_start < 0:
            raise ValueError(f"row_start must be >= 0, got {self.row_start}")

        if (
            self.row_start is not None
            and self.row_end is not None
            and self.row_end <= self.row_start
        ):
            raise ValueError(
                f"row_end ({self.row_end}) must be greater than "
                f"row_start ({self.row_start})"
            )

        # ── Phase 2 visual validation ─────────────────────────────────────
        if self.visual_mode not in ("dots", "circles"):
            raise ValueError(
                f"visual_mode must be 'dots' or 'circles', got '{self.visual_mode}'"
            )

        if self.visual_scale not in ("linear", "log10", "ln"):
            raise ValueError(
                f"visual_scale must be 'linear', 'log10', or 'ln', "
                f"got '{self.visual_scale}'"
            )

        if self.frame_width <= 0:
            raise ValueError(f"frame_width must be positive, got {self.frame_width}")
        if self.frame_height <= 0:
            raise ValueError(f"frame_height must be positive, got {self.frame_height}")

        if self.video_output is not None:
            if not self.video_output.lower().endswith((".mp4", ".avi")):
                raise ValueError(
                    f"video_output must end in .mp4 or .avi, "
                    f"got '{self.video_output}'"
                )
            out_dir = os.path.dirname(self.video_output) or "."
            if not os.path.isdir(out_dir):
                raise ValueError(
                    f"video_output directory does not exist: '{out_dir}'"
                )

        # ── Phase 3 validation ────────────────────────────────────────────
        if not (1 <= self.trail_rows <= 20):
            raise ValueError(
                f"trail_rows must be in [1, 20], got {self.trail_rows}"
            )

        if self.max_frames < 1:
            raise ValueError(
                f"max_frames must be >= 1, got {self.max_frames}"
            )

        # Parameter mapping validation
        pm = self.param_map
        if pm.tone_source not in ("band_index", "wavelength", "column"):
            raise ValueError(
                f"tone_source must be 'band_index', 'wavelength', or 'column', "
                f"got '{pm.tone_source}'"
            )
        if pm.tone_source == "column" and not pm.tone_column:
            raise ValueError(
                "tone_column must be specified when tone_source is 'column'"
            )

        if pm.intensity_source not in ("band_value", "column"):
            raise ValueError(
                f"intensity_source must be 'band_value' or 'column', "
                f"got '{pm.intensity_source}'"
            )
        if pm.intensity_source == "column" and not pm.intensity_column:
            raise ValueError(
                "intensity_column must be specified when "
                "intensity_source is 'column'"
            )

        # ── Phase 5 validation ─────────────────────────────────────────
        if self.marker_size < 10:
            raise ValueError(
                f"marker_size must be >= 10, got {self.marker_size}"
            )

        if self.marker_shape not in ("circle", "square"):
            raise ValueError(
                f"marker_shape must be 'circle' or 'square', "
                f"got '{self.marker_shape}'"
            )

        _VALID_GAIN_MODES = (
            "max_linear", "max_log", "pct90_linear", "pct90_log",
            "median_linear", "median_log", "mean_linear", "mean_log",
        )
        if self.gain_mode not in _VALID_GAIN_MODES:
            raise ValueError(
                f"gain_mode must be one of {_VALID_GAIN_MODES}, "
                f"got '{self.gain_mode}'"
            )

        if not (0.0 <= self.sustain <= 1.0):
            raise ValueError(
                f"sustain must be in [0.0, 1.0], got {self.sustain}"
            )

        if self.timbre not in ("sine", "bell", "chime"):
            raise ValueError(
                f"timbre must be 'sine', 'bell', or 'chime', "
                f"got '{self.timbre}'"
            )
