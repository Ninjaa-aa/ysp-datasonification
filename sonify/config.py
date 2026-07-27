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
    tone_source: Literal["band_index", "wavelength", "lambda_max", "column"] = "band_index"
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
    # ── Trigger (Dr. Malaska's "which peaks should have sound") ───────────
    # Function A of his two-function design: decides *whether* a row sounds,
    # independently of `scale`/`gain_mode`, which decide *how loud* it is.
    threshold: float = 0.0  # 0.0 = off, every row with any signal sounds
    trigger_type: Literal["linear", "log"] = "linear"
    target_tones: Optional[int] = None  # if set, solves `threshold` from the data
    min_freq: float = 150.0
    max_freq: float = 2500.0
    playback_speed: float = 10.0  # rows per second
    volume: float = 0.8  # master gain, 0.0–1.0
    scale: Literal["linear", "log10", "ln"] = "log10"
    freq_mode: Literal["index", "wavelength", "pentatonic"] = "index"
    sample_rate: int = 44100
    pentatonic_root: float = 220.0   # root frequency Hz; default A3
    pentatonic_octaves: int = 3      # octaves to span
    output: Optional[str] = None  # path for .wav; None = play through speakers
    yes: bool = False  # skip interactive band-confirmation prompt
    wavelength_path: Optional[str] = None  # path to wavelength reference CSV

    # ── Visual display (project plan Phase 2) ─────────────────────────────
    visual_mode: Literal["dots", "circles"] = "dots"
    visual_scale: Literal["linear", "log10", "ln"] = "log10"
    colormap: str = "plasma"
    show_labels: bool = False
    video_output: Optional[str] = None  # path for .mp4/.avi output
    live_display: bool = False
    video_title: str = "Sounds of Deep Ice Fluorescence"
    frame_width: int = 1280
    frame_height: int = 720

    # ── Trail display, channel scaling, parameter mapping (plan Phase 3) ──
    trail_rows: int = 5  # number of rows visible simultaneously in trail
    max_frames: int = 500  # safety cap for render_all_frames()
    param_map: ParameterMap = field(default_factory=ParameterMap)

    # ── Minimap, output naming (project plan Phase 4) ─────────────────────
    show_minimap: bool = False
    output_name: Optional[str] = None  # base name for output files

    # ── Display (Dr. Malaska feedback, 2026-07-09) ────────────────────────
    marker_size: int = 120  # matplotlib scatter s= parameter
    marker_shape: Literal["circle", "square"] = "square"  # Dr. Malaska requested square
    show_colorbar: bool = True  # intensity scale colorbar

    # ── Auto-gain (Dr. Malaska feedback, 2026-07-09) ──────────────────────
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

    # ── Sound character (Dr. Malaska: "tinkling wind chimes") ─────────────
    timbre: Literal["sine", "bell", "chime"] = "chime"  # Dr. Malaska: wind chimes

    # ── Sound quality update ──────────────────────────────────────────────
    adsr_shape: Literal["tight", "natural", "slow"] = "natural"
    timbre_partition: bool = True    # split channels into spectral groups
    smoothing: float = 0.3           # temporal amplitude smoothing (0=off, 1=max)
    # Decaying tail so notes ring out into silence (Dr. Malaska's "sustain"
    # request). Decays to zero by design — a held sustain would merge notes
    # into a drone.
    reverb_tail_ms: float = 0.0

    def validate(self) -> None:
        """Validate all parameters, raising ValueError on bad input."""
        if not self.input_path:
            raise ValueError("input_path must be specified")

        if self.threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {self.threshold}")

        if self.trigger_type not in ("linear", "log"):
            raise ValueError(
                f"trigger_type must be 'linear' or 'log', got '{self.trigger_type}'"
            )

        if self.target_tones is not None and self.target_tones < 1:
            raise ValueError(
                f"target_tones must be >= 1, got {self.target_tones}"
            )

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

        if self.freq_mode not in ("index", "wavelength", "pentatonic"):
            raise ValueError(
                f"freq_mode must be 'index', 'wavelength', or 'pentatonic', "
                f"got '{self.freq_mode}'"
            )

        if self.pentatonic_root <= 0:
            raise ValueError(
                f"pentatonic_root must be positive, got {self.pentatonic_root}"
            )

        if not (1 <= self.pentatonic_octaves <= 8):
            raise ValueError(
                f"pentatonic_octaves must be in [1, 8], got {self.pentatonic_octaves}"
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

        # ── Visual validation ─────────────────────────────────────────────
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

        # ── Trail / parameter-mapping validation ──────────────────────────
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
        _VALID_TONE_SOURCES = ("band_index", "wavelength", "lambda_max", "column")
        if pm.tone_source not in _VALID_TONE_SOURCES:
            raise ValueError(
                f"tone_source must be one of {_VALID_TONE_SOURCES}, "
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

        # ── Display / gain / sound validation ──────────────────────────
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

        if self.timbre not in ("sine", "bell", "chime"):
            raise ValueError(
                f"timbre must be 'sine', 'bell', or 'chime', "
                f"got '{self.timbre}'"
            )

        _VALID_ADSR = ("tight", "natural", "slow")
        if self.adsr_shape not in _VALID_ADSR:
            raise ValueError(
                f"adsr_shape must be one of {_VALID_ADSR}, "
                f"got '{self.adsr_shape}'"
            )

        if not (0.0 <= self.smoothing <= 1.0):
            raise ValueError(
                f"smoothing must be in [0.0, 1.0], got {self.smoothing}"
            )

        if self.reverb_tail_ms < 0:
            raise ValueError(
                f"reverb_tail_ms must be >= 0, got {self.reverb_tail_ms}"
            )
