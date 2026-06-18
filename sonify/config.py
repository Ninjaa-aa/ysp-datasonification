"""
SonificationConfig — all user-tunable parameters with validation.

Every CLI parameter from the Phase 1 spec lives here as a typed field.
The CLI runner (scripts/run_sonify.py) builds one of these from argparse,
calls validate(), then passes it down the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


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
