"""
Sonification Toolkit — generic, dataset-agnostic engine.

Phase 1: additive synthesis from multi-channel tabular data.
"""

__version__ = "0.2.0"

from sonify.config import SonificationConfig
from sonify.data_io import load_csv
from sonify.band_detect import detect_band_columns, confirm_with_user
from sonify.preprocess import sort_by_row_order, clean, rebin, rebin_wavelengths
from sonify.mapping import (
    scale_values,
    normalize_per_channel,
    apply_global_gain,
    compute_gain_reference,
    assign_frequencies,
    assign_frequencies_pentatonic,
    load_wavelength_table,
    smooth_amplitude_matrix,
    PENTATONIC_RATIOS,
)
from sonify.synth import synthesize, generate_adsr_envelope, ADSR_SHAPES
from sonify.playback import play
from sonify.export import export_wav

__all__ = [
    "SonificationConfig",
    "load_csv",
    "detect_band_columns",
    "confirm_with_user",
    "sort_by_row_order",
    "clean",
    "rebin",
    "rebin_wavelengths",
    "scale_values",
    "normalize_per_channel",
    "apply_global_gain",
    "compute_gain_reference",
    "assign_frequencies",
    "assign_frequencies_pentatonic",
    "load_wavelength_table",
    "smooth_amplitude_matrix",
    "PENTATONIC_RATIOS",
    "synthesize",
    "generate_adsr_envelope",
    "ADSR_SHAPES",
    "play",
    "export_wav",
]
