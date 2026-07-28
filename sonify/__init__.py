"""
Sonification Toolkit — generic, dataset-agnostic engine.

Additive synthesis from multi-channel tabular data, with an event trigger that
separates *which* measurements sound from *how loud* they are.
"""

__version__ = "1.0.0"

from sonify.config import SonificationConfig
from sonify.data_io import load_csv
from sonify.band_detect import detect_band_columns, confirm_with_user
from sonify.preprocess import sort_by_row_order, clean, rebin, rebin_wavelengths
from sonify.events import (
    apply_trigger,
    row_trigger_mask,
    find_event_clusters,
    summarize_events,
    threshold_for_target_tones,
    count_signals,
)
from sonify.quality import articulation, onset_rate, mean_amplitude, describe
from sonify.mapping import (
    scale_values,
    normalize_per_channel,
    apply_global_gain,
    compute_gain_reference,
    assign_frequencies,
    assign_frequencies_pentatonic,
    assign_frequencies_lambda_max,
    lambda_max_per_row,
    snap_to_scale,
    load_wavelength_table,
    smooth_amplitude_matrix,
    limit_voices,
    apply_envelope_tail,
    PENTATONIC_RATIOS,
)
from sonify.synth import (
    synthesize, generate_adsr_envelope, ADSR_SHAPES,
)
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
    "assign_frequencies_lambda_max",
    "lambda_max_per_row",
    "snap_to_scale",
    "load_wavelength_table",
    "smooth_amplitude_matrix",
    "limit_voices",
    "apply_envelope_tail",
    "PENTATONIC_RATIOS",
    "synthesize",
    "generate_adsr_envelope",
    "ADSR_SHAPES",
    "apply_trigger",
    "row_trigger_mask",
    "find_event_clusters",
    "summarize_events",
    "threshold_for_target_tones",
    "count_signals",
    "articulation",
    "onset_rate",
    "mean_amplitude",
    "describe",
    "play",
    "export_wav",
]
