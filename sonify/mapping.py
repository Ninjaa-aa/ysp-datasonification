"""
Value scaling, per-channel normalization, and frequency assignment.

Precondition: input matrices have already been cleaned by
``preprocess.clean()`` and contain no NaN or negative values.  Do **not**
re-clip here — duplication would mask upstream bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Small epsilon added before log transforms to avoid log(0).
_EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Value scaling
# ---------------------------------------------------------------------------

def scale_values(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Apply the chosen intensity scaling.

    Precondition
    -------------
    Input must be non-negative (guaranteed by ``preprocess.clean()``).
    This function does **not** re-clip negatives.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``, values >= 0.
    mode : str
        One of ``'linear'``, ``'log10'``, ``'ln'``.

    Returns
    -------
    np.ndarray
        Scaled copy of the matrix (never mutates the input).
    """
    if mode == "linear":
        return matrix.copy()
    elif mode == "log10":
        return np.log10(matrix + _EPSILON)
    elif mode == "ln":
        return np.log(matrix + _EPSILON)
    else:
        raise ValueError(f"Unknown scale mode '{mode}'; expected linear/log10/ln")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_per_channel(matrix: np.ndarray) -> np.ndarray:
    """Per-channel min/max normalization to [0, 1].

    Each channel (column) is independently normalized using its own global
    min and max across all rows in the loaded slice.  This preserves the
    depth-structure *within* each band while ensuring every channel's full
    dynamic range is audible — weak bands are not drowned out by bright ones.

    If a channel is constant (max == min), its output is set to 0 everywhere.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``.

    Returns
    -------
    np.ndarray
        Normalized copy with all values in [0, 1].
    """
    result = np.empty_like(matrix)
    for ch in range(matrix.shape[1]):
        col = matrix[:, ch]
        col_min = col.min()
        col_max = col.max()
        if col_max == col_min:
            result[:, ch] = 0.0
        else:
            result[:, ch] = (col - col_min) / (col_max - col_min)
    return result


# ---------------------------------------------------------------------------
# Global gain normalization (Phase 5)
# ---------------------------------------------------------------------------

def compute_gain_reference(matrix: np.ndarray, gain_mode: str) -> tuple[float, float]:
    """Scan the entire band matrix and compute the gain reference point.

    Always prints all four statistics (global max, 90th percentile, median,
    mean) so the user can see the full dataset picture on every run.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` — the scaled values.
    gain_mode : str
        One of the 8 valid gain modes.

    Returns
    -------
    tuple[float, float]
        ``(reference_value, target_amplitude)`` where:
        - ``reference_value``: the statistical anchor
        - ``target_amplitude``: what amplitude that reference maps to
          (1.0 for max/pct90 modes, 0.5 for median/mean modes)
    """
    flat = matrix.ravel()
    global_max = float(np.max(flat))
    pct90 = float(np.percentile(flat, 90))
    median = float(np.median(flat))
    mean = float(np.mean(flat))

    n_rows, n_channels = matrix.shape
    print(f"[GAIN]    Scanning {n_rows} rows x {n_channels} channels...")
    print(f"[GAIN]    Global max: {global_max:.4g}   "
          f"90th pct: {pct90:.4g}   Median: {median:.4g}   Mean: {mean:.4g}")

    # Determine base mode (strip _linear / _log suffix)
    if gain_mode.startswith("max"):
        reference = global_max
        target_amp = 1.0
        ref_name = "global max"
    elif gain_mode.startswith("pct90"):
        reference = pct90
        target_amp = 1.0
        ref_name = "90th percentile"
    elif gain_mode.startswith("median"):
        reference = median
        target_amp = 0.5
        ref_name = "median"
    elif gain_mode.startswith("mean"):
        reference = mean
        target_amp = 0.5
        ref_name = "mean"
    else:
        raise ValueError(f"Unknown gain_mode '{gain_mode}'")

    print(f"[GAIN]    Mode: {gain_mode}  ->  reference = {reference:.4g} "
          f"(maps to amplitude {target_amp})")

    return reference, target_amp


def apply_global_gain(
    matrix: np.ndarray,
    gain_mode: str,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Apply global gain normalization to the full matrix.

    For LINEAR modes:
        - Compute reference value from gain_mode
        - If midpoint mode (median/mean): amplitude = (value / reference) * 0.5
          Values above reference get amplitudes above 0.5, clipped to 1.0
        - If max/pct90 mode: amplitude = value / reference, clipped to 1.0

    For LOG modes:
        - Apply log10 transform first (value + epsilon)
        - Then apply the same reference logic on the log-transformed values
        - Log transform compresses dynamic range so quiet structure is audible
          but the gain reference still anchors the loudest point

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``.
    gain_mode : str
        One of the 8 valid gain modes.
    epsilon : float
        Small value added before log transforms to avoid log(0).

    Returns
    -------
    np.ndarray
        Normalized array in [0, 1], same shape as input.
        All values are clipped to [0, 1].
    """
    is_log = gain_mode.endswith("_log")

    if is_log:
        # Apply log10 transform first, then compute reference on log values
        work = np.log10(np.maximum(matrix, 0) + epsilon)
    else:
        work = matrix.copy()

    # Shift so minimum is 0 (log values can be negative)
    work_min = work.min()
    work = work - work_min

    reference, target_amp = compute_gain_reference(work, gain_mode)

    if reference <= 0:
        # All values are the same (or zero) — return zeros
        return np.zeros_like(matrix)

    # Scale: reference maps to target_amp
    result = (work / reference) * target_amp

    # Clip to [0, 1]
    np.clip(result, 0.0, 1.0, out=result)
    return result


# ---------------------------------------------------------------------------
# Frequency assignment
# ---------------------------------------------------------------------------

def assign_frequencies(
    n_channels: int,
    min_freq: float,
    max_freq: float,
    mode: str = "index",
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Assign a frequency (Hz) to each output channel.

    Parameters
    ----------
    n_channels : int
        Number of output channels (post-rebinning).
    min_freq, max_freq : float
        Frequency window in Hz.
    mode : str
        ``'index'`` — log-spaced by channel index.
        ``'wavelength'`` — placed by actual wavelength via linear interpolation
        on wavelength, log-placement in Hz.
    wavelengths : np.ndarray or None
        1-D array of per-channel wavelength centers (nm).  Required (and must
        have length == ``n_channels``) when ``mode='wavelength'``.

    Returns
    -------
    np.ndarray
        1-D array of length ``n_channels``, frequencies in Hz, strictly
        increasing.
    """
    if n_channels == 1:
        return np.array([min_freq])

    if mode == "index":
        ratio = max_freq / min_freq
        indices = np.arange(n_channels)
        freqs = min_freq * ratio ** (indices / (n_channels - 1))
        return freqs

    elif mode == "wavelength":
        if wavelengths is None:
            raise ValueError(
                "wavelengths array is required for freq_mode='wavelength'"
            )
        if len(wavelengths) != n_channels:
            raise ValueError(
                f"wavelengths array length ({len(wavelengths)}) must match "
                f"n_channels ({n_channels})"
            )

        wl = wavelengths.astype(np.float64)
        wl_min, wl_max = wl.min(), wl.max()

        if wl_max == wl_min:
            # All wavelengths identical → fall back to index mode
            return assign_frequencies(n_channels, min_freq, max_freq, mode="index")

        # Linear interpolation on wavelength: map [wl_min, wl_max] → [0, 1]
        t = (wl - wl_min) / (wl_max - wl_min)

        # Log-placement in Hz: map [0, 1] → [min_freq, max_freq] in log-space
        log_min = np.log(min_freq)
        log_max = np.log(max_freq)
        freqs = np.exp(log_min + t * (log_max - log_min))

        return freqs

    else:
        raise ValueError(f"Unknown freq_mode '{mode}'; expected index/wavelength")


# ---------------------------------------------------------------------------
# Wavelength table loader
# ---------------------------------------------------------------------------

def load_wavelength_table(path: str) -> dict[int, float]:
    """Load a band-number → wavelength-center mapping from a CSV.

    Expected CSV format (Watson band wavelengths):
    ``WATSON Band No.,Wavelength center (nm)``

    Parameters
    ----------
    path : str
        Path to the wavelength reference CSV.

    Returns
    -------
    dict[int, float]
        Mapping of band number (1-based) → wavelength center in nm.
    """
    df = pd.read_csv(path)
    # Use first two columns regardless of exact header naming
    band_col = df.columns[0]
    wl_col = df.columns[1]
    return dict(zip(df[band_col].astype(int), df[wl_col].astype(float)))


# ---------------------------------------------------------------------------
# Parameter mapping (Phase 3)
# ---------------------------------------------------------------------------

def map_tone_from_column(
    column_values: np.ndarray,
    min_freq: float,
    max_freq: float,
) -> np.ndarray:
    """Map a per-row scalar column to per-row frequencies in [min_freq, max_freq].

    Normalizes the column's range to the frequency range (log-spaced).
    Returns array of shape ``(n_rows,)`` — one frequency per row.

    Parameters
    ----------
    column_values : np.ndarray
        1-D array of per-row scalar values.
    min_freq, max_freq : float
        Frequency window in Hz.

    Returns
    -------
    np.ndarray
        1-D array of per-row frequencies in Hz.
    """
    values = column_values.astype(np.float64)
    v_min, v_max = values.min(), values.max()

    if v_max == v_min:
        # Constant column → all frequencies at geometric mean
        return np.full(len(values), np.sqrt(min_freq * max_freq))

    # Normalize to [0, 1]
    t = (values - v_min) / (v_max - v_min)

    # Map to [min_freq, max_freq] in log-space
    log_min = np.log(min_freq)
    log_max = np.log(max_freq)
    freqs = np.exp(log_min + t * (log_max - log_min))

    return freqs


def apply_intensity_column(
    amplitude_matrix: np.ndarray,
    column_values: np.ndarray,
) -> np.ndarray:
    """Multiply each row's amplitudes by the corresponding column value.

    The column values are first normalized to [0, 1] so they act as a
    per-row global amplitude multiplier.

    Parameters
    ----------
    amplitude_matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``.
    column_values : np.ndarray
        1-D array ``(n_rows,)`` of scalar multipliers.

    Returns
    -------
    np.ndarray
        Modulated copy of the amplitude matrix.
    """
    values = column_values.astype(np.float64)
    v_min, v_max = values.min(), values.max()

    if v_max == v_min:
        # Constant column → no modulation (or silence if all zero)
        if v_max == 0:
            return np.zeros_like(amplitude_matrix)
        normalized = np.ones(len(values))
    else:
        normalized = (values - v_min) / (v_max - v_min)

    # Multiply each row by its scalar
    return amplitude_matrix * normalized[:, np.newaxis]

