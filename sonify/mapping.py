"""
Value scaling, per-channel normalization, frequency assignment, and
amplitude smoothing.

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
# Pentatonic scale constants
# ---------------------------------------------------------------------------

# Major pentatonic intervals: root, maj2 (9/8), maj3 (5/4), perf5 (3/2),
# maj6 (5/3).  Any subset of these intervals is consonant by construction
# — no dissonant minor 2nds or tritones exist in this set.
PENTATONIC_RATIOS = (1.0, 1.125, 1.25, 1.5, 1.667)


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
# Global gain normalization (Dr. Malaska's auto-gain request, 2026-07-09)
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
        work = matrix.copy()
    else:
        work = matrix.copy()

    # Clip to zero so true zeros stay silent, instead of shifting up
    work = np.clip(work, 0.0, None)

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

def assign_frequencies_pentatonic(
    n_channels: int,
    root_hz: float = 220.0,
    n_octaves: int = 3,
) -> np.ndarray:
    """Map n_channels to notes of a major pentatonic scale.

    Builds the full note list across ``n_octaves`` (5 notes per octave),
    then selects ``n_channels`` notes distributed evenly across that list.
    If ``n_channels > 5 * n_octaves``, auto-extends octaves to accommodate
    (silently, no error).

    Parameters
    ----------
    n_channels : int
        Number of output frequencies needed.
    root_hz : float
        Root note in Hz (default 220.0 = A3, warm register).
    n_octaves : int
        Number of octaves to span (default 3 = 15 available notes).

    Returns
    -------
    np.ndarray
        Shape ``(n_channels,)``, dtype float64, strictly increasing,
        all values >= ``root_hz``.
    """
    # Auto-extend if needed
    while 5 * n_octaves < n_channels:
        n_octaves += 1

    # Build full note list across all octaves
    notes = []
    for octave in range(n_octaves):
        octave_root = root_hz * (2 ** octave)
        for ratio in PENTATONIC_RATIOS:
            notes.append(octave_root * ratio)
    notes = sorted(set(notes))  # deduplicate, sort ascending

    # Select n_channels evenly distributed notes
    if len(notes) <= n_channels:
        indices = list(range(len(notes)))
    else:
        indices = [round(i * (len(notes) - 1) / (n_channels - 1))
                   for i in range(n_channels)]

    return np.array([notes[i] for i in indices], dtype=np.float64)


def assign_frequencies(
    n_channels: int,
    min_freq: float,
    max_freq: float,
    mode: str = "index",
    wavelengths: np.ndarray | None = None,
    pentatonic_root: float = 220.0,
    pentatonic_octaves: int = 3,
) -> np.ndarray:
    """Assign a frequency (Hz) to each output channel.

    Parameters
    ----------
    n_channels : int
        Number of output channels (post-rebinning).
    min_freq, max_freq : float
        Frequency window in Hz.  **Ignored** in pentatonic mode.
    mode : str
        ``'index'`` — log-spaced by channel index.
        ``'wavelength'`` — placed by actual wavelength via linear interpolation
        on wavelength, log-placement in Hz.
        ``'pentatonic'`` — mapped to major pentatonic scale notes.
    wavelengths : np.ndarray or None
        1-D array of per-channel wavelength centers (nm).  Required (and must
        have length == ``n_channels``) when ``mode='wavelength'``.
    pentatonic_root : float
        Root note in Hz for pentatonic mode (default 220.0 = A3).
    pentatonic_octaves : int
        Number of octaves to span in pentatonic mode (default 3).

    Returns
    -------
    np.ndarray
        1-D array of length ``n_channels``, frequencies in Hz, strictly
        increasing.
    """
    if mode == "pentatonic":
        return assign_frequencies_pentatonic(
            n_channels, root_hz=pentatonic_root, n_octaves=pentatonic_octaves,
        )

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
        raise ValueError(
            f"Unknown freq_mode '{mode}'; expected index/wavelength/pentatonic"
        )


# ---------------------------------------------------------------------------
# Lambda-max mapping (Phase 3 default: "tone, lambda max, intensity volume")
# ---------------------------------------------------------------------------

def lambda_max_per_row(
    matrix: np.ndarray,
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find each row's dominant emission wavelength and its intensity.

    For fluorescence data the band carrying the most signal identifies *which
    fluorophore is emitting*, so it is the physically meaningful driver of
    pitch — different peak wavelengths become different, recognizable notes.
    In this borehole dataset the peak band genuinely varies between events
    (Band 8 at 313.7 nm, Band 13 at 341.2 nm, Band 21 at 385.3 nm, ...).

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` of band intensities.
    wavelengths : np.ndarray
        1-D array of per-channel wavelength centers in nm, length
        ``n_channels``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(peak_wavelength, peak_value)``, each of length ``n_rows``.
        Rows with no signal report their peak value as 0.0.

    Raises
    ------
    ValueError
        If ``wavelengths`` length does not match the channel count.
    """
    if len(wavelengths) != matrix.shape[1]:
        raise ValueError(
            f"wavelengths length ({len(wavelengths)}) must match "
            f"n_channels ({matrix.shape[1]})"
        )

    peak_idx = np.argmax(matrix, axis=1)
    peak_value = matrix[np.arange(matrix.shape[0]), peak_idx]
    peak_wavelength = np.asarray(wavelengths, dtype=np.float64)[peak_idx]
    return peak_wavelength, peak_value


def snap_to_scale(freqs: np.ndarray, scale_freqs: np.ndarray) -> np.ndarray:
    """Quantize each frequency to the nearest note of a musical scale.

    Quantizing trades a little precision for consonance: any combination of
    notes drawn from a pentatonic set is harmonious by construction, which is
    what turns a continuous wavelength sweep into something musical rather
    than microtonal.

    Parameters
    ----------
    freqs : np.ndarray
        1-D array of frequencies in Hz.
    scale_freqs : np.ndarray
        1-D array of allowed note frequencies in Hz.

    Returns
    -------
    np.ndarray
        1-D array of the same length, each value snapped to the nearest
        allowed note.  Snapping is done in log-frequency space so it matches
        how pitch distance is perceived.
    """
    if len(scale_freqs) == 0:
        return np.asarray(freqs, dtype=np.float64).copy()

    log_f = np.log(np.asarray(freqs, dtype=np.float64))
    log_scale = np.log(np.asarray(scale_freqs, dtype=np.float64))
    nearest = np.argmin(np.abs(log_f[:, np.newaxis] - log_scale[np.newaxis, :]), axis=1)
    return np.asarray(scale_freqs, dtype=np.float64)[nearest]


def assign_frequencies_lambda_max(
    matrix: np.ndarray,
    wavelengths: np.ndarray,
    min_freq: float,
    max_freq: float,
    scale_freqs: np.ndarray | None = None,
    wl_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each row's peak wavelength to a pitch, and its peak value to amplitude.

    This is the mapping Dr. Malaska specified as the Phase 3 default:
    *tone = lambda max, intensity = volume*.  It reduces each row to a single
    voice, which is what makes an event read as one identifiable note instead
    of a chord of every channel at once.

    Shorter wavelengths (deep UV) map to lower pitches and longer wavelengths
    to higher, so the audible sweep runs in the same direction as the spectrum.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` of band intensities.
    wavelengths : np.ndarray
        1-D per-channel wavelength centers in nm.
    min_freq, max_freq : float
        Output frequency window in Hz.
    scale_freqs : np.ndarray or None
        If given, snap pitches to these notes (e.g. a pentatonic set).
    wl_range : tuple[float, float] or None
        ``(wl_min, wl_max)`` to anchor the mapping.  Defaults to the span of
        ``wavelengths``.  Pass the instrument's full range so pitches stay
        comparable between a sparse slice and the whole dataset.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(freqs, amplitudes)``, each of length ``n_rows``.  ``freqs`` is in
        Hz; ``amplitudes`` holds the raw peak intensity per row, ready for the
        normal scaling/gain path.
    """
    peak_wl, peak_value = lambda_max_per_row(matrix, wavelengths)

    wl_min, wl_max = wl_range if wl_range is not None else (
        float(np.min(wavelengths)), float(np.max(wavelengths))
    )

    if wl_max == wl_min:
        freqs = np.full(len(peak_wl), np.sqrt(min_freq * max_freq))
    else:
        t = np.clip((peak_wl - wl_min) / (wl_max - wl_min), 0.0, 1.0)
        log_min, log_max = np.log(min_freq), np.log(max_freq)
        freqs = np.exp(log_min + t * (log_max - log_min))

    if scale_freqs is not None:
        freqs = snap_to_scale(freqs, scale_freqs)

    return freqs, peak_value


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
# Parameter mapping (project plan Phase 3)
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



# ---------------------------------------------------------------------------
# Temporal amplitude smoothing (Sound Quality update)
# ---------------------------------------------------------------------------

def smooth_amplitude_matrix(
    matrix: np.ndarray,
    smoothing: float,
    sample_axis: int = 0,
) -> np.ndarray:
    """Apply Gaussian temporal smoothing along the row axis per channel.

    Smoothing is applied AFTER global gain normalization and BEFORE synthesis.
    This suppresses row-to-row amplitude jumps while preserving gradual
    depth-dependent structure.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``, values in [0, 1].
    smoothing : float
        0.0 = no smoothing (identity), 1.0 = maximum (sigma=10 rows).
        Default 0.3 = sigma=3 rows (light smoothing).
    sample_axis : int
        Axis to smooth along (0 = rows/time axis).

    Returns
    -------
    np.ndarray
        Same shape as ``matrix``, values clipped to [0, 1].
    """
    if smoothing <= 0.0:
        return matrix.copy()
    from scipy.ndimage import gaussian_filter1d
    sigma = smoothing * 10.0
    smoothed = gaussian_filter1d(
        matrix.astype(np.float64), sigma=sigma, axis=sample_axis,
    )
    return np.clip(smoothed, 0.0, 1.0)
