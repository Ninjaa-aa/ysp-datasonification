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
