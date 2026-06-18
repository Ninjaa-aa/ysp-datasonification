"""
Preprocessing: cleaning, sorting, and rebinning.

This module is the single canonical place for negative-clipping and NaN
handling. Downstream modules (mapping.scale_values) assume the data is
already non-negative after passing through clean().
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helper for contiguous grouping — used by both rebin() and
# rebin_wavelengths() so they always collapse in lockstep.
# ---------------------------------------------------------------------------

def _contiguous_groups(n_items: int, n_bins: int) -> list[np.ndarray]:
    """Split range(n_items) into n_bins contiguous groups, as equal as possible.

    Returns a list of 1-D numpy arrays of integer indices.
    """
    return [g for g in np.array_split(np.arange(n_items), n_bins)]


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_by_row_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure rows are in monotonic measurement order.

    Sorts by ``row_num`` ascending if available, else by ``depth`` descending
    (instrument descending into the borehole), else returns the original order.
    """
    if "row_num" in df.columns:
        return df.sort_values("row_num", ascending=True).reset_index(drop=True)
    if "depth" in df.columns:
        return df.sort_values("depth", ascending=False).reset_index(drop=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame, band_cols: list[str]) -> np.ndarray:
    """Extract band columns, fill NaN with 0, clip negatives to 0.

    This is the **single canonical place** negative-clipping happens in the
    pipeline.  ``mapping.scale_values()`` assumes its input is already
    non-negative.

    Parameters
    ----------
    df : pd.DataFrame
        The (possibly sliced) dataset.
    band_cols : list[str]
        Column names for the detected band/channel columns.

    Returns
    -------
    np.ndarray
        2-D float64 array of shape ``(n_rows, n_channels)`` with all values
        >= 0.
    """
    matrix = df[band_cols].to_numpy(dtype=np.float64).copy()
    np.nan_to_num(matrix, nan=0.0, copy=False)
    np.clip(matrix, 0.0, None, out=matrix)
    return matrix


# ---------------------------------------------------------------------------
# Rebinning
# ---------------------------------------------------------------------------

def rebin(matrix: np.ndarray, n_bins: int) -> np.ndarray:
    """Reduce channel count by averaging contiguous groups.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array of shape ``(n_rows, n_channels)``.
    n_bins : int
        Desired number of output channels.  If ``>= n_channels``, the matrix
        is returned unchanged.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_rows, n_bins)``.
    """
    n_channels = matrix.shape[1]
    if n_bins >= n_channels:
        return matrix

    groups = _contiguous_groups(n_channels, n_bins)
    rebinned = np.column_stack(
        [matrix[:, g].mean(axis=1) for g in groups]
    )
    return rebinned


def rebin_wavelengths(wavelengths: np.ndarray, n_bins: int) -> np.ndarray:
    """Rebin a 1-D wavelength array using the same grouping as rebin().

    Parameters
    ----------
    wavelengths : np.ndarray
        1-D array of per-band wavelength centers, length ``n_channels``.
    n_bins : int
        Target number of output bins.

    Returns
    -------
    np.ndarray
        1-D array of length ``n_bins``, each value the mean wavelength of
        its group.
    """
    n_channels = len(wavelengths)
    if n_bins >= n_channels:
        return wavelengths.copy()

    groups = _contiguous_groups(n_channels, n_bins)
    return np.array([wavelengths[g].mean() for g in groups])
