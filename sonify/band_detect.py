"""
Auto-detect band/channel columns from a DataFrame header.

Works on any tabular dataset with band-like column naming. Does not
hardcode 'Band_1_bc' or 32 channels — those are just one pattern
among several recognized regexes.
"""

from __future__ import annotations

import re
import sys
from typing import Tuple

import pandas as pd


# Exclusion markers: columns containing these tokens (case-insensitive) are
# housekeeping / noise-estimate columns, not data channels.
_EXCLUDE_TOKENS = re.compile(r"(std|sdt|max|min|err)", re.IGNORECASE)

# Ordered list of regex patterns that identify band/channel columns.
# Each pattern must contain a named group ``num`` capturing the integer index.
_BAND_PATTERNS = [
    re.compile(r"^Band_(?P<num>\d+)_bc$", re.IGNORECASE),   # Band_1_bc
    re.compile(r"^Channel_(?P<num>\d+)$", re.IGNORECASE),    # Channel_3
    re.compile(r"^Band_(?P<num>\d+)$", re.IGNORECASE),       # Band_3
    re.compile(r"^Band(?P<num>\d+)$", re.IGNORECASE),        # Band3
    re.compile(r"^Ch_(?P<num>\d+)$", re.IGNORECASE),         # Ch_5
    re.compile(r"^ch(?P<num>\d+)$", re.IGNORECASE),          # ch5
]


def detect_band_columns(df: pd.DataFrame) -> Tuple[list[str], list[int]]:
    """Auto-detect band/channel columns and their numeric indices.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    tuple[list[str], list[int]]
        ``(column_names, band_indices)`` — parallel lists sorted by ascending
        band index.  ``column_names[i]`` is the DataFrame column name,
        ``band_indices[i]`` is the extracted integer band number.

    Raises
    ------
    ValueError
        If no band columns are detected.
    """
    matches: list[tuple[str, int]] = []

    for col in df.columns:
        # Skip housekeeping / noise-estimate columns
        if _EXCLUDE_TOKENS.search(col):
            continue

        for pattern in _BAND_PATTERNS:
            m = pattern.match(col)
            if m:
                matches.append((col, int(m.group("num"))))
                break  # first matching pattern wins for this column

    if not matches:
        raise ValueError(
            "No band/channel columns detected.  Expected column names like "
            "'Band_1_bc', 'Channel_3', 'Band3', etc."
        )

    # Sort by extracted numeric index so channels are in spectral order
    matches.sort(key=lambda pair: pair[1])

    column_names = [name for name, _ in matches]
    band_indices = [idx for _, idx in matches]
    return column_names, band_indices


def confirm_with_user(
    detected_columns: list[str], skip_confirm: bool = False
) -> bool:
    """Print detected channels and ask for user confirmation.

    Parameters
    ----------
    detected_columns : list[str]
        The detected band column names.
    skip_confirm : bool
        If True, skip the interactive prompt and return True.

    Returns
    -------
    bool
        True if the user (or the ``--yes`` flag) confirms.
    """
    n = len(detected_columns)
    print(f"\nDetected {n} spectral band column(s):")
    for i, col in enumerate(detected_columns, start=1):
        print(f"  {i:3d}. {col}")
    print()

    if skip_confirm:
        print("Auto-confirmed via --yes flag.")
        return True

    answer = input(f"Proceed with these {n} channels? [y/n] ").strip().lower()
    return answer in ("y", "yes")
