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


def _get_excluded_columns(df: pd.DataFrame, band_cols: list[str]) -> tuple[list[str], list[str]]:
    """Identify excluded (noise) and housekeeping columns.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(noise_columns, housekeeping_columns)``
    """
    band_set = set(band_cols)
    noise_cols = []
    housekeeping_cols = []

    for col in df.columns:
        if col in band_set:
            continue
        if _EXCLUDE_TOKENS.search(col):
            noise_cols.append(col)
        else:
            housekeeping_cols.append(col)

    return noise_cols, housekeeping_cols


def _format_channel_table(
    detected_columns: list[str],
    band_indices: list[int],
    wavelength_table: dict[int, float] | None = None,
) -> str:
    """Format detected channels as a readable table string."""
    n = len(detected_columns)
    has_wl = wavelength_table is not None

    # Header
    if has_wl:
        header = f"  {'Index':>5}  {'Column Name':<25}  {'Band No.':>8}   {'Wavelength (nm)':>15}"
        separator = f"  {'-----':>5}  {'-------------------------':<25}  {'--------':>8}   {'---------------':>15}"
    else:
        header = f"  {'Index':>5}  {'Column Name':<25}  {'Band No.':>8}"
        separator = f"  {'-----':>5}  {'-------------------------':<25}  {'--------':>8}"

    lines = [header, separator]

    for i, (col, band_idx) in enumerate(zip(detected_columns, band_indices), start=1):
        if has_wl:
            wl = wavelength_table.get(band_idx, 0.0)
            wl_str = f"{wl:.1f}" if wl > 0 else "—"
            lines.append(f"  {i:5d}  {col:<25}  {band_idx:8d}   {wl_str:>15}")
        else:
            lines.append(f"  {i:5d}  {col:<25}  {band_idx:8d}")

    return "\n".join(lines)


def confirm_with_user(
    detected_columns: list[str],
    band_indices: list[int],
    skip_confirm: bool = False,
    wavelength_table: dict[int, float] | None = None,
    all_columns: list[str] | None = None,
    df: pd.DataFrame | None = None,
) -> list[str]:
    """Print detected channels in formatted table and ask for user confirmation.

    Parameters
    ----------
    detected_columns : list[str]
        The detected band column names.
    band_indices : list[int]
        The extracted band numbers (parallel to detected_columns).
    skip_confirm : bool
        If True, skip the interactive prompt and return the detected columns.
    wavelength_table : dict[int, float] or None
        Band number → wavelength mapping for display.
    all_columns : list[str] or None
        All column names in the DataFrame (for the edit-add submenu).
    df : pd.DataFrame or None
        The DataFrame (for identifying excluded/housekeeping columns).

    Returns
    -------
    list[str]
        The confirmed (possibly edited) list of band column names.
    """
    n = len(detected_columns)
    print(f"\nDetected {n} spectral channel(s):\n")

    # Format and print the table
    table = _format_channel_table(detected_columns, band_indices, wavelength_table)
    print(table)

    # Show excluded / housekeeping summary
    if df is not None:
        noise_cols, housekeeping_cols = _get_excluded_columns(df, detected_columns)
        if noise_cols:
            if len(noise_cols) > 3:
                summary = f"{noise_cols[0]} ... ({len(noise_cols)} columns)"
            else:
                summary = ", ".join(noise_cols)
            print(f"\n  Excluded (noise/housekeeping): {summary}")
        if housekeeping_cols:
            if len(housekeeping_cols) > 6:
                summary = ", ".join(housekeeping_cols[:6]) + f" ... ({len(housekeeping_cols)} columns)"
            else:
                summary = ", ".join(housekeeping_cols)
            print(f"  Housekeeping: {summary}")

    print()

    if skip_confirm:
        print("Auto-confirmed via --yes flag.")
        return list(detected_columns)

    # ── Interactive confirmation loop ─────────────────────────────────────
    current_cols = list(detected_columns)
    current_indices = list(band_indices)

    while True:
        answer = input(
            f"Proceed with these {len(current_cols)} channels? [y/n/edit] "
        ).strip().lower()

        if answer in ("y", "yes"):
            return current_cols
        elif answer in ("n", "no"):
            return []  # empty list signals abort
        elif answer in ("e", "edit"):
            current_cols, current_indices = _edit_channels(
                current_cols, current_indices, all_columns or []
            )
            # Re-display updated table
            print(f"\nUpdated to {len(current_cols)} channel(s):\n")
            table = _format_channel_table(
                current_cols, current_indices, wavelength_table
            )
            print(table)
            print()
        else:
            print("Please enter 'y', 'n', or 'edit'.")


def _edit_channels(
    columns: list[str],
    indices: list[int],
    all_columns: list[str],
) -> tuple[list[str], list[int]]:
    """Interactive sub-menu for editing channel selection."""
    while True:
        print("\n  [a] Add a column by name")
        print("  [r] Remove a column by index")
        print("  [d] Done")
        choice = input("  Choice: ").strip().lower()

        if choice == "a":
            col_name = input("  Column name to add: ").strip()
            if col_name in columns:
                print(f"  '{col_name}' is already in the channel list.")
            elif col_name not in all_columns:
                print(f"  '{col_name}' not found in the dataset columns.")
            else:
                columns.append(col_name)
                # Assign the next index
                next_idx = max(indices) + 1 if indices else 1
                indices.append(next_idx)
                print(f"  Added '{col_name}' at index {next_idx}.")
        elif choice == "r":
            try:
                idx_str = input("  Index to remove (1-based): ").strip()
                idx = int(idx_str) - 1
                if 0 <= idx < len(columns):
                    removed = columns.pop(idx)
                    indices.pop(idx)
                    print(f"  Removed '{removed}'.")
                else:
                    print(f"  Index out of range. Valid: 1-{len(columns)}.")
            except ValueError:
                print("  Please enter a valid integer.")
        elif choice == "d":
            break
        else:
            print("  Please enter 'a', 'r', or 'd'.")

    return columns, indices
