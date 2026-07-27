"""
Event detection: which measurements should make a sound at all.

This implements the first half of the two-function design Dr. Malaska set out
on 2026-07-24:

    "Set a user input function (type and threshold) for which peaks should have
     sound (in this case, linear with appropriate threshold would be good) and
     then another user input function for how that trigger gets sonified
     (intensity converted to a log or linear scale?)."

Function A — the *trigger* — lives here and decides which rows sound.
Function B — the *intensity encoding* — stays in ``mapping.scale_values()`` /
``mapping.apply_global_gain()`` and decides how loud they are.  Keeping them
separate is the point: you can gate on a linear threshold while still encoding
loudness logarithmically.

Background
----------
The borehole dataset is 58.6% exact zeros with a heavy log tail (nonzero median
26, p99 406, max 4921).  Sonifying every row therefore spends most of its time
amplifying background.  Dr. Malaska's 2026-07-09 threshold study
(``data/threshold/``) quantified this — a row counts as a "hit" if *any* band
exceeds the threshold:

    threshold   100    200   300   400   500   600   700   800   900  1000
    signals    1498    243    75    50    34    25    19    13    10     8

with his note: "How many tones would we like to hear in a dataset? Probably
above 10."  That puts the useful working range at roughly 400-900.
``threshold_for_target_tones()`` solves that table backwards.
"""

from __future__ import annotations

import numpy as np


# Reproduces Dr. Malaska's 2026-07-09 spreadsheet; used for documentation and
# as the fixture for the regression test in tests/test_events.py.
MALASKA_THRESHOLD_TABLE = {
    100: 1498, 200: 243, 300: 75, 400: 50, 500: 34,
    600: 25, 700: 19, 800: 13, 900: 10, 1000: 8,
    2000: 4, 3000: 2, 4000: 1,
}


def count_signals(matrix: np.ndarray, threshold: float) -> int:
    """Count rows where *any* channel exceeds ``threshold``.

    This is Dr. Malaska's definition of a "hit" and reproduces his table
    exactly against the raw (unscaled, uncleaned-of-sparsity) band matrix.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` of raw linear band values.
    threshold : float
        Signal intensity threshold.

    Returns
    -------
    int
        Number of rows containing at least one supra-threshold value.
    """
    return int(np.count_nonzero((matrix > threshold).any(axis=1)))


def row_trigger_mask(
    matrix: np.ndarray,
    threshold: float,
    trigger_type: str = "linear",
) -> np.ndarray:
    """Return a per-row boolean mask of which rows should sound.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``, raw linear band values (>= 0).
    threshold : float
        Trigger threshold.  ``<= 0`` triggers every row that has any signal.
    trigger_type : str
        ``'linear'`` — compare raw values against ``threshold`` (Dr. Malaska's
        recommendation, and what reproduces his table).
        ``'log'`` — compare ``log10`` of the values against ``log10(threshold)``,
        for datasets where a multiplicative criterion is more natural.

    Returns
    -------
    np.ndarray
        1-D boolean array of length ``n_rows``.
    """
    if trigger_type not in ("linear", "log"):
        raise ValueError(
            f"trigger_type must be 'linear' or 'log', got '{trigger_type}'"
        )

    if threshold <= 0:
        return (matrix > 0).any(axis=1)

    if trigger_type == "linear":
        return (matrix > threshold).any(axis=1)

    # Log trigger: identical ordering to linear for positive data, but stated
    # in dex so it stays meaningful if the caller thinks in orders of magnitude.
    eps = 1e-10
    return (np.log10(matrix + eps) > np.log10(threshold)).any(axis=1)


def apply_trigger(
    matrix: np.ndarray,
    threshold: float,
    trigger_type: str = "linear",
) -> np.ndarray:
    """Silence every row that does not clear the trigger.

    Whole rows are gated rather than individual cells: a fluorescence event is
    a spectral shape across bands, so gating cell-by-cell would shred the very
    peak that triggered it and leave a thin, artificial-sounding remnant.

    Must be called on raw/cleaned values, *before* ``scale_values()`` — a
    linear threshold compared against log-scaled magnitudes is meaningless.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``, values >= 0 (post ``clean()``).
    threshold : float
        Trigger threshold.  ``0.0`` disables gating entirely.
    trigger_type : str
        ``'linear'`` or ``'log'``.

    Returns
    -------
    np.ndarray
        Gated copy of the matrix (never mutates the input).
    """
    if threshold <= 0.0:
        return matrix.copy()

    mask = row_trigger_mask(matrix, threshold, trigger_type)
    result = matrix.copy()
    result[~mask] = 0.0
    return result


def threshold_for_target_tones(
    matrix: np.ndarray,
    target_tones: int,
    lo: float = 1.0,
    hi: float | None = None,
) -> float:
    """Find the threshold that yields approximately ``target_tones`` events.

    Inverts Dr. Malaska's threshold curve by bisection, so a user can ask for
    "about 25 tones" instead of guessing an intensity.  The curve is a
    monotonically decreasing step function, so the result is the threshold
    whose signal count is closest to the target.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` of raw linear band values.
    target_tones : int
        Desired number of sounding events (Dr. Malaska: "probably above 10").
    lo : float
        Lower bound of the search.
    hi : float or None
        Upper bound; defaults to the matrix maximum.

    Returns
    -------
    float
        Threshold producing the closest achievable count to ``target_tones``.

    Raises
    ------
    ValueError
        If ``target_tones`` is not positive.
    """
    if target_tones < 1:
        raise ValueError(f"target_tones must be >= 1, got {target_tones}")

    if hi is None:
        hi = float(np.max(matrix))
    if hi <= lo:
        return lo

    # Bisection on a decreasing step function.
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if count_signals(matrix, mid) > target_tones:
            lo = mid  # too many events -> raise the threshold
        else:
            hi = mid  # too few -> lower it

    # lo and hi now bracket the step; return whichever lands closer.
    best = min(
        (lo, hi),
        key=lambda t: abs(count_signals(matrix, t) - target_tones),
    )
    return float(best)


def find_event_clusters(mask: np.ndarray) -> list[tuple[int, int]]:
    """Group a boolean row mask into contiguous runs.

    Real fluorescence events span several adjacent rows — in this dataset they
    arrive in clusters (e.g. rows 197-201, then 359-364) separated by gaps of
    200-450 rows.  Treating each cluster as one note, rather than one note per
    row, is what makes the result read as a discrete event instead of a
    stutter.

    Parameters
    ----------
    mask : np.ndarray
        1-D boolean array; ``True`` where the row should sound.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` half-open index pairs, one per contiguous run.
    """
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []

    # Pad with False so transitions at the array edges are detected.
    padded = np.concatenate(([False], m, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def summarize_events(
    matrix: np.ndarray,
    mask: np.ndarray,
) -> list[dict]:
    """Describe each event cluster: location, peak band, peak value.

    Useful for logging and for driving lambda-max pitch mapping.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` of raw linear band values.
    mask : np.ndarray
        1-D boolean row mask.

    Returns
    -------
    list[dict]
        One dict per cluster with keys ``start``, ``end``, ``n_rows``,
        ``peak_row``, ``peak_band``, ``peak_value``.
    """
    events = []
    for start, end in find_event_clusters(mask):
        block = matrix[start:end]
        flat_idx = int(np.argmax(block))
        r, c = np.unravel_index(flat_idx, block.shape)
        events.append({
            "start": start,
            "end": end,
            "n_rows": end - start,
            "peak_row": start + int(r),
            "peak_band": int(c),
            "peak_value": float(block[r, c]),
        })
    return events
