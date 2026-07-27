"""
Objective audio-quality metrics for sonification output.

These exist because two separate sound bugs shipped in this project, and each
needed a *different* metric to catch:

1. The gain-normalization bug mapped true zeros to ~0.95 amplitude, so every
   row fired at full volume ("tak tak tak").  Caught by ``mean_amplitude`` on
   the matrix entering synthesis — the buggy pipeline read 0.62 vs 0.31 for
   the fixed one.  Note-level articulation barely moved (0.943 vs 0.974), so
   articulation alone would **not** have caught it.

2. A "legato" envelope merged adjacent rows into sustained tones, collapsing
   the chime into a continuous drone.  Caught by ``articulation`` (0.52 vs
   0.97).  Mean amplitude went *up*, so the amplitude guard alone would not
   have caught this one either.

Both metrics are therefore required.  Neither is sufficient alone.
"""

from __future__ import annotations

import numpy as np


def frame_envelope(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 5.0,
) -> np.ndarray:
    """Compute a per-frame RMS envelope, peak-normalized to [0, 1].

    Parameters
    ----------
    waveform : np.ndarray
        1-D mono waveform (or 2-D, which is averaged to mono).
    sample_rate : int
        Audio sample rate.
    frame_ms : float
        Frame length in milliseconds.

    Returns
    -------
    np.ndarray
        1-D envelope, peak-normalized.  Empty if the waveform is shorter
        than one frame.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    w = max(1, int(sample_rate * frame_ms / 1000.0))
    n = len(x) // w
    if n < 1:
        return np.array([], dtype=np.float64)

    frames = x[: n * w].reshape(n, w)
    env = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = env.max()
    return env / peak if peak > 0 else env


def articulation(
    waveform: np.ndarray,
    sample_rate: int,
    peak_height: float = 0.25,
    min_gap_ms: float = 50.0,
) -> float:
    """Measure how completely the signal falls to silence *between* notes.

    Returns ``1 - median(trough between consecutive peaks)``.  A value near
    1.0 means notes are cleanly separated by silence (wind-chime character);
    a low value means the sound never gets out of its own way (drone).

    Reference values measured on this project's outputs:
      - ``chime_full.wav`` (approved by Dr. Malaska): 0.998
      - ``chime_fixed.wav`` (approved): 0.974
      - legato-regression build: 0.516

    Parameters
    ----------
    waveform : np.ndarray
        1-D mono waveform.
    sample_rate : int
        Audio sample rate.
    peak_height : float
        Minimum normalized envelope height to count as a note peak.
    min_gap_ms : float
        Minimum spacing between detected peaks, in milliseconds.

    Returns
    -------
    float
        Articulation in [0, 1].  ``nan`` if fewer than two peaks are found
        (too little signal to judge).
    """
    from scipy.signal import find_peaks

    env = frame_envelope(waveform, sample_rate)
    if len(env) < 3:
        return float("nan")

    frame_ms = 5.0
    distance = max(1, int(min_gap_ms / frame_ms))
    peaks, _ = find_peaks(env, height=peak_height, distance=distance)
    if len(peaks) < 2:
        return float("nan")

    troughs = [
        env[a:b].min() for a, b in zip(peaks[:-1], peaks[1:]) if b - a > 2
    ]
    if not troughs:
        return float("nan")

    return float(1.0 - np.median(troughs))


def onset_rate(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 50.0,
    jump_frac: float = 0.15,
) -> float:
    """Count note attacks per second.

    For a per-row sonification this should track ``playback_speed`` — a rate
    far below it means notes are being merged (drone); far above means
    spurious retriggering.

    Parameters
    ----------
    waveform : np.ndarray
        1-D mono waveform.
    sample_rate : int
        Audio sample rate.
    frame_ms : float
        Analysis frame length in milliseconds.
    jump_frac : float
        Frame-to-frame RMS rise, as a fraction of peak, that counts as an onset.

    Returns
    -------
    float
        Onsets per second.
    """
    env = frame_envelope(waveform, sample_rate, frame_ms=frame_ms)
    if len(env) < 2:
        return 0.0

    duration = len(waveform) / sample_rate
    if duration <= 0:
        return 0.0

    n_onsets = int(np.sum(np.diff(env) > jump_frac * env.max()))
    return n_onsets / duration


def mean_amplitude(amplitude_matrix: np.ndarray) -> float:
    """Mean of the amplitude matrix entering synthesis.

    This is the guard against gain-normalization bugs.  On this dataset
    (58.6% true zeros) a correct pipeline lands near 0.31; the historical
    min-shift bug produced 0.62 because it lifted silence to ~0.95.

    Parameters
    ----------
    amplitude_matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` with values in [0, 1].

    Returns
    -------
    float
        Mean amplitude.
    """
    return float(np.mean(amplitude_matrix))


def crest_factor_db(waveform: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB.

    Higher means more dynamic contrast (distinct peaks over a quiet floor);
    lower means a compressed wall of sound.

    Returns
    -------
    float
        Crest factor in dB, or ``nan`` for a silent waveform.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    rms = np.sqrt(np.mean(x ** 2))
    if rms <= 0 or peak <= 0:
        return float("nan")
    return float(20.0 * np.log10(peak / rms))


def spectral_flatness(waveform: np.ndarray, sample_rate: int) -> float:
    """Wiener entropy: ~0 for tonal/pitched content, ~1 for white noise.

    Guards against the sonification degenerating into broadband hiss.

    Silent frames are excluded.  Digital silence is *perfectly* flat, so on a
    sparse event-driven render (which can be >99% silence) including those
    frames would report ~0.99 and falsely look like noise.  Only frames
    carrying real signal say anything about timbre.

    Returns
    -------
    float
        Mean spectral flatness across sounding frames, or ``nan`` if the
        waveform is entirely silent.
    """
    from scipy.signal import stft

    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < 64:
        return float("nan")

    nperseg = min(4096, len(x))
    _, _, Z = stft(x, fs=sample_rate, nperseg=nperseg)
    power = np.abs(Z) ** 2

    # Keep only frames with meaningful energy relative to the loudest frame.
    frame_energy = power.sum(axis=0)
    if frame_energy.max() <= 0:
        return float("nan")
    sounding = frame_energy > 1e-6 * frame_energy.max()
    if not sounding.any():
        return float("nan")

    power = power[:, sounding] + 1e-12
    gmean = np.exp(np.mean(np.log(power), axis=0))
    amean = np.mean(power, axis=0)
    return float(np.mean(gmean / amean))


def silence_fraction(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 50.0,
    threshold: float = 0.01,
) -> float:
    """Fraction of frames whose RMS is below ``threshold`` of the peak.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    env = frame_envelope(waveform, sample_rate, frame_ms=frame_ms)
    if len(env) == 0:
        return float("nan")
    return float(np.mean(env < threshold))


def describe(waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute all waveform-level metrics at once.

    Returns
    -------
    dict[str, float]
        Keys: ``duration_s``, ``peak``, ``rms``, ``crest_db``,
        ``articulation``, ``onset_rate``, ``spectral_flatness``,
        ``silence_fraction``.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    return {
        "duration_s": len(x) / sample_rate,
        "peak": float(np.abs(x).max()) if len(x) else float("nan"),
        "rms": float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan"),
        "crest_db": crest_factor_db(x),
        "articulation": articulation(x, sample_rate),
        "onset_rate": onset_rate(x, sample_rate),
        "spectral_flatness": spectral_flatness(x, sample_rate),
        "silence_fraction": silence_fraction(x, sample_rate),
    }
