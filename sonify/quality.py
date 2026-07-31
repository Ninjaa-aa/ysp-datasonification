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


def polyphony(amplitude_matrix: np.ndarray, threshold: float = 0.05) -> dict[str, float]:
    """How many voices sound simultaneously.

    Density is the single largest driver of harshness in this toolkit: sounding
    every channel at once turns the output into a permanent cluster chord, and
    a real wind chime strikes one tube at a time.  Before voice limiting this
    dataset sounded 5+ voices on 74% of rows and all eight on 23.8%.

    Parameters
    ----------
    amplitude_matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` of amplitudes in [0, 1].
    threshold : float
        Amplitude above which a channel counts as sounding.

    Returns
    -------
    dict[str, float]
        ``mean``, ``median``, ``max`` simultaneous voices, and ``frac_solo``
        (fraction of rows sounding exactly one voice).
    """
    active = np.count_nonzero(amplitude_matrix > threshold, axis=1)
    return {
        "mean": float(np.mean(active)),
        "median": float(np.median(active)),
        "max": float(np.max(active)) if active.size else 0.0,
        "frac_solo": float(np.mean(active == 1)),
    }


def _partial_dissonance(f1: float, f2: float, a1: float, a2: float) -> float:
    """Plomp-Levelt sensory dissonance between two partials (Sethares' form).

    Peaks when the pair is separated by roughly a quarter of a critical band —
    around 55-75 Hz in the 400-900 Hz region — and falls to zero for unisons
    and for wide separations.
    """
    if f1 > f2:
        f1, f2, a1, a2 = f2, f1, a2, a1
    s = 0.24 / (0.0207 * f1 + 18.96)
    d = f2 - f1
    return min(a1, a2) * (np.exp(-3.5 * s * d) - np.exp(-5.75 * s * d))


def sensory_dissonance(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Total Sethares/Plomp-Levelt dissonance of a set of simultaneous partials.

    This is the metric that caught the toolkit's harshness after the rhythmic
    guards (``articulation``, ``onset_rate``) missed it entirely — those measure
    whether notes are separated in *time* and say nothing about whether notes
    sounding *together* clash.

    Calibration on known spectra:
      - single harmonic tone  ~0.001
      - major triad           ~0.34
      - untuned chime preset  ~0.65  (worse than a triad)

    Parameters
    ----------
    freqs : np.ndarray
        1-D partial frequencies in Hz.
    amps : np.ndarray
        1-D partial amplitudes, same length.

    Returns
    -------
    float
        Summed pairwise dissonance.  Lower is more consonant.
    """
    f = np.asarray(freqs, dtype=np.float64)
    a = np.asarray(amps, dtype=np.float64)
    if len(f) != len(a):
        raise ValueError(f"freqs ({len(f)}) and amps ({len(a)}) must be the same length")
    return float(sum(
        _partial_dissonance(f[i], f[j], a[i], a[j])
        for i in range(len(f)) for j in range(i + 1, len(f))
    ))


def spectral_roughness(
    waveform: np.ndarray,
    sample_rate: int,
    max_frames: int = 100,
    peak_height: float = 0.08,
) -> float:
    """Sensory dissonance measured on rendered audio rather than on a note list.

    Picks spectral peaks per frame and sums their pairwise dissonance, so it
    catches harshness regardless of where it came from — scale, timbre partials,
    or too many simultaneous voices.

    Reference values from this project: legacy chime 0.330, tuned chime 0.049.

    Parameters
    ----------
    waveform : np.ndarray
        1-D mono waveform.
    sample_rate : int
        Audio sample rate.
    max_frames : int
        Number of frames to sample across the file.
    peak_height : float
        Peak threshold as a fraction of the frame's loudest bin.

    Returns
    -------
    float
        Mean roughness across sounding frames, or ``nan`` if silent.
    """
    from scipy.signal import stft, find_peaks

    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < 8192:
        return float("nan")

    f, _, Z = stft(x, fs=sample_rate, nperseg=8192)
    mag = np.abs(Z)
    keep = f < 8000
    f, mag = f[keep], mag[keep]

    idx = np.linspace(0, mag.shape[1] - 1, min(max_frames, mag.shape[1])).astype(int)
    rough = []
    for k in idx:
        col = mag[:, k]
        if col.max() <= 0:
            continue
        peaks, _ = find_peaks(col, height=peak_height * col.max())
        if len(peaks) < 2:
            rough.append(0.0)
            continue
        rough.append(sensory_dissonance(f[peaks], col[peaks] / col.max()))

    return float(np.mean(rough)) if rough else float("nan")


def audibility(
    waveform: np.ndarray,
    sample_rate: int,
    floor_db: float = -40.0,
    frame_ms: float = 50.0,
) -> dict[str, float]:
    """How much of the file a listener actually hears, and the worst dead air.

    This exists because every other metric here is blind to it.  ``roughness``,
    ``articulation`` and ``spectral_flatness`` all analyse only the frames that
    *contain* sound, so a 13-minute render holding 10 seconds of audio scored
    beautifully on all three while being, to a listener, silent.  Four such
    files were sent out before anyone noticed.

    Parameters
    ----------
    waveform : np.ndarray
        1-D mono waveform.
    sample_rate : int
        Audio sample rate.
    floor_db : float
        Frame RMS below this (dBFS) counts as inaudible on normal playback.
    frame_ms : float
        Analysis frame length in milliseconds.

    Returns
    -------
    dict[str, float]
        ``audible_fraction`` (0-1), ``first_sound_s``, ``longest_gap_s``,
        and ``sounding_s``.  Times are ``nan`` when nothing is audible.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    w = max(1, int(sample_rate * frame_ms / 1000.0))
    n = len(x) // w
    if n < 1:
        return {"audible_fraction": 0.0, "first_sound_s": float("nan"),
                "longest_gap_s": float("nan"), "sounding_s": 0.0}

    frames = x[: n * w].reshape(n, w)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    audible = rms > 10.0 ** (floor_db / 20.0)
    idx = np.flatnonzero(audible)
    step = frame_ms / 1000.0

    if idx.size == 0:
        return {"audible_fraction": 0.0, "first_sound_s": float("nan"),
                "longest_gap_s": float(n * step), "sounding_s": 0.0}

    gaps = np.diff(idx) * step if idx.size > 1 else np.array([0.0])
    # Silence before the first sound and after the last also counts as dead air.
    lead = idx[0] * step
    trail = (n - 1 - idx[-1]) * step

    return {
        "audible_fraction": float(idx.size / n),
        "first_sound_s": float(lead),
        "longest_gap_s": float(max(gaps.max(), lead, trail)),
        "sounding_s": float(idx.size * step),
    }


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
        Keys: ``duration_s``, ``peak``, ``rms``, ``crest_db``, ``roughness``,
        ``articulation``, ``onset_rate``, ``spectral_flatness``,
        ``silence_fraction``, plus everything from :func:`audibility`
        (``audible_fraction``, ``first_sound_s``, ``longest_gap_s``,
        ``sounding_s``).
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    return {
        "duration_s": len(x) / sample_rate,
        "peak": float(np.abs(x).max()) if len(x) else float("nan"),
        "rms": float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan"),
        "crest_db": crest_factor_db(x),
        "roughness": spectral_roughness(x, sample_rate),
        **audibility(x, sample_rate),
        "articulation": articulation(x, sample_rate),
        "onset_rate": onset_rate(x, sample_rate),
        "spectral_flatness": spectral_flatness(x, sample_rate),
        "silence_fraction": silence_fraction(x, sample_rate),
    }
