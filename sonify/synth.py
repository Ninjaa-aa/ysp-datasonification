"""
Additive synthesis engine with phase-continuous oscillators and
anti-click boundary fading.

Produces a mono float64 waveform in [-1, 1] ready for playback or
WAV export.
"""

from __future__ import annotations

import numpy as np


def synthesize(
    amplitude_matrix: np.ndarray,
    freqs: np.ndarray,
    seconds_per_row: float,
    sample_rate: int,
    volume: float = 0.8,
) -> np.ndarray:
    """Phase-continuous additive synthesis.

    Each row in ``amplitude_matrix`` becomes one fixed-length audio segment.
    Segments are concatenated in row order.  Per-channel phase is carried
    across segment boundaries to avoid clicks.  A short raised-cosine
    fade-in/out is applied at each boundary as a second safety net.

    Parameters
    ----------
    amplitude_matrix : np.ndarray
        2-D array ``(n_rows, n_channels)`` with values in ``[0, 1]``.
    freqs : np.ndarray
        1-D array of length ``n_channels``, frequencies in Hz.
    seconds_per_row : float
        Duration of each row segment (= ``1 / playback_speed``).
    sample_rate : int
        Audio sample rate (e.g. 44100).
    volume : float
        Master gain in ``[0, 1]``, applied before peak normalization.

    Returns
    -------
    np.ndarray
        1-D float64 mono waveform, peak-normalized to ``[-1, 1]``.
    """
    n_rows, n_channels = amplitude_matrix.shape
    segment_samples = round(seconds_per_row * sample_rate)

    if segment_samples < 1:
        segment_samples = 1

    # Pre-compute fade envelope (raised-cosine), capped so at least half
    # the segment stays at full amplitude.
    default_fade = round(0.010 * sample_rate)  # 10 ms
    fade_samples = min(default_fade, segment_samples // 4)

    envelope = np.ones(segment_samples, dtype=np.float64)
    if fade_samples > 0:
        # Raised-cosine fade-in: 0 → 1
        fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        # Raised-cosine fade-out: 1 → 0
        fade_out = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        envelope[:fade_samples] = fade_in
        envelope[-fade_samples:] = fade_out

    # Time vector for one segment
    t = np.arange(segment_samples, dtype=np.float64) / sample_rate

    # Running phase per channel (carries across segments)
    phases = np.zeros(n_channels, dtype=np.float64)

    # Pre-allocate the full waveform
    total_samples = n_rows * segment_samples
    waveform = np.zeros(total_samples, dtype=np.float64)

    two_pi = 2.0 * np.pi

    for row_idx in range(n_rows):
        segment = np.zeros(segment_samples, dtype=np.float64)

        for ch in range(n_channels):
            amp = amplitude_matrix[row_idx, ch]
            freq = freqs[ch]

            # Generate phase-continuous sine
            angles = two_pi * freq * t + phases[ch]
            osc = amp * np.sin(angles)

            # Apply per-channel fade envelope
            osc *= envelope

            segment += osc

            # Update running phase for this channel
            phases[ch] += two_pi * freq * segment_samples / sample_rate
            phases[ch] %= two_pi

        # Write segment into the full waveform
        start = row_idx * segment_samples
        waveform[start : start + segment_samples] = segment

    # Gain staging
    waveform *= volume

    # Peak-normalize to [-1, 1]
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform /= peak

    return waveform
