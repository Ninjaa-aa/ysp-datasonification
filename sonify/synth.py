"""
Additive synthesis engine with phase-continuous oscillators and
anti-click boundary fading.

Phase 5 additions:
- Sustain: blend previous row's amplitude into current row for smoother
  transitions
- Timbre: sine (pure), bell (4 harmonic partials), chime (4 inharmonic
  partials for metallic shimmer)

Produces a mono float64 waveform in [-1, 1] ready for playback or
WAV export.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Timbre partial definitions
# ---------------------------------------------------------------------------

# Bell: harmonic partials at integer multiples of the fundamental
_BELL_RATIOS = np.array([1.0, 2.0, 3.0, 4.0])
_BELL_WEIGHTS = np.array([1.00, 0.50, 0.25, 0.12])
_BELL_NORM = float(_BELL_WEIGHTS.sum())  # ~1.87

# Chime: inharmonic partials from tubular bell overtone spectrum
# The non-integer ratios produce the characteristic metallic shimmer
_CHIME_RATIOS = np.array([1.0, 2.756, 5.404, 8.933])
_CHIME_WEIGHTS = np.array([1.00, 0.50, 0.25, 0.12])
_CHIME_NORM = float(_CHIME_WEIGHTS.sum())  # ~1.87

# Number of partials per timbre (for phase tracking)
_N_PARTIALS = {
    "sine": 1,
    "bell": len(_BELL_RATIOS),
    "chime": len(_CHIME_RATIOS),
}


def synthesize(
    amplitude_matrix: np.ndarray,
    freqs: np.ndarray,
    seconds_per_row: float,
    sample_rate: int,
    volume: float = 0.8,
    sustain: float = 0.0,
    timbre: str = "sine",
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
        Either 1-D array of length ``n_channels`` (fixed per-channel
        frequencies), or 2-D array of shape ``(n_rows, n_channels)``
        (per-row frequencies, for column-driven tone mapping).
    seconds_per_row : float
        Duration of each row segment (= ``1 / playback_speed``).
    sample_rate : int
        Audio sample rate (e.g. 44100).
    volume : float
        Master gain in ``[0, 1]``, applied before peak normalization.
    sustain : float
        Amplitude sustain blend factor in ``[0, 1]``.  0.0 = no sustain
        (Phase 1 behavior).  Higher values blend the previous row's
        amplitude into the start of the current row.
    timbre : str
        ``'sine'`` (pure sine, Phase 1 behavior), ``'bell'`` (4 harmonic
        partials), or ``'chime'`` (4 inharmonic partials for metallic
        shimmer).

    Returns
    -------
    np.ndarray
        1-D float64 mono waveform, peak-normalized to ``[-1, 1]``.

    Notes
    -----
    Bell and chime modes do 4x the oscillator work per channel.  For 32
    channels at 4000 rows, this is still fast (numpy vectorized sine is
    cheap).  At 2048 channels it becomes relevant — consider reducing
    n_bins if synthesis is too slow.
    """
    n_rows, n_channels = amplitude_matrix.shape
    segment_samples = round(seconds_per_row * sample_rate)

    if segment_samples < 1:
        segment_samples = 1

    # Determine if freqs are per-row or fixed
    per_row_freqs = freqs.ndim == 2

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

    # Determine partial count and properties based on timbre
    if timbre == "bell":
        partial_ratios = _BELL_RATIOS
        partial_weights = _BELL_WEIGHTS
        partial_norm = _BELL_NORM
        n_partials = len(partial_ratios)
    elif timbre == "chime":
        partial_ratios = _CHIME_RATIOS
        partial_weights = _CHIME_WEIGHTS
        partial_norm = _CHIME_NORM
        n_partials = len(partial_ratios)
    else:
        # sine mode — single partial
        n_partials = 1

    # Running phase per channel per partial (carries across segments)
    phases = np.zeros((n_channels, n_partials), dtype=np.float64)

    # Previous row amplitudes for sustain blending
    prev_amps = np.zeros(n_channels, dtype=np.float64)

    # Pre-allocate the full waveform
    total_samples = n_rows * segment_samples
    waveform = np.zeros(total_samples, dtype=np.float64)

    two_pi = 2.0 * np.pi

    for row_idx in range(n_rows):
        segment = np.zeros(segment_samples, dtype=np.float64)

        # Get frequencies for this row
        row_freqs = freqs[row_idx] if per_row_freqs else freqs

        for ch in range(n_channels):
            new_amp = amplitude_matrix[row_idx, ch]
            freq = row_freqs[ch]

            # ── Sustain: blend previous amplitude into start ──────────
            if sustain > 0 and row_idx > 0:
                effective_start = (1 - sustain) * new_amp + sustain * prev_amps[ch]
                # Linear ramp from effective_start to new_amp across segment
                amp_envelope = np.linspace(effective_start, new_amp, segment_samples)
            else:
                amp_envelope = new_amp  # scalar, broadcast works fine

            if timbre == "sine":
                # ── Sine mode: single partial (Phase 1 behavior) ──────
                angles = two_pi * freq * t + phases[ch, 0]
                osc = amp_envelope * np.sin(angles)
                osc *= envelope
                segment += osc

                # Update running phase
                phases[ch, 0] += two_pi * freq * segment_samples / sample_rate
                phases[ch, 0] %= two_pi
            else:
                # ── Bell / Chime: multiple partials ───────────────────
                for p in range(n_partials):
                    partial_freq = freq * partial_ratios[p]
                    partial_weight = partial_weights[p]

                    angles = two_pi * partial_freq * t + phases[ch, p]
                    osc = amp_envelope * partial_weight * np.sin(angles)
                    osc *= envelope
                    segment += osc

                    # Update running phase for this partial
                    phases[ch, p] += two_pi * partial_freq * segment_samples / sample_rate
                    phases[ch, p] %= two_pi

                # Normalize by sum of weights to keep amplitude in range
                # (applied per-channel to avoid cumulative overcount)

            prev_amps[ch] = new_amp

        # Normalize bell/chime segment by partial weight sum
        if timbre in ("bell", "chime"):
            segment /= partial_norm

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
