"""
Additive synthesis engine with phase-continuous oscillators, ADSR
envelopes, and timbral partitioning.

Sound Quality update:
- ADSR envelope replaces the old fade-in/fade-out + sustain ramp
- Timbral partitioning: split channels into spectral groups with different
  timbres and ADSR shapes (bell/chime/sine)
- Sustain parameter kept for backward compat but deprecated

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
# Reduced to 2 partials only to prevent frequency crowding between pentatonic notes
_CHIME_RATIOS = np.array([1.0, 2.756])
_CHIME_WEIGHTS = np.array([1.00, 0.35])
_CHIME_NORM = float(_CHIME_WEIGHTS.sum())  # 1.35

# Number of partials per timbre (for phase tracking)
_N_PARTIALS = {
    "sine": 1,
    "bell": len(_BELL_RATIOS),
    "chime": len(_CHIME_RATIOS),
}

# Timbre properties lookup: (ratios, weights, norm, n_partials)
_TIMBRE_PROPS = {
    "bell":  (_BELL_RATIOS, _BELL_WEIGHTS, _BELL_NORM, len(_BELL_RATIOS)),
    "chime": (_CHIME_RATIOS, _CHIME_WEIGHTS, _CHIME_NORM, len(_CHIME_RATIOS)),
}


# ---------------------------------------------------------------------------
# ADSR envelope shapes
# ---------------------------------------------------------------------------

# (attack_ms, decay_ms, sustain_level, release_ms)
ADSR_SHAPES = {
    "tight":   (5,  90,  0.0, 5),    # percussive, staccato plink
    "natural": (15, 60,  0.3, 80),   # bell-like, default
    "slow":    (50, 100, 0.2, 150),  # pad/ambient
}

# Timbral partition assignments: group -> (timbre, adsr_shape)
_PARTITION_GROUPS = {
    0: ("bell",  "natural"),  # deep UV / low bands
    1: ("chime", "tight"),    # mid UV / middle bands
    2: ("sine",  "slow"),     # near UV / high bands
}


def generate_adsr_envelope(
    segment_samples: int,
    sample_rate: int,
    attack_ms: float,
    decay_ms: float,
    sustain_level: float,
    release_ms: float,
) -> np.ndarray:
    """Generate a normalized ADSR amplitude envelope for one row segment.

    The four phases:
      Attack:  ramp from 0 → 1.0 over attack_ms milliseconds
      Decay:   ramp from 1.0 → sustain_level over decay_ms milliseconds
      Sustain: hold at sustain_level for the remaining middle of the segment
      Release: ramp from sustain_level → 0 over release_ms milliseconds

    Each phase is capped so no single phase can consume more than 20% of
    the segment, even at very high playback speeds.

    Parameters
    ----------
    segment_samples : int
        Total samples in this row segment.
    sample_rate : int
        Samples per second (e.g. 44100).
    attack_ms : float
        Attack time in milliseconds.
    decay_ms : float
        Decay time in milliseconds.
    sustain_level : float
        Amplitude level during sustain (0.0–1.0).
    release_ms : float
        Release time in milliseconds.

    Returns
    -------
    np.ndarray
        Shape ``(segment_samples,)``, values in [0.0, 1.0].
    """
    if segment_samples <= 0:
        return np.array([], dtype=np.float64)

    cap = max(1, segment_samples // 5)

    a = min(int(attack_ms  * sample_rate / 1000), cap)
    d = min(int(decay_ms   * sample_rate / 1000), cap)
    r = min(int(release_ms * sample_rate / 1000), cap)
    s = max(0, segment_samples - a - d - r)

    parts = []
    if a > 0:
        parts.append(np.linspace(0.0, 1.0, a))
    if d > 0:
        t_decay = np.linspace(0, 1, d)
        tau = 0.3
        decay = sustain_level + (1.0 - sustain_level) * np.exp(-t_decay / tau)
        decay = (decay - decay[-1]) / (decay[0] - decay[-1] + 1e-10)
        decay = sustain_level + decay * (1.0 - sustain_level)
        parts.append(decay)
    if s > 0:
        parts.append(np.full(s, sustain_level))
    if r > 0:
        t_release = np.linspace(0, 1, r)
        release = sustain_level * np.exp(-t_release / 0.3)
        release = (release - release[-1]) / (release[0] - release[-1] + 1e-10)
        release = release * sustain_level
        parts.append(release)

    if not parts:
        return np.zeros(segment_samples, dtype=np.float64)

    envelope = np.concatenate(parts)

    # Guard: length must match segment_samples exactly (rounding can cause off-by-one)
    if len(envelope) < segment_samples:
        pad_val = envelope[-1] if len(envelope) > 0 else 0.0
        envelope = np.append(
            envelope, np.full(segment_samples - len(envelope), pad_val)
        )
    return envelope[:segment_samples]


def synthesize(
    amplitude_matrix: np.ndarray,
    freqs: np.ndarray,
    seconds_per_row: float,
    sample_rate: int,
    volume: float = 0.8,
    sustain: float = 0.0,  # deprecated: ignored when adsr_shape is set
    timbre: str = "sine",
    adsr_shape: str = "natural",
    timbre_partition: bool = True,
) -> np.ndarray:
    """Phase-continuous additive synthesis with ADSR envelopes.

    Each row in ``amplitude_matrix`` becomes one fixed-length audio segment.
    Segments are concatenated in row order.  Per-channel phase is carried
    across segment boundaries to avoid clicks.  A proper ADSR envelope
    shapes the amplitude of each note.

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
        Deprecated. Kept for backward compatibility. Ignored — ADSR
        envelope handles all amplitude shaping.
    timbre : str
        ``'sine'`` (pure sine), ``'bell'`` (4 harmonic partials), or
        ``'chime'`` (4 inharmonic partials).  **Ignored when
        ``timbre_partition=True``** — each spectral group uses its own
        timbre in that case.
    adsr_shape : str
        ``'tight'``, ``'natural'``, or ``'slow'``.  Controls the ADSR
        envelope shape.  **Overridden per-group when
        ``timbre_partition=True``.**
    timbre_partition : bool
        If True (default), split channels into 3 spectral groups with
        different timbres and ADSR shapes:
        - Group 0 (low bands): bell + natural ADSR
        - Group 1 (mid bands): chime + tight ADSR
        - Group 2 (high bands): sine + slow ADSR
        If False, all channels use the global ``timbre`` and ``adsr_shape``.

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

    # Time vector for one segment
    t = np.arange(segment_samples, dtype=np.float64) / sample_rate

    # ── Build per-channel assignments ──────────────────────────────────
    # Each channel gets: timbre_name, adsr_params, n_partials, partial_ratios,
    # partial_weights, partial_norm
    ch_timbre = [timbre] * n_channels
    ch_adsr = [ADSR_SHAPES[adsr_shape]] * n_channels

    if timbre_partition and n_channels >= 3:
        # Split into 3 groups by index
        groups = np.array_split(range(n_channels), 3)
        for group_idx, channel_indices in enumerate(groups):
            grp_timbre, grp_adsr_name = _PARTITION_GROUPS[group_idx]
            grp_adsr = ADSR_SHAPES[grp_adsr_name]
            for ch in channel_indices:
                ch_timbre[ch] = grp_timbre
                ch_adsr[ch] = grp_adsr

    # Determine max partials across all channels for phase tracking
    max_partials = max(_N_PARTIALS.get(t, 1) for t in ch_timbre)

    # Running phase per channel per partial (carries across segments)
    phases = np.zeros((n_channels, max_partials), dtype=np.float64)

    # Pre-allocate the full waveform
    total_samples = n_rows * segment_samples
    waveform = np.zeros(total_samples, dtype=np.float64)

    two_pi = 2.0 * np.pi

    for row_idx in range(n_rows):
        segment = np.zeros(segment_samples, dtype=np.float64)

        # Get frequencies for this row
        row_freqs = freqs[row_idx] if per_row_freqs else freqs

        for ch in range(n_channels):
            amp = amplitude_matrix[row_idx, ch]
            freq = row_freqs[ch]
            t_name = ch_timbre[ch]
            adsr_params = ch_adsr[ch]

            # Generate ADSR envelope for this segment
            envelope = generate_adsr_envelope(
                segment_samples, sample_rate, *adsr_params
            )

            if t_name == "sine":
                # ── Sine mode: single partial ─────────────────────────
                angles = two_pi * freq * t + phases[ch, 0]
                osc = amp * envelope * np.sin(angles)
                segment += osc

                # Update running phase
                phases[ch, 0] += two_pi * freq * segment_samples / sample_rate
                phases[ch, 0] %= two_pi
            else:
                # ── Bell / Chime: multiple partials ───────────────────
                partial_ratios, partial_weights, partial_norm, n_partials = (
                    _TIMBRE_PROPS[t_name]
                )
                ch_signal = np.zeros(segment_samples, dtype=np.float64)

                for p in range(n_partials):
                    partial_freq = freq * partial_ratios[p]
                    partial_weight = partial_weights[p]

                    angles = two_pi * partial_freq * t + phases[ch, p]
                    osc = partial_weight * np.sin(angles)
                    ch_signal += osc

                    # Update running phase for this partial
                    phases[ch, p] += (
                        two_pi * partial_freq * segment_samples / sample_rate
                    )
                    phases[ch, p] %= two_pi

                # Normalize by sum of weights, apply amplitude and envelope
                ch_signal = amp * envelope * ch_signal / partial_norm
                segment += ch_signal

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
