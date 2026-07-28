"""Harmonic quality guards: sensory dissonance, roughness, polyphony.

These exist because the rhythmic guards in ``test_audio_quality.py``
(articulation, onset rate) provably cannot see harshness.  A dense, clashing
cluster chord scores a perfect 0.998 articulation — the toolkit shipped exactly
that, sounding 5+ voices on 74% of rows with bell overtones landing 55-75 Hz
from other channels' fundamentals.

Every guard here also asserts that the *untuned* configuration fails it.  A test
that cannot fail is not a guard.
"""

import os

import numpy as np
import pandas as pd
import pytest

from sonify.quality import (
    polyphony,
    sensory_dissonance,
    spectral_roughness,
)
from sonify.mapping import (
    apply_global_gain,
    assign_frequencies_pentatonic,
    limit_voices,
    scale_values,
)
from sonify.preprocess import clean, rebin
from sonify.synth import _PARTITION_GROUPS, _TIMBRE_PROPS


_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw",
    "2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv",
)
_BANDS = [f"Band_{i}_bc" for i in range(1, 33)]

# Ceiling for the tuned chime preset. Measured 0.049 on rendered audio; the
# untuned build measured 0.330, so 0.10 separates them with wide margin.
ROUGHNESS_CEILING = 0.10


def preset_spectrum(root, octaves, n_channels=8, voices=None, low_timbre=None):
    """Build the simultaneous partial spectrum a preset would sound."""
    f0 = assign_frequencies_pentatonic(n_channels, root, octaves)
    groups = np.array_split(range(n_channels), 3)
    voices = range(n_channels) if voices is None else voices

    freqs, amps = [], []
    for ch in voices:
        gi = next(i for i, g in enumerate(groups) if ch in g)
        name = _PARTITION_GROUPS[gi][0]
        if gi == 0 and low_timbre is not None:
            name = low_timbre
        if name == "sine":
            ratios, weights = np.array([1.0]), np.array([1.0])
        else:
            ratios, weights = _TIMBRE_PROPS[name][0], _TIMBRE_PROPS[name][1]
        for r, w in zip(ratios, weights):
            freqs.append(f0[ch] * r)
            amps.append(w)
    return np.array(freqs), np.array(amps)


class TestSensoryDissonanceCalibration:
    """The model must reproduce known reference spectra, or its numbers mean nothing."""

    def test_unison_is_consonant(self):
        assert sensory_dissonance([440.0, 440.0], [1.0, 1.0]) < 0.01

    def test_single_harmonic_tone_is_consonant(self):
        freqs = np.array([220.0 * k for k in range(1, 5)])
        amps = np.array([1.0, 0.5, 0.25, 0.12])
        assert sensory_dissonance(freqs, amps) < 0.01

    def test_major_triad_is_moderate(self):
        """A major triad is consonant music but not zero — ~0.34 measured."""
        freqs, amps = [], []
        for f in (220.0, 275.0, 330.0):
            for k, w in zip(range(1, 5), [1.0, 0.5, 0.25, 0.12]):
                freqs.append(f * k)
                amps.append(w)
        assert 0.2 < sensory_dissonance(freqs, amps) < 0.5

    def test_semitone_cluster_is_dissonant(self):
        """Adjacent semitones in the rough zone must score high."""
        base = np.array([440.0, 466.2, 493.9])
        assert sensory_dissonance(base, np.ones(3)) > \
               sensory_dissonance(np.array([440.0, 880.0, 1760.0]), np.ones(3))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            sensory_dissonance([440.0, 880.0], [1.0])


class TestChimePresetHarmonics:
    """The tuned chime preset must be far more consonant than the untuned one."""

    @staticmethod
    def _trio_dissonances():
        from itertools import combinations
        return [
            sensory_dissonance(*preset_spectrum(220.0, 3, voices=c))
            for c in combinations(range(8), 3)
        ]

    def test_voice_limiting_is_the_dominant_fix(self):
        """A typical 3-voice chord is far more consonant than all 8 at once.

        Measured: mean trio 0.033 against 0.304 for all eight voices.  The mean
        is the right statistic here — which three channels are loudest varies
        row to row, so no single trio is representative.
        """
        all_v = sensory_dissonance(*preset_spectrum(220.0, 3))
        assert float(np.mean(self._trio_dissonances())) < 0.25 * all_v

    def test_even_the_worst_trio_beats_the_original(self):
        """The worst case is three adjacent low notes — rare, and still better.

        Measured 0.162 against 0.649 for the original 8-voice/bell rendering.
        """
        original = sensory_dissonance(*preset_spectrum(220.0, 3, low_timbre="bell"))
        assert max(self._trio_dissonances()) < 0.5 * original

    def test_soft_low_group_beats_bell(self):
        """bell's 2x/3x/4x harmonics collide with other fundamentals."""
        with_bell = sensory_dissonance(*preset_spectrum(220.0, 3, low_timbre="bell"))
        with_soft = sensory_dissonance(*preset_spectrum(220.0, 3, low_timbre="soft"))
        assert with_soft < with_bell

    def test_untuned_config_would_fail(self):
        """Proves the guard has teeth: 8 voices with bell lows is genuinely bad."""
        untuned = sensory_dissonance(*preset_spectrum(220.0, 3, low_timbre="bell"))
        triad = 0.34
        assert untuned > triad, "untuned preset should be worse than a major triad"

    def test_partition_low_group_is_soft(self):
        """Regression guard on the partition table itself."""
        assert _PARTITION_GROUPS[0][0] == "soft"

    def test_soft_timbre_has_quiet_octave_only(self):
        ratios, weights = _TIMBRE_PROPS["soft"][0], _TIMBRE_PROPS["soft"][1]
        np.testing.assert_array_equal(ratios, [1.0, 2.0])
        assert weights[1] <= 0.2, "octave partial must stay quiet"


class TestPolyphony:
    def test_counts_simultaneous_voices(self):
        m = np.array([[0.9, 0.0, 0.0], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0]])
        p = polyphony(m)
        assert p["max"] == 3
        assert p["frac_solo"] == pytest.approx(1 / 3)

    def test_limit_voices_caps_polyphony(self):
        rng = np.random.default_rng(0)
        m = rng.random((200, 8))
        assert polyphony(limit_voices(m, 3))["max"] <= 3

    def test_unlimited_polyphony_is_high(self):
        """Teeth: without limiting, all voices sound."""
        rng = np.random.default_rng(0)
        m = rng.random((200, 8))
        assert polyphony(m)["mean"] > 5


@pytest.mark.skipif(not os.path.isfile(_DATA), reason="borehole dataset not present")
class TestRealDataPolyphony:
    """On the actual dataset, voice limiting must genuinely reduce density."""

    @staticmethod
    def _amplitudes():
        import contextlib, io
        df = pd.read_csv(_DATA).iloc[:1000]
        m = scale_values(rebin(clean(df, _BANDS), 8), "log10")
        with contextlib.redirect_stdout(io.StringIO()):
            return apply_global_gain(m, "max_log")

    def test_untuned_dataset_is_dense(self):
        """Teeth: this is the defect — a permanent cluster chord."""
        assert polyphony(self._amplitudes())["mean"] > 5.0

    def test_limited_dataset_respects_cap(self):
        p = polyphony(limit_voices(self._amplitudes(), 3))
        assert p["mean"] <= 3.0
        assert p["max"] <= 3

    def test_limiting_raises_solo_fraction(self):
        amp = self._amplitudes()
        assert polyphony(limit_voices(amp, 1))["frac_solo"] > \
               polyphony(amp)["frac_solo"]


class TestSpectralRoughness:
    def test_pure_tone_is_smooth(self):
        sr = 44100
        t = np.arange(sr) / sr
        assert spectral_roughness(np.sin(2 * np.pi * 440 * t), sr) < 0.05

    def test_beating_pair_is_rough(self):
        """Two tones ~60 Hz apart sit at the peak of the roughness curve."""
        sr = 44100
        t = np.arange(sr) / sr
        rough = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 560 * t)
        smooth = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 1000 * t)
        assert spectral_roughness(rough, sr) > spectral_roughness(smooth, sr)

    def test_silence_returns_nan(self):
        assert np.isnan(spectral_roughness(np.zeros(44100), 44100))
