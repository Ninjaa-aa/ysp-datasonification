"""Tests for sonify.quality and the sound-quality regression guards.

These guards exist because two different sound bugs shipped, each needing a
different metric.  Every guard here also asserts that the *broken* case fails —
a test that cannot fail is not a guard.
"""

import os

import numpy as np
import pandas as pd
import pytest

from sonify.quality import (
    articulation,
    crest_factor_db,
    frame_envelope,
    mean_amplitude,
    onset_rate,
    silence_fraction,
    spectral_flatness,
    describe,
)
from sonify.preprocess import clean, rebin
from sonify.mapping import scale_values, apply_global_gain
from sonify.synth import synthesize


_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw",
    "2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv",
)
_BANDS = [f"Band_{i}_bc" for i in range(1, 33)]

SR = 44100


def _pulse_train(n_pulses=10, sr=SR, period_s=0.2, tone_s=0.05, freq=440.0):
    """Discrete notes separated by true silence (well-articulated)."""
    out = np.zeros(int(n_pulses * period_s * sr))
    tone_n = int(tone_s * sr)
    t = np.arange(tone_n) / sr
    env = np.exp(-t / 0.01)
    for i in range(n_pulses):
        start = int(i * period_s * sr)
        out[start:start + tone_n] = env * np.sin(2 * np.pi * freq * t)
    return out


def _continuous_tone(duration_s=2.0, sr=SR, freq=440.0):
    """An unbroken tone (a drone — the legato failure mode)."""
    t = np.arange(int(duration_s * sr)) / sr
    return 0.8 * np.sin(2 * np.pi * freq * t)


class TestFrameEnvelope:
    def test_normalized_to_unit_peak(self):
        env = frame_envelope(_pulse_train(), SR)
        assert env.max() == pytest.approx(1.0)
        assert env.min() >= 0.0

    def test_short_input_returns_empty(self):
        assert len(frame_envelope(np.zeros(3), SR)) == 0


class TestArticulation:
    def test_pulse_train_is_well_articulated(self):
        assert articulation(_pulse_train(), SR) > 0.95

    def test_continuous_tone_is_not_articulated(self):
        """The drone failure mode must score low — this is the legato guard."""
        art = articulation(_continuous_tone(), SR)
        assert np.isnan(art) or art < 0.5

    def test_insufficient_peaks_returns_nan(self):
        assert np.isnan(articulation(np.zeros(SR), SR))


class TestOnsetRate:
    def test_matches_known_pulse_rate(self):
        # 10 pulses over 2.0 s => 5 onsets/sec
        rate = onset_rate(_pulse_train(n_pulses=10, period_s=0.2), SR)
        assert rate == pytest.approx(5.0, abs=1.0)

    def test_continuous_tone_has_almost_no_onsets(self):
        assert onset_rate(_continuous_tone(), SR) < 1.0


class TestOtherMetrics:
    def test_crest_factor_sine_is_about_3db(self):
        t = np.arange(SR) / SR
        assert crest_factor_db(np.sin(2 * np.pi * 440 * t)) == pytest.approx(3.0, abs=0.2)

    def test_spectral_flatness_tone_low_noise_high(self):
        t = np.arange(SR) / SR
        tone = np.sin(2 * np.pi * 440 * t)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.3, SR)
        assert spectral_flatness(tone, SR) < spectral_flatness(noise, SR)

    def test_silence_fraction_of_silence_is_one(self):
        assert silence_fraction(np.zeros(SR), SR) == pytest.approx(1.0)

    def test_describe_returns_all_keys(self):
        m = describe(_pulse_train(), SR)
        assert set(m) == {
            "duration_s", "peak", "rms", "crest_db", "roughness",
            "articulation", "onset_rate", "spectral_flatness",
            "silence_fraction",
        }


@pytest.mark.skipif(not os.path.isfile(_DATA), reason="borehole dataset not present")
class TestPipelineAmplitudeGuard:
    """Guards the gain-normalization family of bugs.

    Articulation does NOT catch these (the broken build scored 0.943 vs 0.974
    for the good one), so amplitude-domain checks are what matter here.

    Both assertions are deliberately *slice-independent*.  An absolute mean
    threshold is not usable: a correct pipeline reads 0.28 on sparse rows but
    0.50 on the dense near-surface zone, which overlaps the range a min-shift
    produces on sparse data.  So silence is checked as an exact invariant, and
    dynamic range is checked comparatively on the same slice.
    """

    SLICES = [(0, 400), (1500, 1900), (3000, 3400)]

    @staticmethod
    def _scaled(df):
        return scale_values(rebin(clean(df, _BANDS), 8), "log10")

    @staticmethod
    def _gain(m, gain_mode="max_log"):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):  # gain fn prints a report
            return apply_global_gain(m, gain_mode)

    @staticmethod
    def _min_shift(m):
        """The transform the pipeline must never go back to.

        Shifting by the global minimum compresses every nonzero value toward
        full scale, which is what made silence-adjacent noise as loud as real
        fluorescence peaks.
        """
        b = m - m.min()
        return np.clip(b / b.max(), 0.0, 1.0)

    @pytest.mark.parametrize("start,end", SLICES)
    def test_source_zeros_stay_silent(self, start, end):
        """Exact invariant: a zero in the source is silence in the output."""
        df = pd.read_csv(_DATA).iloc[start:end]
        raw = rebin(clean(df, _BANDS), 8)
        amp = self._gain(self._scaled(df))
        zeros = raw == 0
        assert zeros.any(), "slice should contain some silence"
        assert amp[zeros].max() < 0.01

    @pytest.mark.parametrize("start,end", SLICES)
    def test_dynamic_range_beats_min_shift(self, start, end):
        """The pipeline must stay clearly quieter than a range-compressing shift.

        Measured separation is 0.24-0.34 across slices; 0.1 is a safe floor.
        """
        df = pd.read_csv(_DATA).iloc[start:end]
        m = self._scaled(df)
        assert mean_amplitude(self._gain(m)) < mean_amplitude(self._min_shift(m)) - 0.1


@pytest.mark.skipif(not os.path.isfile(_DATA), reason="borehole dataset not present")
class TestSynthesisArticulationGuard:
    """Guards the legato regression that collapsed the chime into a drone.

    Mean amplitude does NOT catch this (it went *up* when the sound broke),
    so articulation is the metric that matters here.
    """

    @staticmethod
    def _render(n_rows=200, playback_speed=5.0):
        df = pd.read_csv(_DATA).iloc[:n_rows]
        m = rebin(clean(df, _BANDS), 8)
        m = scale_values(m, "log10")
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            amp = apply_global_gain(m, "max_log")
        freqs = np.array([220., 275., 366.7, 495., 660., 880., 1100., 1467.])
        return synthesize(
            amp, freqs, 1.0 / playback_speed, SR,
            timbre="chime", adsr_shape="tight", timbre_partition=True,
        )

    def test_chime_preset_stays_articulated(self):
        """Reference builds score 0.974-0.998; the legato regression scored 0.516."""
        assert articulation(self._render(), SR) >= 0.95

    def test_onset_rate_tracks_playback_speed(self):
        """One attack per row. The legato regression dropped this to 0.10/s."""
        assert onset_rate(self._render(playback_speed=5.0), SR) == pytest.approx(5.0, abs=1.0)

    def test_guard_rejects_a_drone(self):
        """A sustained drone must fail the guard, proving it has teeth."""
        art = articulation(_continuous_tone(duration_s=5.0), SR)
        assert np.isnan(art) or art < 0.95
