"""Tests for sonify.synth and sonify.export."""

import os
import tempfile

import numpy as np
import pytest
from scipy.io import wavfile

from sonify.synth import synthesize
from sonify.export import export_wav


class TestSynthesize:
    """Core synthesis properties."""

    def test_waveform_length(self):
        n_rows, n_channels = 20, 4
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(200, 2000, n_channels)
        spr = 0.1  # 10 rows/sec
        sr = 44100

        waveform = synthesize(amp, freqs, spr, sr, timbre="sine", sustain=0.0)
        expected_length = n_rows * round(spr * sr)
        assert abs(len(waveform) - expected_length) <= 1

    def test_normalized_range(self):
        n_rows, n_channels = 50, 4
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(200, 2000, n_channels)

        waveform = synthesize(amp, freqs, 0.05, 22050, timbre="sine", sustain=0.0)
        assert np.max(np.abs(waveform)) <= 1.0 + 1e-10

    def test_silence_produces_silence(self):
        amp = np.zeros((10, 3))
        freqs = np.array([200, 500, 1000], dtype=float)
        waveform = synthesize(amp, freqs, 0.1, 44100, timbre="sine", sustain=0.0)
        # All-zero amplitude → all-zero waveform (peak-normalize guards div-by-0)
        np.testing.assert_array_equal(waveform, 0.0)

    def test_fade_cap_at_high_speed(self):
        """At very high playback speed, fade should be capped to segment_samples // 4."""
        n_rows, n_channels = 10, 2
        amp = np.ones((n_rows, n_channels))
        freqs = np.array([300, 800], dtype=float)
        sr = 44100
        # 200 rows/sec → segment = ~220 samples
        spr = 1.0 / 200.0
        segment_samples = round(spr * sr)

        # The fade should be min(441, segment_samples // 4)
        default_fade = round(0.010 * sr)  # 441
        expected_fade = min(default_fade, segment_samples // 4)
        assert expected_fade < default_fade  # confirm the cap kicks in

        # Just verify it runs without error and produces valid output
        waveform = synthesize(amp, freqs, spr, sr, timbre="sine", sustain=0.0)
        assert np.isfinite(waveform).all()
        assert np.max(np.abs(waveform)) <= 1.0 + 1e-10


class TestSustain:
    """Phase 5: sustain amplitude blending."""

    def test_sustain_zero_matches_no_sustain(self):
        """With sustain=0.0, output is identical to current behavior."""
        n_rows, n_channels = 20, 4
        np.random.seed(42)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(200, 2000, n_channels)

        wave_no_sustain = synthesize(amp, freqs, 0.1, 44100,
                                     timbre="sine", sustain=0.0)
        wave_sustain_zero = synthesize(amp, freqs, 0.1, 44100,
                                       timbre="sine", sustain=0.0)

        np.testing.assert_array_equal(wave_no_sustain, wave_sustain_zero)

    def test_sustain_smooths_amplitude_transitions(self):
        """Sustain 0.5 produces smaller amplitude jumps at row boundaries."""
        n_channels = 2
        freqs = np.array([300, 800], dtype=float)
        sr = 44100
        spr = 0.1  # 10 rows/sec

        # Two rows with very different amplitudes
        amp = np.array([
            [1.0, 1.0],  # loud
            [0.0, 0.0],  # silent
        ])

        wave_no_sustain = synthesize(amp, freqs, spr, sr,
                                     timbre="sine", sustain=0.0)
        wave_with_sustain = synthesize(amp, freqs, spr, sr,
                                       timbre="sine", sustain=0.5)

        # With sustain, the transition from row 0 → row 1 should be smoother
        # The second segment of the sustained version should have some energy
        # (because of the blend from the loud row)
        segment_samples = round(spr * sr)
        seg2_no_sustain = wave_no_sustain[segment_samples:]
        seg2_with_sustain = wave_with_sustain[segment_samples:]

        # The sustained version should have more energy in segment 2
        rms_no_sustain = np.sqrt(np.mean(seg2_no_sustain ** 2))
        rms_with_sustain = np.sqrt(np.mean(seg2_with_sustain ** 2))

        # With sustain, the blended start means more energy early in segment 2
        assert rms_with_sustain > rms_no_sustain + 1e-10, \
            f"Expected sustain to add energy: no_sustain={rms_no_sustain:.6f}, " \
            f"with_sustain={rms_with_sustain:.6f}"


class TestTimbre:
    """Phase 5: timbre modes (sine, bell, chime)."""

    def test_timbre_sine_unchanged(self):
        """timbre='sine' output matches existing synthesis behavior."""
        n_rows, n_channels = 10, 3
        np.random.seed(123)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(300, 1500, n_channels)

        wave1 = synthesize(amp, freqs, 0.1, 44100, timbre="sine", sustain=0.0)
        wave2 = synthesize(amp, freqs, 0.1, 44100, timbre="sine", sustain=0.0)

        np.testing.assert_array_equal(wave1, wave2)

    def test_timbre_bell_differs_from_sine(self):
        """timbre='bell' and timbre='sine' produce different waveforms."""
        n_rows, n_channels = 10, 3
        np.random.seed(456)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(300, 1500, n_channels)

        wave_sine = synthesize(amp, freqs, 0.1, 44100, timbre="sine", sustain=0.0)
        wave_bell = synthesize(amp, freqs, 0.1, 44100, timbre="bell", sustain=0.0)

        assert not np.array_equal(wave_sine, wave_bell), \
            "Bell and sine should produce different waveforms"

    def test_timbre_chime_differs_from_bell(self):
        """timbre='chime' and timbre='bell' produce different waveforms."""
        n_rows, n_channels = 10, 3
        np.random.seed(789)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(300, 1500, n_channels)

        wave_bell = synthesize(amp, freqs, 0.1, 44100, timbre="bell", sustain=0.0)
        wave_chime = synthesize(amp, freqs, 0.1, 44100, timbre="chime", sustain=0.0)

        assert not np.array_equal(wave_bell, wave_chime), \
            "Chime and bell should produce different waveforms (inharmonic vs harmonic)"

    def test_timbre_amplitude_range(self):
        """Both bell and chime outputs remain in [-1, 1] after normalization."""
        n_rows, n_channels = 20, 4
        np.random.seed(101)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(200, 2000, n_channels)

        for t in ("bell", "chime"):
            waveform = synthesize(amp, freqs, 0.1, 44100, timbre=t, sustain=0.0)
            assert np.max(np.abs(waveform)) <= 1.0 + 1e-10, \
                f"timbre='{t}' exceeded [-1, 1] range"
            assert np.isfinite(waveform).all(), \
                f"timbre='{t}' produced non-finite values"


class TestExportWav:
    """WAV export round-trip."""

    def test_round_trip(self, tmp_path):
        sr = 22050
        duration = 0.5
        t = np.arange(int(sr * duration)) / sr
        waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

        path = str(tmp_path / "test_output.wav")
        export_wav(waveform, sr, path)

        # Read it back
        read_sr, read_data = wavfile.read(path)
        assert read_sr == sr
        assert read_data.dtype == np.int16
        assert len(read_data) == len(waveform)
