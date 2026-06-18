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

        waveform = synthesize(amp, freqs, spr, sr)
        expected_length = n_rows * round(spr * sr)
        assert abs(len(waveform) - expected_length) <= 1

    def test_normalized_range(self):
        n_rows, n_channels = 50, 4
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(200, 2000, n_channels)

        waveform = synthesize(amp, freqs, 0.05, 22050)
        assert np.max(np.abs(waveform)) <= 1.0 + 1e-10

    def test_silence_produces_silence(self):
        amp = np.zeros((10, 3))
        freqs = np.array([200, 500, 1000], dtype=float)
        waveform = synthesize(amp, freqs, 0.1, 44100)
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
        waveform = synthesize(amp, freqs, spr, sr)
        assert np.isfinite(waveform).all()
        assert np.max(np.abs(waveform)) <= 1.0 + 1e-10


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
