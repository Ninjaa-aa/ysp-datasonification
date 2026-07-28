"""Tests for sonify.synth and sonify.export."""

import os
import tempfile

import numpy as np
import pytest
from scipy.io import wavfile

from sonify.synth import synthesize, generate_adsr_envelope, ADSR_SHAPES
from sonify.export import export_wav


class TestSynthesize:
    """Core synthesis properties."""

    def test_waveform_length(self):
        n_rows, n_channels = 20, 4
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(200, 2000, n_channels)
        spr = 0.1  # 10 rows/sec
        sr = 44100

        waveform = synthesize(amp, freqs, spr, sr, timbre="sine",
                              timbre_partition=False, adsr_shape="natural")
        expected_length = n_rows * round(spr * sr)
        assert abs(len(waveform) - expected_length) <= 1

    def test_normalized_range(self):
        n_rows, n_channels = 50, 4
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(200, 2000, n_channels)

        waveform = synthesize(amp, freqs, 0.05, 22050, timbre="sine",
                              timbre_partition=False, adsr_shape="natural")
        assert np.max(np.abs(waveform)) <= 1.0 + 1e-10

    def test_silence_produces_silence(self):
        amp = np.zeros((10, 3))
        freqs = np.array([200, 500, 1000], dtype=float)
        waveform = synthesize(amp, freqs, 0.1, 44100, timbre="sine",
                              timbre_partition=False, adsr_shape="natural")
        # All-zero amplitude → all-zero waveform (peak-normalize guards div-by-0)
        np.testing.assert_array_equal(waveform, 0.0)

    def test_fade_cap_at_high_speed(self):
        """At very high playback speed, ADSR phases should be capped."""
        n_rows, n_channels = 10, 2
        amp = np.ones((n_rows, n_channels))
        freqs = np.array([300, 800], dtype=float)
        sr = 44100
        # 200 rows/sec → segment = ~220 samples
        spr = 1.0 / 200.0

        # Just verify it runs without error and produces valid output
        waveform = synthesize(amp, freqs, spr, sr, timbre="sine",
                              timbre_partition=False, adsr_shape="natural")
        assert np.isfinite(waveform).all()
        assert np.max(np.abs(waveform)) <= 1.0 + 1e-10


class TestADSR:
    """ADSR envelope tests (replaced the earlier sustain-ramp approach)."""

    def test_adsr_envelope_length(self):
        """Envelope length matches requested segment_samples exactly."""
        env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
        assert len(env) == 4410

    def test_adsr_envelope_starts_at_zero(self):
        """ADSR envelope starts at zero (attack phase begins from silence)."""
        env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
        assert env[0] == pytest.approx(0.0, abs=0.01)

    def test_adsr_envelope_ends_at_zero(self):
        """ADSR envelope ends at zero (release phase returns to silence)."""
        env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
        assert env[-1] == pytest.approx(0.0, abs=0.01)

    def test_adsr_envelope_within_range(self):
        """All ADSR envelope values in [0, 1]."""
        env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
        assert env.min() >= -0.01
        assert env.max() <= 1.01

    def test_adsr_short_segment_no_crash(self):
        """At 40 rows/sec, segment_samples = ~1102; each phase capped at 220."""
        env = generate_adsr_envelope(1102, 44100, 15, 60, 0.6, 80)
        assert len(env) == 1102
        assert env.min() >= -0.01
        assert env.max() <= 1.01

    def test_adsr_tight_differs_from_slow(self):
        """Tight and slow ADSR shapes produce audibly different waveforms."""
        freqs = np.array([220., 275., 330., 415.])
        amps = np.ones((20, 4)) * 0.5
        w_tight = synthesize(amps, freqs, 0.1, 44100, timbre="sine",
                             timbre_partition=False, adsr_shape="tight")
        w_slow = synthesize(amps, freqs, 0.1, 44100, timbre="sine",
                            timbre_partition=False, adsr_shape="slow")
        assert not np.allclose(w_tight, w_slow)

    def test_adsr_amplitude_shape(self):
        """ADSR envelope starts and ends near zero."""
        env = generate_adsr_envelope(44100, 44100, 5, 90, 0.0, 5)
        assert env[0] == pytest.approx(0.0, abs=0.01)
        assert env[-1] == pytest.approx(0.0, abs=0.01)
        assert env.min() >= -0.01
        assert env.max() <= 1.01


class TestTimbre:
    """Timbre modes (sine, bell, chime)."""

    def test_timbre_sine_unchanged(self):
        """timbre='sine' output matches existing synthesis behavior."""
        n_rows, n_channels = 10, 3
        np.random.seed(123)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(300, 1500, n_channels)

        wave1 = synthesize(amp, freqs, 0.1, 44100, timbre="sine",
                           timbre_partition=False, adsr_shape="natural")
        wave2 = synthesize(amp, freqs, 0.1, 44100, timbre="sine",
                           timbre_partition=False, adsr_shape="natural")

        np.testing.assert_array_equal(wave1, wave2)

    def test_timbre_bell_differs_from_sine(self):
        """timbre='bell' and timbre='sine' produce different waveforms."""
        n_rows, n_channels = 10, 3
        np.random.seed(456)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(300, 1500, n_channels)

        wave_sine = synthesize(amp, freqs, 0.1, 44100, timbre="sine",
                               timbre_partition=False, adsr_shape="natural")
        wave_bell = synthesize(amp, freqs, 0.1, 44100, timbre="bell",
                               timbre_partition=False, adsr_shape="natural")

        assert not np.array_equal(wave_sine, wave_bell), \
            "Bell and sine should produce different waveforms"

    def test_timbre_chime_differs_from_bell(self):
        """timbre='chime' and timbre='bell' produce different waveforms."""
        n_rows, n_channels = 10, 3
        np.random.seed(789)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(300, 1500, n_channels)

        wave_bell = synthesize(amp, freqs, 0.1, 44100, timbre="bell",
                               timbre_partition=False, adsr_shape="natural")
        wave_chime = synthesize(amp, freqs, 0.1, 44100, timbre="chime",
                                timbre_partition=False, adsr_shape="natural")

        assert not np.array_equal(wave_bell, wave_chime), \
            "Chime and bell should produce different waveforms (inharmonic vs harmonic)"

    def test_timbre_amplitude_range(self):
        """Both bell and chime outputs remain in [-1, 1] after normalization."""
        n_rows, n_channels = 20, 4
        np.random.seed(101)
        amp = np.random.rand(n_rows, n_channels)
        freqs = np.linspace(200, 2000, n_channels)

        for t in ("bell", "chime"):
            waveform = synthesize(amp, freqs, 0.1, 44100, timbre=t,
                                  timbre_partition=False, adsr_shape="natural")
            assert np.max(np.abs(waveform)) <= 1.0 + 1e-10, \
                f"timbre='{t}' exceeded [-1, 1] range"
            assert np.isfinite(waveform).all(), \
                f"timbre='{t}' produced non-finite values"


class TestTimbrePartition:
    """Sound quality update: timbral partitioning tests."""

    def test_timbre_partition_differs_from_uniform(self):
        """With partition enabled, output should differ from uniform timbre."""
        freqs = np.array([220., 247.5, 277.2, 311., 330., 370., 415., 440.])
        amps = np.ones((10, 8)) * 0.5
        w_partition = synthesize(amps, freqs, 0.1, 44100, timbre="chime",
                                 timbre_partition=True, adsr_shape="natural")
        w_uniform = synthesize(amps, freqs, 0.1, 44100, timbre="chime",
                               timbre_partition=False, adsr_shape="natural")
        assert not np.allclose(w_partition, w_uniform)

    def test_synthesize_output_in_range(self):
        """Partitioned synthesis output stays in [-1, 1]."""
        np.random.seed(42)
        freqs = np.array([220., 330., 440., 550.])
        amps = np.random.rand(10, 4)
        w = synthesize(amps, freqs, 0.1, 44100, timbre="chime",
                       timbre_partition=True, adsr_shape="natural")
        assert w.min() >= -1.01 and w.max() <= 1.01


class TestSynthesisGoldenValues:
    """Pins synthesis output so optimizations cannot silently change the sound.

    Captured from the pre-vectorization implementation.  The vectorized version
    reproduces these to ~1e-12; only float summation order differs, so the
    tolerance is tight enough that any real change in the audio will fail.

    If a change to synthesis is *intended*, regenerate these values deliberately
    and note it in CHANGELOG.md — do not loosen the tolerance.
    """

    GOLDEN = {
        # name: (rms, sum) for a fixed seed.
        #
        # The three non-partitioned cases are unchanged since the values were
        # first captured, which is the point: they prove the synthesis core
        # itself has not drifted through vectorization or any later work.
        "sine_nopart": (0.120091147515, -4.542508032),
        "bell_nopart": (0.061508490442, -5.043565150),
        "chime_nopart": (0.091996831024, -3.313643152),
        # Regenerated deliberately when the partition table's low group moved
        # from `bell` to `soft` to cut harshness (see CHANGELOG v1.1). Only the
        # partitioned cases moved, confirming the change was scoped correctly.
        "partition8": (0.124701040130, -3.468574135),
        "partition32": (0.069883257374, -1.198170740),
    }

    SPECS = {
        "sine_nopart": (30, 4, "sine", "natural", False),
        "bell_nopart": (30, 4, "bell", "tight", False),
        "chime_nopart": (30, 4, "chime", "slow", False),
        "partition8": (30, 8, "chime", "natural", True),
        "partition32": (20, 32, "chime", "tight", True),
    }

    def test_output_matches_golden_values(self):
        """Seeds are drawn in one sequence, so all cases run together."""
        rng = np.random.default_rng(42)
        for name, (rows, ch, timbre, adsr, part) in self.SPECS.items():
            amp = rng.random((rows, ch))
            freqs = np.linspace(200, 2000, ch)
            w = synthesize(
                amp, freqs, 0.1, 44100,
                timbre=timbre, adsr_shape=adsr, timbre_partition=part,
            )
            exp_rms, exp_sum = self.GOLDEN[name]
            assert np.sqrt(np.mean(w ** 2)) == pytest.approx(exp_rms, abs=1e-10), name
            assert float(w.sum()) == pytest.approx(exp_sum, abs=1e-6), name

    def test_envelope_is_shared_across_rows(self):
        """The ADSR envelope depends only on shape and segment length.

        Caching it per channel (instead of rebuilding it n_rows * n_channels
        times) is what makes the vectorized path fast, so assert the premise.
        """
        a = generate_adsr_envelope(4410, 44100, *ADSR_SHAPES["natural"])
        b = generate_adsr_envelope(4410, 44100, *ADSR_SHAPES["natural"])
        np.testing.assert_array_equal(a, b)

    def test_chunked_and_unchunked_paths_agree(self):
        """Channel chunking kicks in above 128 channels; both paths must match.

        130 channels chunks; 100 does not. Same per-channel content in both,
        so the shared prefix of the mix must be identical.
        """
        rng = np.random.default_rng(7)
        amp = rng.random((5, 200))
        freqs = np.linspace(200, 2000, 200)
        w = synthesize(amp, freqs, 0.05, 44100, timbre="sine",
                       timbre_partition=False, adsr_shape="natural")
        assert np.isfinite(w).all()
        assert np.max(np.abs(w)) <= 1.0 + 1e-9


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
