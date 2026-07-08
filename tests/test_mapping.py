"""Tests for sonify.mapping."""

import numpy as np
import pytest

from sonify.mapping import scale_values, normalize_per_channel, assign_frequencies


class TestScaleValues:
    """All 3 scale modes run without error on data with zeros."""

    def test_linear(self):
        m = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = scale_values(m, "linear")
        np.testing.assert_array_equal(result, m)
        # Must be a copy
        assert result is not m

    def test_log10_with_zeros(self):
        m = np.array([[0.0, 1.0], [10.0, 100.0]])
        result = scale_values(m, "log10")
        # log10(0 + eps) ≈ -10, log10(1 + eps) ≈ 0, log10(10 + eps) ≈ 1
        assert result.shape == m.shape
        assert np.isfinite(result).all()

    def test_ln_with_zeros(self):
        m = np.array([[0.0, 5.0], [1.0, 0.0]])
        result = scale_values(m, "ln")
        assert result.shape == m.shape
        assert np.isfinite(result).all()

    def test_unknown_mode_raises(self):
        m = np.array([[1.0]])
        with pytest.raises(ValueError, match="Unknown scale mode"):
            scale_values(m, "unknown")


class TestNormalizePerChannel:
    """Per-channel normalization to [0, 1]."""

    def test_output_range(self):
        m = np.array([[1.0, 10.0], [5.0, 50.0], [3.0, 30.0]])
        result = normalize_per_channel(m)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_constant_channel_is_zero(self):
        m = np.array([[5.0, 10.0], [5.0, 20.0], [5.0, 30.0]])
        result = normalize_per_channel(m)
        # Channel 0 is constant → all zeros
        np.testing.assert_array_equal(result[:, 0], [0.0, 0.0, 0.0])

    def test_weak_channel_reaches_one(self):
        """A low-intensity channel must still reach 1.0 at its own peak.
        This is the key behavior per-channel normalization exists for.
        """
        m = np.array([
            [100.0, 0.5],   # Channel 0 bright, channel 1 dim
            [200.0, 1.0],   # Channel 1's peak
            [150.0, 0.3],
        ])
        result = normalize_per_channel(m)
        # Channel 1's max (row 1) should be 1.0
        assert result[1, 1] == 1.0
        # Channel 0's max (row 1) should also be 1.0
        assert result[1, 0] == 1.0

    def test_per_channel_independence(self):
        """Channels are normalized independently, not globally."""
        m = np.array([
            [0.0, 0.0],
            [10.0, 1000.0],
        ])
        result = normalize_per_channel(m)
        # Both channels should have 0→1 range despite vastly different scales
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(result[1], [1.0, 1.0])


class TestAssignFrequencies:
    """Frequency assignment: correct count, range, spacing."""

    def test_index_mode_count_and_range(self):
        freqs = assign_frequencies(8, 150.0, 2500.0, mode="index")
        assert len(freqs) == 8
        assert freqs[0] == pytest.approx(150.0)
        assert freqs[-1] == pytest.approx(2500.0)

    def test_index_mode_strictly_increasing(self):
        freqs = assign_frequencies(16, 150.0, 2500.0, mode="index")
        assert all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1))

    def test_index_mode_log_spaced(self):
        """Consecutive frequency ratios should be constant (log-spacing)."""
        freqs = assign_frequencies(10, 200.0, 4000.0, mode="index")
        ratios = freqs[1:] / freqs[:-1]
        np.testing.assert_array_almost_equal(ratios, ratios[0])

    def test_single_channel(self):
        freqs = assign_frequencies(1, 150.0, 2500.0, mode="index")
        assert len(freqs) == 1
        assert freqs[0] == 150.0

    def test_wavelength_mode_requires_wavelengths(self):
        with pytest.raises(ValueError, match="wavelengths array is required"):
            assign_frequencies(4, 150.0, 2500.0, mode="wavelength", wavelengths=None)

    def test_wavelength_mode_length_mismatch_raises(self):
        wl = np.array([275.0, 300.0, 350.0])
        with pytest.raises(ValueError, match="must match n_channels"):
            assign_frequencies(5, 150.0, 2500.0, mode="wavelength", wavelengths=wl)

    def test_wavelength_mode_within_range(self):
        wl = np.linspace(275, 446, 8)
        freqs = assign_frequencies(8, 150.0, 2500.0, mode="wavelength", wavelengths=wl)
        assert len(freqs) == 8
        assert freqs.min() >= 150.0 - 0.01
        assert freqs.max() <= 2500.0 + 0.01


class TestMapToneFromColumn:
    """Phase 3: column-driven frequency mapping."""

    def test_map_tone_from_column(self):
        """Output length matches input, all values within [min_freq, max_freq]."""
        from sonify.mapping import map_tone_from_column

        column_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        min_freq, max_freq = 200.0, 4000.0

        freqs = map_tone_from_column(column_values, min_freq, max_freq)

        assert len(freqs) == len(column_values)
        assert freqs.min() >= min_freq - 0.01
        assert freqs.max() <= max_freq + 0.01
        # Should be monotonically increasing (input is monotonically increasing)
        assert all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1))


class TestApplyIntensityColumn:
    """Phase 3: column-driven intensity modulation."""

    def test_apply_intensity_column(self):
        """Zero column value produces silent row; nonzero scales correctly."""
        from sonify.mapping import apply_intensity_column

        amplitude_matrix = np.array([
            [0.5, 0.8],
            [0.3, 0.6],
            [1.0, 1.0],
        ])
        # Column values: first row min (becomes 0), last row max (becomes 1)
        column_values = np.array([0.0, 5.0, 10.0])

        result = apply_intensity_column(amplitude_matrix, column_values)

        assert result.shape == amplitude_matrix.shape
        # First row (column=0, normalized to 0) → all zeros
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
        # Last row (column=10, normalized to 1) → unchanged
        np.testing.assert_array_almost_equal(result[2], [1.0, 1.0])
        # Middle row (column=5, normalized to 0.5) → scaled by 0.5
        np.testing.assert_array_almost_equal(result[1], [0.15, 0.3])
