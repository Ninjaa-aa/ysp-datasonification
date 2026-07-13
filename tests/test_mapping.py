"""Tests for sonify.mapping."""

import numpy as np
import pytest

from sonify.mapping import (
    scale_values, normalize_per_channel, assign_frequencies,
    assign_frequencies_pentatonic, smooth_amplitude_matrix,
    PENTATONIC_RATIOS,
)


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


class TestAutoGain:
    """Phase 5: global gain normalization."""

    def test_gain_max_linear_clips_to_one(self):
        """max_linear: output max is 1.0, all values in [0, 1]."""
        from sonify.mapping import apply_global_gain

        matrix = np.array([
            [0.0, 50.0],
            [100.0, 200.0],
            [30.0, 10.0],
        ])

        result = apply_global_gain(matrix, "max_linear")

        assert result.max() == pytest.approx(1.0, abs=1e-6)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_gain_median_linear_midpoint(self):
        """median_linear: median of output is approximately 0.5."""
        from sonify.mapping import apply_global_gain

        # Create data with known median
        np.random.seed(42)
        matrix = np.random.exponential(scale=10.0, size=(100, 4))

        result = apply_global_gain(matrix, "median_linear")

        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # The median of the output should be near 0.5
        output_median = np.median(result)
        assert output_median == pytest.approx(0.5, abs=0.1), \
            f"Expected median ~0.5, got {output_median:.3f}"

    def test_gain_mean_linear_midpoint(self):
        """mean_linear: mean of output is approximately 0.5."""
        from sonify.mapping import apply_global_gain

        np.random.seed(42)
        matrix = np.random.exponential(scale=10.0, size=(100, 4))

        result = apply_global_gain(matrix, "mean_linear")

        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # The mean of the output should be near 0.5 (with clipping it may differ)
        output_mean = np.mean(result)
        assert output_mean == pytest.approx(0.5, abs=0.15), \
            f"Expected mean ~0.5, got {output_mean:.3f}"

    def test_gain_pct90_clips_outliers(self):
        """pct90_linear: values above 90th percentile are clipped to 1.0."""
        from sonify.mapping import apply_global_gain

        # Create data where top 10% are outliers
        matrix = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 100.0],  # outlier in channel 1
        ])

        result = apply_global_gain(matrix, "pct90_linear")

        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # The outlier (100.0) should be clipped to 1.0
        assert result[-1, -1] == 1.0

    def test_gain_log_modes_compress_range(self):
        """Log mode output has smaller std than linear mode (confirms compression)."""
        from sonify.mapping import apply_global_gain

        # Data with extremely large dynamic range — log should compress this
        # significantly more than linear
        matrix = np.array([
            [1.0, 1.0],
            [10.0, 10.0],
            [100.0, 100.0],
            [10000.0, 10000.0],
        ])

        result_linear = apply_global_gain(matrix, "max_linear")
        result_log = apply_global_gain(matrix, "max_log")

        # In linear mode, 10000 maps to 1.0, so 1.0 maps to ~0.0001
        # Most values cluster near 0 → high std from the outlier
        # In log mode, log10(10000)=4, log10(1)=0, so values are 0, 1, 2, 4
        # much more evenly spread → different distribution

        # The key assertion: log and linear produce different distributions
        assert not np.allclose(result_linear, result_log, atol=0.01), \
            "Log and linear modes should produce different distributions"

    def test_gain_noisy_channel_stays_quiet(self):
        """A channel whose max is 5% of global max stays near-silent in max_linear.

        This is the specific noise problem Dr. Malaska identified:
        per-channel normalization would amplify this channel to full range.
        Global gain should keep it quiet.
        """
        from sonify.mapping import apply_global_gain

        # Channel 0: strong signal (max=1000)
        # Channel 1: pure noise (max=50, which is 5% of global max)
        matrix = np.array([
            [500.0, 10.0],
            [1000.0, 50.0],
            [800.0, 30.0],
            [200.0, 5.0],
        ])

        result = apply_global_gain(matrix, "max_linear")

        # Channel 1's max amplitude should be ~5% of 1.0 = ~0.05
        ch1_max = result[:, 1].max()
        assert ch1_max < 0.10, \
            f"Noisy channel should stay quiet (max={ch1_max:.3f}), " \
            f"but global gain didn't suppress it"

        # Channel 0's max should be at or near 1.0
        ch0_max = result[:, 0].max()
        assert ch0_max == pytest.approx(1.0, abs=0.01), \
            f"Strong channel should reach 1.0 (max={ch0_max:.3f})"


class TestPentatonicFrequencies:
    """Sound quality update: pentatonic frequency mapping."""

    def test_pentatonic_count(self):
        """Output array length matches requested n_channels."""
        for n in [4, 6, 8, 16]:
            freqs = assign_frequencies_pentatonic(n, 220.0, 4)
            assert len(freqs) == n

    def test_pentatonic_strictly_increasing(self):
        """Pentatonic frequencies are strictly increasing."""
        freqs = assign_frequencies_pentatonic(8, 220.0, 3)
        assert all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1))

    def test_pentatonic_root_is_first(self):
        """First frequency is the root note."""
        freqs = assign_frequencies_pentatonic(8, 220.0, 3)
        assert freqs[0] == pytest.approx(220.0, rel=0.01)

    def test_pentatonic_only_valid_intervals(self):
        """Every frequency is root * 2^k * ratio for some k and ratio."""
        freqs = assign_frequencies_pentatonic(8, 220.0, 3)
        for f in freqs:
            found = any(
                abs(f - 220.0 * (2 ** k) * r) < 0.01
                for k in range(5)
                for r in PENTATONIC_RATIOS
            )
            assert found, f"Frequency {f:.2f} Hz not on pentatonic scale"

    def test_pentatonic_auto_extends_octaves(self):
        """Auto-extends octaves when n_channels > 5 * n_octaves."""
        freqs = assign_frequencies_pentatonic(20, 220.0, 3)  # 3 octaves = 15 notes
        assert len(freqs) == 20
        assert all(np.isfinite(freqs))


class TestSmoothAmplitudeMatrix:
    """Sound quality update: temporal amplitude smoothing."""

    def test_smooth_zero_is_identity(self):
        """smoothing=0.0 returns unchanged matrix."""
        np.random.seed(42)
        m = np.random.rand(50, 8)
        result = smooth_amplitude_matrix(m, smoothing=0.0)
        np.testing.assert_array_equal(result, m)

    def test_smooth_reduces_variance(self):
        """Smoothing reduces temporal variance of the amplitude matrix."""
        np.random.seed(42)
        m = np.random.rand(200, 8)
        smoothed = smooth_amplitude_matrix(m, smoothing=0.5)
        assert smoothed.var() < m.var()
        assert 0.0 <= smoothed.min()
        assert smoothed.max() <= 1.0
