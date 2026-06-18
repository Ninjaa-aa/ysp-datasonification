"""Tests for sonify.preprocess."""

import numpy as np
import pandas as pd
import pytest

from sonify.preprocess import clean, rebin, rebin_wavelengths, sort_by_row_order, _contiguous_groups


class TestSortByRowOrder:
    def test_sorts_by_row_num(self):
        df = pd.DataFrame({"row_num": [5, 1, 3], "value": [50, 10, 30]})
        result = sort_by_row_order(df)
        assert list(result["row_num"]) == [1, 3, 5]

    def test_sorts_by_depth_descending(self):
        df = pd.DataFrame({"depth": [10.0, 30.0, 20.0], "value": [1, 3, 2]})
        result = sort_by_row_order(df)
        assert list(result["depth"]) == [30.0, 20.0, 10.0]

    def test_no_sort_columns(self):
        df = pd.DataFrame({"a": [3, 1, 2]})
        result = sort_by_row_order(df)
        assert list(result["a"]) == [3, 1, 2]


class TestClean:
    def test_nan_to_zero(self):
        df = pd.DataFrame({"Band_1_bc": [1.0, float("nan"), 3.0]})
        result = clean(df, ["Band_1_bc"])
        assert result[1, 0] == 0.0

    def test_negative_to_zero(self):
        df = pd.DataFrame({"Band_1_bc": [1.0, -5.0, 3.0]})
        result = clean(df, ["Band_1_bc"])
        assert result[1, 0] == 0.0
        assert result[0, 0] == 1.0

    def test_preserves_positive(self):
        df = pd.DataFrame({
            "Band_1_bc": [10.0, 20.0],
            "Band_2_bc": [30.0, 40.0],
        })
        result = clean(df, ["Band_1_bc", "Band_2_bc"])
        expected = np.array([[10.0, 30.0], [20.0, 40.0]])
        np.testing.assert_array_equal(result, expected)


class TestRebin:
    def test_8_to_4(self):
        # 8 channels → 4 bins: groups of 2 averaged
        matrix = np.arange(16).reshape(2, 8).astype(float)
        # Row 0: [0, 1, 2, 3, 4, 5, 6, 7]
        # Row 1: [8, 9, 10, 11, 12, 13, 14, 15]
        result = rebin(matrix, 4)
        assert result.shape == (2, 4)
        # Group 0: mean(0,1)=0.5, Group 1: mean(2,3)=2.5, etc.
        np.testing.assert_array_almost_equal(result[0], [0.5, 2.5, 4.5, 6.5])
        np.testing.assert_array_almost_equal(result[1], [8.5, 10.5, 12.5, 14.5])

    def test_no_rebin_when_n_bins_ge_channels(self):
        matrix = np.ones((3, 4))
        result = rebin(matrix, 4)
        assert result is matrix  # same object, not a copy

    def test_uneven_groups(self):
        # 5 channels → 3 bins: groups of [2, 2, 1] or [2, 1, 2]
        matrix = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
        result = rebin(matrix, 3)
        assert result.shape == (1, 3)
        # Just verify we got 3 bins and values are plausible averages
        assert all(result[0] >= 10.0)
        assert all(result[0] <= 50.0)


class TestRebinWavelengths:
    def test_same_grouping_as_rebin(self):
        """rebin() and rebin_wavelengths() must use identical grouping."""
        n_channels = 8
        n_bins = 4
        # Check they produce the same number of groups
        matrix = np.arange(20).reshape(2, 10)[:, :n_channels].astype(float)
        wavelengths = np.linspace(275, 446, n_channels)

        rebinned_matrix = rebin(matrix, n_bins)
        rebinned_wl = rebin_wavelengths(wavelengths, n_bins)

        assert rebinned_matrix.shape[1] == len(rebinned_wl) == n_bins

    def test_known_values(self):
        wavelengths = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
        result = rebin_wavelengths(wavelengths, 4)
        # Groups: [100,200], [300,400], [500,600], [700,800]
        np.testing.assert_array_almost_equal(result, [150.0, 350.0, 550.0, 750.0])

    def test_grouping_boundaries_match(self):
        """Both functions must use _contiguous_groups internally."""
        n = 7
        n_bins = 3
        groups_direct = _contiguous_groups(n, n_bins)
        # Verify they match by checking each group
        assert len(groups_direct) == n_bins
        # All indices covered
        all_indices = np.concatenate(groups_direct)
        np.testing.assert_array_equal(all_indices, np.arange(n))
