"""Tests for sonify.band_detect."""

import pandas as pd
import pytest

from sonify.band_detect import detect_band_columns


def _make_df(columns: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame with the given column names."""
    return pd.DataFrame({c: [0.0] for c in columns})


class TestDetectBandColumns:
    """Band detection: finds the right columns, excludes noise columns."""

    def test_band_bc_pattern(self):
        cols = [
            "row_num", "depth",
            "Band_1_bc", "Band_2_bc", "Band_3_bc", "Band_4_bc",
            "Band_1_std1_MAX_SDT", "Band_2_std1_MAX_SDT",
            "row_bc", "depth_bc",
        ]
        df = _make_df(cols)
        names, indices = detect_band_columns(df)

        assert names == ["Band_1_bc", "Band_2_bc", "Band_3_bc", "Band_4_bc"]
        assert indices == [1, 2, 3, 4]

    def test_excludes_std_sdt_max_columns(self):
        cols = [
            "Band_1_bc", "Band_2_bc",
            "Band_1_std1_MAX_SDT", "Band_2_std1_MAX_SDT",
            "Band_3_std", "Band_4_max",
        ]
        df = _make_df(cols)
        names, indices = detect_band_columns(df)

        # Only the _bc columns should survive
        assert names == ["Band_1_bc", "Band_2_bc"]
        assert indices == [1, 2]

    def test_channel_pattern(self):
        cols = ["Channel_1", "Channel_2", "Channel_10", "something_else"]
        df = _make_df(cols)
        names, indices = detect_band_columns(df)

        assert names == ["Channel_1", "Channel_2", "Channel_10"]
        assert indices == [1, 2, 10]

    def test_band_no_suffix_pattern(self):
        cols = ["Band_5", "Band_10", "Band_15"]
        df = _make_df(cols)
        names, indices = detect_band_columns(df)

        assert names == ["Band_5", "Band_10", "Band_15"]
        assert indices == [5, 10, 15]

    def test_sorted_by_index(self):
        cols = ["Band_10_bc", "Band_2_bc", "Band_1_bc", "Band_5_bc"]
        df = _make_df(cols)
        names, indices = detect_band_columns(df)

        assert indices == [1, 2, 5, 10]
        assert names == ["Band_1_bc", "Band_2_bc", "Band_5_bc", "Band_10_bc"]

    def test_raises_when_no_bands(self):
        cols = ["row_num", "depth", "timestamp", "some_other_col"]
        df = _make_df(cols)

        with pytest.raises(ValueError, match="No band/channel columns detected"):
            detect_band_columns(df)

    def test_band_index_mapping_correct(self):
        """Band_3_bc should map to index 3, not 1 or 0."""
        cols = ["Band_3_bc", "Band_7_bc", "Band_15_bc"]
        df = _make_df(cols)
        names, indices = detect_band_columns(df)

        assert dict(zip(names, indices)) == {
            "Band_3_bc": 3,
            "Band_7_bc": 7,
            "Band_15_bc": 15,
        }
