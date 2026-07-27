"""Tests for sonify.events — Dr. Malaska's trigger function."""

import os

import numpy as np
import pandas as pd
import pytest

from sonify.events import (
    MALASKA_THRESHOLD_TABLE,
    apply_trigger,
    count_signals,
    find_event_clusters,
    row_trigger_mask,
    summarize_events,
    threshold_for_target_tones,
)
from sonify.preprocess import clean


_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw",
    "2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv",
)
_BANDS = [f"Band_{i}_bc" for i in range(1, 33)]


class TestRowTriggerMask:
    def test_linear_gates_on_any_band(self):
        m = np.array([[0.0, 5.0], [100.0, 0.0], [1.0, 2.0]])
        np.testing.assert_array_equal(
            row_trigger_mask(m, 50.0, "linear"), [False, True, False]
        )

    def test_zero_threshold_passes_any_signal(self):
        m = np.array([[0.0, 0.0], [0.0, 0.1]])
        np.testing.assert_array_equal(
            row_trigger_mask(m, 0.0), [False, True]
        )

    def test_log_and_linear_agree_on_positive_data(self):
        rng = np.random.default_rng(0)
        m = rng.random((50, 4)) * 1000
        np.testing.assert_array_equal(
            row_trigger_mask(m, 500.0, "linear"),
            row_trigger_mask(m, 500.0, "log"),
        )

    def test_invalid_trigger_type_raises(self):
        with pytest.raises(ValueError, match="trigger_type"):
            row_trigger_mask(np.ones((2, 2)), 1.0, "bogus")


class TestApplyTrigger:
    def test_silences_subthreshold_rows_entirely(self):
        m = np.array([[1.0, 2.0], [900.0, 3.0]])
        out = apply_trigger(m, 500.0)
        np.testing.assert_array_equal(out, [[0.0, 0.0], [900.0, 3.0]])

    def test_preserves_full_spectral_shape_of_triggered_rows(self):
        """A triggered row keeps its weak bands — the peak's shape matters."""
        m = np.array([[900.0, 1.0, 0.5]])
        np.testing.assert_array_equal(apply_trigger(m, 500.0), m)

    def test_zero_threshold_is_identity(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = apply_trigger(m, 0.0)
        np.testing.assert_array_equal(out, m)
        assert out is not m

    def test_does_not_mutate_input(self):
        m = np.array([[1.0, 2.0]])
        apply_trigger(m, 500.0)
        np.testing.assert_array_equal(m, [[1.0, 2.0]])


class TestFindEventClusters:
    def test_groups_contiguous_runs(self):
        mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
        assert find_event_clusters(mask) == [(1, 3), (5, 6)]

    def test_run_at_both_edges(self):
        mask = np.array([1, 1, 0, 1], dtype=bool)
        assert find_event_clusters(mask) == [(0, 2), (3, 4)]

    def test_all_false_and_empty(self):
        assert find_event_clusters(np.zeros(5, dtype=bool)) == []
        assert find_event_clusters(np.array([], dtype=bool)) == []

    def test_all_true_is_one_cluster(self):
        assert find_event_clusters(np.ones(4, dtype=bool)) == [(0, 4)]


class TestSummarizeEvents:
    def test_reports_peak_band_and_value(self):
        m = np.array([[0.0, 0.0], [10.0, 900.0], [0.0, 0.0]])
        mask = np.array([False, True, False])
        ev = summarize_events(m, mask)
        assert len(ev) == 1
        assert ev[0]["peak_row"] == 1
        assert ev[0]["peak_band"] == 1
        assert ev[0]["peak_value"] == 900.0


class TestThresholdForTargetTones:
    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError, match="target_tones"):
            threshold_for_target_tones(np.ones((3, 3)), 0)

    def test_higher_target_gives_lower_threshold(self):
        rng = np.random.default_rng(1)
        m = rng.random((500, 8)) * 1000
        assert (threshold_for_target_tones(m, 100)
                < threshold_for_target_tones(m, 10))


@pytest.mark.skipif(not os.path.isfile(_DATA), reason="borehole dataset not present")
class TestMalaskaThresholdTable:
    """The trigger must reproduce Dr. Malaska's 2026-07-09 analysis exactly.

    If this fails, either the trigger definition drifted or the dataset changed.
    """

    @staticmethod
    def _matrix():
        return clean(pd.read_csv(_DATA), _BANDS)

    @pytest.mark.parametrize("threshold,expected", sorted(MALASKA_THRESHOLD_TABLE.items()))
    def test_signal_counts_match_spreadsheet(self, threshold, expected):
        assert count_signals(self._matrix(), threshold) == expected

    def test_first_signal_depth_matches(self):
        """His table puts the first real signal at 89.435 m for thresholds 600-1000."""
        df = pd.read_csv(_DATA)
        mask = row_trigger_mask(clean(df, _BANDS), 600.0)
        first_depth = df.depth.values[np.argmax(mask)]
        assert first_depth == pytest.approx(89.435, abs=0.001)

    def test_target_tones_lands_in_his_working_range(self):
        """Asking for ~25 tones should pick a threshold near his 600 row."""
        m = self._matrix()
        thr = threshold_for_target_tones(m, 25)
        assert count_signals(m, thr) == pytest.approx(25, abs=3)
        assert 400 <= thr <= 900
