"""Tests for sonify.config validation.

``SonificationConfig.validate()`` is the single gate between CLI input and the
pipeline, so every rule it enforces should fail loudly when broken.
"""

import pytest

from sonify.config import SonificationConfig, ParameterMap


def cfg(**kw):
    """A valid config, overridden by kwargs."""
    return SonificationConfig(input_path="data.csv", **kw)


class TestValidBaseline:
    def test_minimal_config_validates(self):
        cfg().validate()  # must not raise

    def test_missing_input_path_raises(self):
        with pytest.raises(ValueError, match="input_path"):
            SonificationConfig().validate()


class TestTriggerValidation:
    """Trigger settings — Dr. Malaska's two-function design."""

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be >= 0"):
            cfg(threshold=-1.0).validate()

    def test_zero_threshold_is_valid(self):
        cfg(threshold=0.0).validate()

    def test_bad_trigger_type_raises(self):
        with pytest.raises(ValueError, match="trigger_type"):
            cfg(trigger_type="quadratic").validate()

    @pytest.mark.parametrize("t", ["linear", "log"])
    def test_valid_trigger_types(self, t):
        cfg(trigger_type=t).validate()

    def test_zero_target_tones_raises(self):
        with pytest.raises(ValueError, match="target_tones"):
            cfg(target_tones=0).validate()

    def test_none_target_tones_is_valid(self):
        cfg(target_tones=None).validate()


class TestReverbTailValidation:
    def test_negative_tail_raises(self):
        with pytest.raises(ValueError, match="reverb_tail_ms"):
            cfg(reverb_tail_ms=-1.0).validate()

    def test_zero_and_positive_tails_are_valid(self):
        cfg(reverb_tail_ms=0.0).validate()
        cfg(reverb_tail_ms=1200.0).validate()


class TestFrequencyValidation:
    def test_non_positive_min_freq_raises(self):
        with pytest.raises(ValueError, match="min_freq"):
            cfg(min_freq=0.0).validate()

    def test_max_below_min_raises(self):
        with pytest.raises(ValueError, match="max_freq"):
            cfg(min_freq=1000.0, max_freq=500.0).validate()

    def test_bad_freq_mode_raises(self):
        with pytest.raises(ValueError, match="freq_mode"):
            cfg(freq_mode="linear").validate()

    def test_pentatonic_octaves_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pentatonic_octaves"):
            cfg(pentatonic_octaves=0).validate()
        with pytest.raises(ValueError, match="pentatonic_octaves"):
            cfg(pentatonic_octaves=9).validate()


class TestParameterMapValidation:
    def test_lambda_max_is_accepted(self):
        """Phase 3's stated default tone source."""
        cfg(param_map=ParameterMap(tone_source="lambda_max")).validate()

    def test_bad_tone_source_raises(self):
        with pytest.raises(ValueError, match="tone_source"):
            cfg(param_map=ParameterMap(tone_source="colour")).validate()

    def test_column_source_without_column_raises(self):
        with pytest.raises(ValueError, match="tone_column"):
            cfg(param_map=ParameterMap(tone_source="column")).validate()

    def test_intensity_column_source_without_column_raises(self):
        with pytest.raises(ValueError, match="intensity_column"):
            cfg(param_map=ParameterMap(intensity_source="column")).validate()


class TestOtherRanges:
    def test_volume_out_of_range_raises(self):
        with pytest.raises(ValueError, match="volume"):
            cfg(volume=1.5).validate()

    def test_non_positive_playback_speed_raises(self):
        with pytest.raises(ValueError, match="playback_speed"):
            cfg(playback_speed=0.0).validate()

    def test_bad_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            cfg(scale="sqrt").validate()

    def test_bad_gain_mode_raises(self):
        with pytest.raises(ValueError, match="gain_mode"):
            cfg(gain_mode="max").validate()

    def test_smoothing_out_of_range_raises(self):
        with pytest.raises(ValueError, match="smoothing"):
            cfg(smoothing=2.0).validate()

    def test_trail_rows_out_of_range_raises(self):
        with pytest.raises(ValueError, match="trail_rows"):
            cfg(trail_rows=0).validate()
        with pytest.raises(ValueError, match="trail_rows"):
            cfg(trail_rows=21).validate()

    def test_row_end_before_row_start_raises(self):
        with pytest.raises(ValueError, match="row_end"):
            cfg(row_start=100, row_end=50).validate()

    def test_bad_marker_shape_raises(self):
        with pytest.raises(ValueError, match="marker_shape"):
            cfg(marker_shape="triangle").validate()

    def test_video_output_bad_extension_raises(self):
        with pytest.raises(ValueError, match="video_output"):
            cfg(video_output="out.mov").validate()
