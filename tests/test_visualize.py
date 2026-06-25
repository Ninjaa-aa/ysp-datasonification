"""Tests for sonify.visualize (Phase 2 visual rendering engine)."""

import numpy as np
import pytest


class TestApplyVisualScale:
    """Visual scaling modes all run and produce valid output."""

    def test_visual_scale_modes(self):
        """All three visual scale modes run without error on data with zeros."""
        matrix = np.array([
            [0.0, 1.0, 5.0],
            [10.0, 0.0, 3.0],
            [2.0, 8.0, 0.0],
        ])

        from sonify.visualize import apply_visual_scale

        for mode in ("linear", "log10", "ln"):
            result = apply_visual_scale(matrix, mode)
            assert result.shape == matrix.shape, f"Shape mismatch for mode '{mode}'"
            assert result.min() >= 0.0, f"Negative values for mode '{mode}'"
            assert result.max() <= 1.0, f"Values > 1 for mode '{mode}'"
            assert np.isfinite(result).all(), f"Non-finite values for mode '{mode}'"


class TestRenderFrame:
    """Frame rendering produces correct shape and meaningful content."""

    @pytest.fixture(autouse=True)
    def _set_agg_backend(self):
        """Ensure Agg backend is set for all rendering tests."""
        import matplotlib
        matplotlib.use("Agg")

    def test_frame_shape(self):
        """render_frame() returns an RGB array of the expected dimensions."""
        from sonify.visualize import render_frame

        amplitudes = np.array([0.2, 0.5, 0.8, 1.0])
        frame = render_frame(amplitudes, fig_width=1280, fig_height=720)

        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (720, 1280, 3)

    def test_dots_mode_vs_circles_mode(self):
        """Both modes run and produce visually different frames."""
        from sonify.visualize import render_frame

        amplitudes = np.array([0.0, 0.3, 0.6, 1.0])

        frame_dots = render_frame(amplitudes, mode="dots",
                                  fig_width=640, fig_height=360)
        frame_circles = render_frame(amplitudes, mode="circles",
                                     fig_width=640, fig_height=360)

        # Both should produce valid frames
        assert frame_dots.shape == (360, 640, 3)
        assert frame_circles.shape == (360, 640, 3)

        # Modes should produce different output
        assert not np.array_equal(frame_dots, frame_circles), \
            "Dots and circles modes produced identical frames"

    def test_colormap_applied(self):
        """Frame with nonzero intensities is not all-black."""
        from sonify.visualize import render_frame

        amplitudes = np.array([0.5, 0.8, 1.0, 0.3])
        frame = render_frame(amplitudes, colormap="plasma",
                             fig_width=640, fig_height=360)

        # The dark background is #0A0A0A (10, 10, 10), not pure black.
        # With nonzero intensities and a colormap, there should be
        # bright pixels somewhere in the frame.
        max_pixel = frame.max()
        assert max_pixel > 20, \
            f"Frame appears all-black (max pixel value: {max_pixel})"

    def test_depth_label_changes_frame(self):
        """Depth label is shown when depth is provided (independent of show_labels)."""
        from sonify.visualize import render_frame

        amplitudes = np.array([0.5, 0.5, 0.5, 0.5])

        frame_no_depth = render_frame(amplitudes, depth=None, show_labels=False,
                                      fig_width=640, fig_height=360)
        frame_with_depth = render_frame(amplitudes, depth=123.45, show_labels=False,
                                        fig_width=640, fig_height=360)

        assert not np.array_equal(frame_no_depth, frame_with_depth), \
            "Depth label should change the frame when depth is provided"

    def test_channel_labels_change_frame(self):
        """Channel labels (show_labels) produce different frames, independent of depth."""
        from sonify.visualize import render_frame

        amplitudes = np.array([0.5, 0.5, 0.5, 0.5])

        frame_no_labels = render_frame(amplitudes, show_labels=False,
                                       fig_width=640, fig_height=360)
        frame_with_labels = render_frame(amplitudes, show_labels=True,
                                         fig_width=640, fig_height=360)

        assert not np.array_equal(frame_no_labels, frame_with_labels), \
            "Channel labels should change the frame when show_labels=True"
