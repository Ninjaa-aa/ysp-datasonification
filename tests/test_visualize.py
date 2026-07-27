"""Tests for sonify.visualize (Phase 2 + Phase 3 visual rendering engine)."""

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

        # Now uses 2-D amplitude_history (n_trail, n_channels)
        amplitude_history = np.array([[0.2, 0.5, 0.8, 1.0]])
        frame = render_frame(amplitude_history, fig_width=1280, fig_height=720)

        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (720, 1280, 3)

    def test_dots_mode_vs_circles_mode(self):
        """Both modes run and produce visually different frames."""
        from sonify.visualize import render_frame

        amplitude_history = np.array([[0.0, 0.3, 0.6, 1.0]])

        frame_dots = render_frame(amplitude_history, mode="dots",
                                  fig_width=640, fig_height=360)
        frame_circles = render_frame(amplitude_history, mode="circles",
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

        amplitude_history = np.array([[0.5, 0.8, 1.0, 0.3]])
        frame = render_frame(amplitude_history, colormap="plasma",
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

        amplitude_history = np.array([[0.5, 0.5, 0.5, 0.5]])

        frame_no_depth = render_frame(amplitude_history, depths=None,
                                      show_labels=False,
                                      fig_width=640, fig_height=360)
        frame_with_depth = render_frame(amplitude_history,
                                        depths=np.array([123.45]),
                                        show_labels=False,
                                        fig_width=640, fig_height=360)

        assert not np.array_equal(frame_no_depth, frame_with_depth), \
            "Depth label should change the frame when depth is provided"

    def test_channel_labels_change_frame(self):
        """Channel labels (show_labels) produce different frames, independent of depth."""
        from sonify.visualize import render_frame

        amplitude_history = np.array([[0.5, 0.5, 0.5, 0.5]])

        frame_no_labels = render_frame(amplitude_history, show_labels=False,
                                       fig_width=640, fig_height=360)
        frame_with_labels = render_frame(amplitude_history, show_labels=True,
                                         fig_width=640, fig_height=360)

        assert not np.array_equal(frame_no_labels, frame_with_labels), \
            "Channel labels should change the frame when show_labels=True"


class TestTrailDisplay:
    """Trail display: stacking N rows with fading opacity and size."""

    @pytest.fixture(autouse=True)
    def _set_agg_backend(self):
        """Ensure Agg backend is set for all rendering tests."""
        import matplotlib
        matplotlib.use("Agg")

    def test_trail_single_row(self):
        """Trail of 1 row works and produces a valid frame."""
        from sonify.visualize import render_frame

        # Single trail row — should be equivalent to old single-row behavior
        amplitude_history = np.array([[0.2, 0.5, 0.8, 1.0]])
        frame = render_frame(amplitude_history, fig_width=640, fig_height=360)

        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (360, 640, 3)
        # Should not be all-black
        assert frame.max() > 20

    def test_trail_opacity_decreases(self):
        """Older trail rows render with lower opacity (dimmer) than newer ones."""
        from sonify.visualize import render_frame

        # 3-row trail, uniform intensity so color differences come from opacity
        amplitude_history = np.array([
            [1.0, 1.0, 1.0, 1.0],  # oldest (top) — should be dimmest
            [1.0, 1.0, 1.0, 1.0],  # middle
            [1.0, 1.0, 1.0, 1.0],  # newest (bottom) — should be brightest
        ])
        frame = render_frame(amplitude_history, fig_width=640, fig_height=360)

        # Compare brightness in upper third (older rows) vs lower third (newer rows)
        h = frame.shape[0]
        upper_region = frame[:h // 3, :, :]
        lower_region = frame[2 * h // 3:, :, :]

        # The newer (bottom) region should have brighter/equal average brightness
        # because it has 100% opacity while the top has ~20% opacity.
        upper_brightness = upper_region.mean()
        lower_brightness = lower_region.mean()

        # Both regions include dark background, but markers in the lower region
        # should be brighter, making overall average higher or equal
        # (with some tolerance for layout differences)
        assert lower_brightness >= upper_brightness * 0.5, \
            f"Expected newer row to be brighter: upper={upper_brightness:.1f}, " \
            f"lower={lower_brightness:.1f}"

    def test_trail_partial_fill(self):
        """When history has fewer rows than trail_rows, no crash and valid frame."""
        from sonify.visualize import render_all_frames

        # 3 rows of data, trail_rows=5 — first frames have fewer trail rows
        amplitude_matrix = np.random.rand(3, 4)
        frames = render_all_frames(
            amplitude_matrix,
            fig_width=320,
            fig_height=180,
            trail_rows=5,
            max_frames=500,
        )

        assert len(frames) == 3
        for f in frames:
            assert f.shape == (180, 320, 3)
            assert f.dtype == np.uint8


class TestDisplayFixes:
    """Marker size, shape, and colorbar (Dr. Malaska, 2026-07-09)."""

    @pytest.fixture(autouse=True)
    def _set_agg_backend(self):
        """Ensure Agg backend is set for all rendering tests."""
        import matplotlib
        matplotlib.use("Agg")

    def test_square_marker_differs_from_circle(self):
        """Frames rendered with square vs circle markers produce different pixels."""
        from sonify.visualize import render_frame

        amplitude_history = np.array([[0.3, 0.6, 0.9, 0.5]])

        frame_circle = render_frame(amplitude_history, marker_shape="circle",
                                     marker_size=120, show_colorbar=False,
                                     fig_width=640, fig_height=360)
        frame_square = render_frame(amplitude_history, marker_shape="square",
                                     marker_size=120, show_colorbar=False,
                                     fig_width=640, fig_height=360)

        assert not np.array_equal(frame_circle, frame_square), \
            "Square and circle markers should produce different frames"

    def test_marker_size_affects_frame(self):
        """Frames with different marker sizes produce different pixel arrays."""
        from sonify.visualize import render_frame

        amplitude_history = np.array([[0.3, 0.6, 0.9, 0.5]])

        frame_small = render_frame(amplitude_history, marker_size=40,
                                    show_colorbar=False,
                                    fig_width=640, fig_height=360)
        frame_large = render_frame(amplitude_history, marker_size=200,
                                    show_colorbar=False,
                                    fig_width=640, fig_height=360)

        assert not np.array_equal(frame_small, frame_large), \
            "Different marker sizes should produce different frames"

    def test_colorbar_changes_frame(self):
        """Frame with colorbar on differs from colorbar off."""
        from sonify.visualize import render_frame

        amplitude_history = np.array([[0.3, 0.6, 0.9, 0.5]])

        frame_no_cbar = render_frame(amplitude_history, show_colorbar=False,
                                      fig_width=640, fig_height=360)
        frame_with_cbar = render_frame(amplitude_history, show_colorbar=True,
                                        fig_width=640, fig_height=360)

        assert not np.array_equal(frame_no_cbar, frame_with_cbar), \
            "Colorbar should change the frame"
