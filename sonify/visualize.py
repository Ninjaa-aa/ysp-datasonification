"""
Visual rendering engine for the sonification toolkit (Phase 2).

Renders per-row intensity data as colored dots or circles on a dark
background.  Fully generic — no hardcoded band count, column names,
or dataset-specific logic.

The visual layer applies its own intensity scaling (``visual_scale``)
independently of the audio scaling (``scale``), so audio and visual can
use different scaling modes simultaneously.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sonify.mapping import scale_values, normalize_per_channel


# ---------------------------------------------------------------------------
# Visual scaling (independent of audio scaling)
# ---------------------------------------------------------------------------

def apply_visual_scale(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Apply visual-specific scaling then re-normalize to [0, 1].

    Delegates to ``mapping.scale_values()`` + ``mapping.normalize_per_channel()``
    so no scaling logic is duplicated.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``, values >= 0 (pre-clipped by
        ``preprocess.clean()``).
    mode : str
        One of ``'linear'``, ``'log10'``, ``'ln'``.

    Returns
    -------
    np.ndarray
        2-D array ``(n_rows, n_channels)`` with all values in [0, 1].
    """
    scaled = scale_values(matrix, mode)
    return normalize_per_channel(scaled)


# ---------------------------------------------------------------------------
# Single-frame rendering
# ---------------------------------------------------------------------------

def render_frame(
    amplitudes: np.ndarray,
    mode: str = "dots",
    colormap: str = "plasma",
    depth: float | None = None,
    wavelengths: np.ndarray | None = None,
    show_labels: bool = False,
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1280,
    fig_height: int = 720,
) -> np.ndarray:
    """Render one frame as an RGB numpy array.

    Caller is responsible for setting the matplotlib backend before calling
    this function (e.g. ``matplotlib.use('Agg')`` for batch rendering).

    Parameters
    ----------
    amplitudes : np.ndarray
        1-D array of length ``n_channels``, values in [0, 1].
    mode : str
        ``'dots'`` (fixed-size, color encodes intensity) or
        ``'circles'`` (radius scales with intensity, color also maps intensity).
    colormap : str
        Any valid matplotlib colormap name.
    depth : float or None
        Current depth (m) to display in the frame.  Shown whenever not None,
        independent of ``show_labels``.
    wavelengths : np.ndarray or None
        Per-channel wavelength centers (nm) for channel labels.
    show_labels : bool
        If True, display channel index or wavelength below each dot/circle.
    title : str
        Title string displayed at the top of the frame.
    fig_width, fig_height : int
        Frame dimensions in pixels.

    Returns
    -------
    np.ndarray
        RGB array of shape ``(fig_height, fig_width, 3)``, dtype uint8.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    n_channels = len(amplitudes)
    dpi = 100
    fig_w_inches = fig_width / dpi
    fig_h_inches = fig_height / dpi

    fig, ax = plt.subplots(1, 1, figsize=(fig_w_inches, fig_h_inches), dpi=dpi)
    fig.patch.set_facecolor("#0A0A0A")
    ax.set_facecolor("#0A0A0A")

    # Layout: channels evenly spaced along x-axis
    x_positions = np.linspace(0.1, 0.9, n_channels)
    y_center = 0.5

    # Get colormap
    cmap = matplotlib.colormaps[colormap]

    # Map intensities to colors
    colors = cmap(amplitudes)

    if mode == "dots":
        # Fixed-size dots, color encodes intensity
        marker_size = max(800 / n_channels, 30)
        ax.scatter(
            x_positions,
            [y_center] * n_channels,
            s=marker_size,
            c=colors,
            edgecolors="none",
            zorder=5,
        )
    elif mode == "circles":
        # Circle radius scales with intensity
        min_size = max(200 / n_channels, 10)
        max_size = max(3000 / n_channels, 100)
        sizes = min_size + amplitudes * (max_size - min_size)
        ax.scatter(
            x_positions,
            [y_center] * n_channels,
            s=sizes,
            c=colors,
            edgecolors="none",
            zorder=5,
        )
    else:
        raise ValueError(f"Unknown visual mode '{mode}'; expected 'dots' or 'circles'")

    # Channel labels (below each marker)
    if show_labels:
        for i, x in enumerate(x_positions):
            if wavelengths is not None and i < len(wavelengths):
                label = f"{wavelengths[i]:.0f} nm"
            else:
                label = f"Ch {i + 1}"
            ax.text(
                x, 0.32, label,
                color="#AAAAAA",
                fontsize=max(6, min(9, 200 / n_channels)),
                ha="center", va="top",
                transform=ax.transAxes,
            )

    # Title
    ax.text(
        0.5, 0.95, title,
        color="#DDDDDD",
        fontsize=14,
        fontweight="bold",
        ha="center", va="top",
        transform=ax.transAxes,
    )

    # Depth label (shown whenever depth is provided, independent of show_labels)
    if depth is not None:
        ax.text(
            0.05, 0.08, f"Depth: {depth:.2f} m",
            color="#88CCFF",
            fontsize=11,
            ha="left", va="bottom",
            transform=ax.transAxes,
        )

    # Clean up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.tight_layout(pad=0)

    # Render to RGB array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = buf[:, :, :3].copy()

    plt.close(fig)
    return rgb


# ---------------------------------------------------------------------------
# Batch frame rendering (for video export)
# ---------------------------------------------------------------------------

def render_all_frames(
    amplitude_matrix: np.ndarray,
    mode: str = "dots",
    colormap: str = "plasma",
    depths: np.ndarray | None = None,
    wavelengths: np.ndarray | None = None,
    show_labels: bool = False,
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1280,
    fig_height: int = 720,
) -> list[np.ndarray]:
    """Render all frames for video export.

    Sets the Agg backend internally so no GUI window is opened.
    Must not be called in the same process as ``live_display()``.

    Parameters
    ----------
    amplitude_matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``, values in [0, 1]
        (already visual-scaled and normalized).
    depths : np.ndarray or None
        1-D array of depth values (one per row), or None.
    Other parameters: see ``render_frame()``.

    Returns
    -------
    list[np.ndarray]
        List of RGB frames, each ``(fig_height, fig_width, 3)`` uint8.
    """
    import matplotlib
    matplotlib.use("Agg")
    # pyplot imported after backend is set
    import matplotlib.pyplot as plt  # noqa: F401

    n_rows = amplitude_matrix.shape[0]
    frames: list[np.ndarray] = []

    print(f"Rendering {n_rows} visual frames ({fig_width}x{fig_height}) ...")

    for row_idx in range(n_rows):
        depth_val = float(depths[row_idx]) if depths is not None else None
        frame = render_frame(
            amplitudes=amplitude_matrix[row_idx],
            mode=mode,
            colormap=colormap,
            depth=depth_val,
            wavelengths=wavelengths,
            show_labels=show_labels,
            title=title,
            fig_width=fig_width,
            fig_height=fig_height,
        )
        frames.append(frame)

        # Progress feedback every 50 frames
        if (row_idx + 1) % 50 == 0 or row_idx == n_rows - 1:
            print(f"  Frame {row_idx + 1}/{n_rows}")

    print(f"Rendered {n_rows} frames.")
    return frames


# ---------------------------------------------------------------------------
# Live animated display (optional, best-effort sync)
# ---------------------------------------------------------------------------

def live_display(
    amplitude_matrix: np.ndarray,
    waveform: np.ndarray,
    sample_rate: int,
    playback_speed: float,
    mode: str = "dots",
    colormap: str = "plasma",
    depths: np.ndarray | None = None,
    wavelengths: np.ndarray | None = None,
    show_labels: bool = False,
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1280,
    fig_height: int = 720,
) -> None:
    """Show a matplotlib animation synchronized with audio playback.

    Does NOT set the matplotlib backend — relies on the system's interactive
    backend (Qt5Agg, TkAgg, etc.).  Must not be called in the same process
    as ``render_all_frames()``, which sets the Agg (non-interactive) backend.

    Sync is best-effort: matplotlib animation timing is not sample-accurate.
    For precise audio/video sync, use ``--video-output`` instead.

    Parameters
    ----------
    amplitude_matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``, values in [0, 1].
    waveform : np.ndarray
        1-D float64 mono waveform for audio playback.
    sample_rate : int
        Audio sample rate.
    playback_speed : float
        Rows per second (also determines animation interval).
    Other parameters: see ``render_frame()``.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n_rows, n_channels = amplitude_matrix.shape
    interval_ms = 1000.0 / playback_speed

    dpi = 100
    fig_w_inches = fig_width / dpi
    fig_h_inches = fig_height / dpi

    fig, ax = plt.subplots(1, 1, figsize=(fig_w_inches, fig_h_inches), dpi=dpi)
    fig.patch.set_facecolor("#0A0A0A")
    ax.set_facecolor("#0A0A0A")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cmap = matplotlib.colormaps[colormap]
    x_positions = np.linspace(0.1, 0.9, n_channels)
    y_center = 0.5

    # Initial scatter
    scatter = ax.scatter(
        x_positions, [y_center] * n_channels,
        s=100, c=cmap(amplitude_matrix[0]),
        edgecolors="none",
    )

    # Title
    title_text = ax.text(
        0.5, 0.95, title,
        color="#DDDDDD", fontsize=14, fontweight="bold",
        ha="center", va="top", transform=ax.transAxes,
    )

    # Depth label
    depth_text = ax.text(
        0.05, 0.08, "",
        color="#88CCFF", fontsize=11,
        ha="left", va="bottom", transform=ax.transAxes,
    )

    def update(frame_idx: int):
        amps = amplitude_matrix[frame_idx]
        colors = cmap(amps)
        scatter.set_facecolors(colors)

        if mode == "circles":
            min_size = max(200 / n_channels, 10)
            max_size = max(3000 / n_channels, 100)
            sizes = min_size + amps * (max_size - min_size)
            scatter.set_sizes(sizes)

        if depths is not None:
            depth_text.set_text(f"Depth: {depths[frame_idx]:.2f} m")

        return scatter, depth_text

    anim = FuncAnimation(
        fig, update, frames=n_rows,
        interval=interval_ms, blit=False, repeat=False,
    )

    # Start audio playback in background
    try:
        import sounddevice as sd
        print(f"Playing audio via sounddevice ({len(waveform)/sample_rate:.1f}s) ...")
        sd.play(waveform.astype(np.float32), samplerate=sample_rate)
    except Exception:
        print("Warning: Could not start audio playback. "
              "Displaying animation without audio.")

    print("Live display started (best-effort sync). Close the window to stop.")
    plt.show()
