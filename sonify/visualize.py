"""
Visual rendering engine for the sonification toolkit.

Renders per-row intensity data as colored dots/circles (≤ 256 channels) or
heatmap strips (> 256 channels) on a dark background.  Supports a **trail**
display — N rows visible simultaneously, with older rows fading in opacity
and marker size.

Display refinements requested by Dr. Malaska (2026-07-09):
- Configurable marker size and shape (circle/square)
- Intensity colorbar
- Global gain normalization via apply_visual_scale() gain_mode parameter

Fully generic — no hardcoded band count, column names, or dataset-specific
logic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sonify.mapping import scale_values


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Above this channel count, render as a heatmap strip instead of scatter
# dots/circles.  Scatter is readable at ≤ 256 channels; heatmap is faster
# and more readable at high counts.
SCATTER_CHANNEL_LIMIT = 256


# ---------------------------------------------------------------------------
# Visual scaling (independent of audio scaling)
# ---------------------------------------------------------------------------

def apply_visual_scale(
    matrix: np.ndarray,
    visual_scale: str,
    gain_mode: str = "max_linear",
) -> np.ndarray:
    """Apply visual-specific scaling then gain normalization to [0, 1].

    Parameters
    ----------
    matrix : np.ndarray
        2-D array ``(n_rows, n_channels)``, values >= 0 (pre-clipped by
        ``preprocess.clean()``).
    visual_scale : str
        One of ``'linear'``, ``'log10'``, ``'ln'`` — the pre-normalization
        transform.
    gain_mode : str
        The gain normalization mode — same modes as the audio path.
        Controls the reference point for mapping to [0, 1].
        Default ``"max_linear"`` preserves backward-compatible behavior.

    Returns
    -------
    np.ndarray
        2-D array ``(n_rows, n_channels)`` with all values in [0, 1].
    """
    from sonify.mapping import apply_global_gain

    scaled = scale_values(matrix, visual_scale)
    return apply_global_gain(scaled, gain_mode)


# ---------------------------------------------------------------------------
# Single-frame rendering (with trail support)
# ---------------------------------------------------------------------------

def render_frame(
    amplitude_history: np.ndarray,
    mode: str = "dots",
    colormap: str = "plasma",
    depths: np.ndarray | None = None,
    wavelengths: np.ndarray | None = None,
    show_labels: bool = False,
    title: str = "Sounds of Deep Ice Fluorescence",
    fig_width: int = 1280,
    fig_height: int = 720,
    minimap_image: np.ndarray | None = None,
    minimap_position: float | None = None,
    marker_size: int = 120,
    marker_shape: str = "square",
    show_colorbar: bool = True,
) -> np.ndarray:
    """Render one frame as an RGB numpy array.

    Caller is responsible for setting the matplotlib backend before calling
    this function (e.g. ``matplotlib.use('Agg')`` for batch rendering).

    Parameters
    ----------
    amplitude_history : np.ndarray
        2-D array of shape ``(n_trail, n_channels)``, values in [0, 1].
        The last row ``[-1]`` is the current row; earlier rows are older.
    mode : str
        ``'dots'`` (fixed-size, color encodes intensity) or
        ``'circles'`` (radius scales with intensity, color also maps intensity).
    colormap : str
        Any valid matplotlib colormap name.
    depths : np.ndarray or None
        1-D array of depth values, one per trail row (same length as
        ``amplitude_history.shape[0]``).  Shown on the left side of each row.
    wavelengths : np.ndarray or None
        Per-channel wavelength centers (nm) for channel labels.
    show_labels : bool
        If True, display channel index or wavelength below each dot/circle.
    title : str
        Title string displayed at the top of the frame.
    fig_width, fig_height : int
        Frame dimensions in pixels.
    minimap_image : np.ndarray or None
        Pre-rendered minimap RGB array to composite into the right panel.
    minimap_position : float or None
        Fractional position [0, 1] within the minimap for the current row.
    marker_size : int
        Base marker size for scatter plots (matplotlib ``s=`` parameter).
    marker_shape : str
        ``'circle'`` or ``'square'`` — maps to matplotlib markers ``'o'``/``'s'``.
    show_colorbar : bool
        If True, add a colorbar showing the intensity-to-color mapping.

    Returns
    -------
    np.ndarray
        RGB array of shape ``(fig_height, fig_width, 3)``, dtype uint8.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n_trail, n_channels = amplitude_history.shape
    dpi = 100
    fig_w_inches = fig_width / dpi
    fig_h_inches = fig_height / dpi

    use_heatmap = n_channels > SCATTER_CHANNEL_LIMIT
    show_minimap = minimap_image is not None and minimap_position is not None

    # Matplotlib marker character
    marker_char = "s" if marker_shape == "square" else "o"

    if show_minimap:
        fig = plt.figure(figsize=(fig_w_inches, fig_h_inches), dpi=dpi)
        gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig, wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax_mini = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(fig_w_inches, fig_h_inches), dpi=dpi)
        ax_mini = None

    fig.patch.set_facecolor("#0A0A0A")
    ax.set_facecolor("#0A0A0A")

    # Get colormap
    cmap = matplotlib.colormaps[colormap]

    if use_heatmap:
        # ── Heatmap strip rendering (for > 256 channels) ─────────────
        ax.imshow(
            amplitude_history,
            aspect="auto",
            cmap=colormap,
            vmin=0,
            vmax=1,
            origin="upper",
        )
        ax.set_xlabel("Channel", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("Trail Row", color="#AAAAAA", fontsize=9)
        ax.tick_params(colors="#666666", labelsize=7)

        # Depth labels on the y-axis
        if depths is not None:
            ax.set_yticks(range(n_trail))
            ax.set_yticklabels(
                [f"{d:.2f} m" for d in depths],
                color="#88CCFF", fontsize=8,
            )
    else:
        # ── Scatter-based rendering (dots/circles) ───────────────────
        x_positions = np.linspace(0.1, 0.9, n_channels)

        # Vertical positions for trail rows: evenly spaced, oldest at top
        if n_trail == 1:
            y_positions = [0.5]
        else:
            y_positions = np.linspace(0.75, 0.25, n_trail)

        # Opacity and size scaling: oldest (index 0) → faded, newest (last) → full
        for trail_idx in range(n_trail):
            # Age fraction: 0.0 = oldest, 1.0 = newest
            if n_trail == 1:
                age_frac = 1.0
            else:
                age_frac = trail_idx / (n_trail - 1)

            # Opacity: 20% → 100%
            opacity = 0.2 + 0.8 * age_frac
            # Size fraction: 40% → 100%
            size_frac = 0.4 + 0.6 * age_frac

            amps = amplitude_history[trail_idx]
            colors = cmap(amps)
            # Apply opacity to alpha channel
            colors[:, 3] = opacity

            y_center = y_positions[trail_idx]

            if mode == "dots":
                # Fixed marker size scaled by trail age
                sizes = marker_size * size_frac
                ax.scatter(
                    x_positions,
                    [y_center] * n_channels,
                    s=sizes,
                    c=colors,
                    marker=marker_char,
                    edgecolors="none",
                    zorder=5,
                )
            elif mode == "circles":
                # Size scales with intensity: base + (base * 3 * amp) → 1x to 4x range
                sizes = (marker_size + marker_size * 3 * amps) * size_frac
                ax.scatter(
                    x_positions,
                    [y_center] * n_channels,
                    s=sizes,
                    c=colors,
                    marker=marker_char,
                    edgecolors="none",
                    zorder=5,
                )
            else:
                raise ValueError(
                    f"Unknown visual mode '{mode}'; expected 'dots' or 'circles'"
                )

            # Depth label for this trail row
            if depths is not None and trail_idx < len(depths):
                ax.text(
                    0.02, y_center,
                    f"{depths[trail_idx]:.2f} m",
                    color="#88CCFF",
                    fontsize=9,
                    alpha=opacity,
                    ha="left", va="center",
                    transform=ax.transAxes,
                )

        # Channel labels (below the bottom-most trail row)
        if show_labels:
            label_y = 0.12 if n_trail > 1 else 0.32
            for i, x in enumerate(x_positions):
                if wavelengths is not None and i < len(wavelengths):
                    label = f"{wavelengths[i]:.0f} nm"
                else:
                    label = f"Ch {i + 1}"
                ax.text(
                    x, label_y, label,
                    color="#AAAAAA",
                    fontsize=max(6, min(9, 200 / n_channels)),
                    ha="center", va="top",
                    transform=ax.transAxes,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Title
    ax.set_title(
        title,
        color="#DDDDDD",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    # ── Colorbar (Dr. Malaska: "add a scale somewhere?") ──────────────
    if show_colorbar:
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("Intensity", color="white", fontsize=8)
        cbar.ax.tick_params(colors="white", labelsize=7)

    # ── Minimap panel ─────────────────────────────────────────────────
    if show_minimap and ax_mini is not None:
        ax_mini.set_facecolor("#0A0A0A")
        ax_mini.imshow(minimap_image, aspect="auto", origin="upper")
        # Draw position line
        h = minimap_image.shape[0]
        line_y = minimap_position * h
        ax_mini.axhline(y=line_y, color="#FF4444", linewidth=1.5, alpha=0.9)
        ax_mini.set_title("Overview", color="#AAAAAA", fontsize=9, pad=5)
        ax_mini.tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False,
        )
        for spine in ax_mini.spines.values():
            spine.set_color("#333333")

    fig.tight_layout(pad=0.5)

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
    trail_rows: int = 5,
    max_frames: int = 500,
    show_minimap: bool = False,
    marker_size: int = 120,
    marker_shape: str = "square",
    show_colorbar: bool = True,
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
    trail_rows : int
        Number of rows visible simultaneously in the trail display.
    max_frames : int
        Maximum number of frames to render.  If n_rows exceeds this, a
        warning is emitted and rendering is truncated.
    show_minimap : bool
        If True, render a minimap overview panel on the right side.
    marker_size : int
        Base marker size for scatter plots.
    marker_shape : str
        ``'circle'`` or ``'square'``.
    show_colorbar : bool
        If True, add a colorbar showing the intensity-to-color mapping.
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

    # ── Max frames safety cap ─────────────────────────────────────────
    if n_rows > max_frames:
        import warnings
        warnings.warn(
            f"Row count ({n_rows}) exceeds max_frames ({max_frames}). "
            f"Only the first {max_frames} rows will be rendered. "
            f"Use --max-frames to increase the limit.",
            stacklevel=2,
        )
        n_rows = max_frames

    frames: list[np.ndarray] = []

    # ── Pre-render minimap if requested ───────────────────────────────
    minimap_img = None
    if show_minimap:
        import matplotlib
        cmap = matplotlib.colormaps[colormap]
        # Render full amplitude_matrix as heatmap image
        minimap_rgba = cmap(amplitude_matrix[:n_rows])  # (n_rows, n_ch, 4)
        minimap_img = (minimap_rgba[:, :, :3] * 255).astype(np.uint8)

    print(f"Rendering {n_rows} visual frames ({fig_width}x{fig_height}, "
          f"trail={trail_rows}) ...")

    # Progress milestones at 25% intervals
    milestones = set()
    for pct in (25, 50, 75, 100):
        milestone_idx = max(0, round(n_rows * pct / 100) - 1)
        milestones.add(milestone_idx)

    for row_idx in range(n_rows):
        # Build trail window: amplitude_matrix[max(0, i-trail+1) : i+1]
        start = max(0, row_idx - trail_rows + 1)
        history = amplitude_matrix[start : row_idx + 1]

        # Corresponding depths
        trail_depths = None
        if depths is not None:
            trail_depths = depths[start : row_idx + 1]

        # Minimap position
        mini_pos = None
        if minimap_img is not None:
            mini_pos = row_idx / max(1, n_rows - 1)

        frame = render_frame(
            amplitude_history=history,
            mode=mode,
            colormap=colormap,
            depths=trail_depths,
            wavelengths=wavelengths,
            show_labels=show_labels,
            title=title,
            fig_width=fig_width,
            fig_height=fig_height,
            minimap_image=minimap_img,
            minimap_position=mini_pos,
            marker_size=marker_size,
            marker_shape=marker_shape,
            show_colorbar=show_colorbar,
        )
        frames.append(frame)

        # Progress feedback at milestones
        if row_idx in milestones:
            pct = round((row_idx + 1) / n_rows * 100)
            print(f"  Frame {row_idx + 1}/{n_rows} ({pct}%)")

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
    trail_rows: int = 5,
    marker_size: int = 120,
    marker_shape: str = "square",
    show_colorbar: bool = True,
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
    trail_rows : int
        Number of rows visible simultaneously in the trail display.
    marker_size : int
        Base marker size for scatter plots.
    marker_shape : str
        ``'circle'`` or ``'square'``.
    show_colorbar : bool
        If True, add a colorbar showing the intensity-to-color mapping.
    Other parameters: see ``render_frame()``.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from collections import deque

    n_rows, n_channels = amplitude_matrix.shape
    interval_ms = 1000.0 / playback_speed

    use_heatmap = n_channels > SCATTER_CHANNEL_LIMIT

    # Matplotlib marker character
    marker_char = "s" if marker_shape == "square" else "o"

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

    # Trail history deque
    history: deque = deque(maxlen=trail_rows)
    history.append(amplitude_matrix[0])

    # Title
    title_text = ax.text(
        0.5, 0.95, title,
        color="#DDDDDD", fontsize=14, fontweight="bold",
        ha="center", va="top", transform=ax.transAxes,
    )

    def update(frame_idx: int):
        ax.clear()
        ax.set_facecolor("#0A0A0A")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Update trail history
        if frame_idx > 0:
            history.append(amplitude_matrix[frame_idx])
        trail = np.array(list(history))
        n_trail = trail.shape[0]

        if use_heatmap:
            ax.imshow(
                trail,
                aspect="auto",
                cmap=colormap,
                vmin=0, vmax=1,
                origin="upper",
                extent=[0, 1, 0, 1],
            )
        else:
            if n_trail == 1:
                y_positions = [0.5]
            else:
                y_positions = np.linspace(0.75, 0.25, n_trail)

            for trail_idx in range(n_trail):
                age_frac = 1.0 if n_trail == 1 else trail_idx / (n_trail - 1)
                opacity = 0.2 + 0.8 * age_frac
                size_frac = 0.4 + 0.6 * age_frac

                amps = trail[trail_idx]
                colors = cmap(amps)
                colors[:, 3] = opacity

                y_center = y_positions[trail_idx]

                if mode == "circles":
                    sizes = (marker_size + marker_size * 3 * amps) * size_frac
                else:
                    sizes = marker_size * size_frac

                ax.scatter(
                    x_positions, [y_center] * n_channels,
                    s=sizes, c=colors, marker=marker_char,
                    edgecolors="none",
                )

                if depths is not None:
                    start = max(0, frame_idx - n_trail + 1 + trail_idx)
                    if start < len(depths):
                        ax.text(
                            0.02, y_center,
                            f"{depths[start]:.2f} m",
                            color="#88CCFF", fontsize=9,
                            alpha=opacity,
                            ha="left", va="center",
                            transform=ax.transAxes,
                        )

        # Colorbar (re-add each frame since ax is cleared)
        if show_colorbar:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=0, vmax=1)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
            cbar.set_label("Intensity", color="white", fontsize=8)
            cbar.ax.tick_params(colors="white", labelsize=7)

        # Title
        ax.text(
            0.5, 0.95, title,
            color="#DDDDDD", fontsize=14, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes,
        )

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
