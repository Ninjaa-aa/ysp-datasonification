"""
Video export: mux rendered frames with a WAV audio track.

All heavy dependencies (``opencv-python``, ``moviepy``) are imported lazily
inside the export function, so the rest of the pipeline works if they are
not installed.  Users who only need audio output are not forced to install
video libraries.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np


def export_video(
    frames: list[np.ndarray],
    wav_path: str,
    output_path: str,
    fps: float,
) -> None:
    """Write rendered frames as a video with synchronized audio.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of RGB frames, each ``(H, W, 3)`` uint8.
    wav_path : str
        Path to the WAV audio file on disk.  Must already exist.
    output_path : str
        Final output path (``.mp4`` or ``.avi``).
    fps : float
        Frames per second (= ``playback_speed``, rows per second).

    Raises
    ------
    ImportError
        If ``opencv-python`` or ``moviepy`` are not installed.
    FileNotFoundError
        If ``wav_path`` does not exist.
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    if not frames:
        raise ValueError("No frames to export")

    # --- Step 1: Write silent video with OpenCV ---
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video export. "
            "Install it with: pip install opencv-python"
        )

    h, w, _ = frames[0].shape

    # Choose codec based on output extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Temporary silent video file
    temp_fd, temp_video = tempfile.mkstemp(suffix=ext)
    os.close(temp_fd)

    print(f"Writing {len(frames)} frames to temporary video ({w}x{h} @ {fps} fps) ...")

    writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))
    try:
        for frame in frames:
            # Convert RGB → BGR for OpenCV
            bgr = frame[:, :, ::-1]
            writer.write(bgr)
    finally:
        writer.release()

    try:
        from moviepy.editor import VideoFileClip, AudioFileClip  # noqa
    except ImportError:
        # Clean up temp file before raising
        if os.path.isfile(temp_video):
            os.remove(temp_video)
        raise ImportError(
            "moviepy is required for video export. "
            "Install it with: pip install 'moviepy>=1.0,<2.0'"
        )

    print(f"Muxing audio from {wav_path} into video ...")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    video_clip = VideoFileClip(temp_video)
    audio_clip = AudioFileClip(wav_path)

    # Set audio on the video clip
    final_clip = video_clip.set_audio(audio_clip)

    # Write final output
    final_clip.write_videofile(
        output_path,
        codec="libx264" if ext == ".mp4" else None,
        audio_codec="aac" if ext == ".mp4" else None,
        logger=None,  # suppress moviepy's verbose output
    )

    # Clean up
    final_clip.close()
    video_clip.close()
    audio_clip.close()

    if os.path.isfile(temp_video):
        os.remove(temp_video)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported video: {output_path} ({file_size_mb:.1f} MB)")
