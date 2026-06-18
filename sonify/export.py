"""
WAV file export.

Converts a float64 [-1, 1] waveform to int16 and writes via
scipy.io.wavfile.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io import wavfile


def export_wav(waveform: np.ndarray, sample_rate: int, path: str) -> None:
    """Export a waveform to a .wav file.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float64 array in ``[-1, 1]``.
    sample_rate : int
        Audio sample rate.
    path : str
        Output file path (will create parent directories if needed).
    """
    # Ensure the output directory exists
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Convert float [-1, 1] → int16
    int16_data = (waveform * 32767).clip(-32768, 32767).astype(np.int16)

    wavfile.write(path, sample_rate, int16_data)
    duration = len(waveform) / sample_rate
    print(f"Exported {duration:.1f}s WAV to: {path}")
