"""
Speaker playback with sounddevice.

If the library is not available, prints install instructions rather than
silently doing nothing.
"""

from __future__ import annotations

import numpy as np


def play(waveform: np.ndarray, sample_rate: int) -> None:
    """Play a float64 waveform through the system speakers.

    Uses ``sounddevice`` for playback.
    If it fails, raises RuntimeError with install instructions.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float64 array in ``[-1, 1]``.
    sample_rate : int
        Audio sample rate.
    """
    try:
        import sounddevice as sd

        print(f"Playing audio via sounddevice ({len(waveform)/sample_rate:.1f}s) ...")
        sd.play(waveform.astype(np.float32), samplerate=sample_rate)
        sd.wait()
        print("Playback finished.")
    except Exception as sd_err:
        sd_message = str(sd_err)
        raise RuntimeError(
            f"Could not play audio.\n"
            f"  sounddevice error: {sd_message}\n\n"
            f"Please install sounddevice:\n"
            f"  pip install sounddevice\n"
        )
