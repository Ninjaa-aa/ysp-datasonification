"""
Speaker playback with sounddevice → simpleaudio fallback.

If neither library is available, prints install instructions rather than
silently doing nothing.
"""

from __future__ import annotations

import numpy as np


def play(waveform: np.ndarray, sample_rate: int) -> None:
    """Play a float64 waveform through the system speakers.

    Tries ``sounddevice`` first, then falls back to ``simpleaudio``.
    If both fail, raises RuntimeError with install instructions.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float64 array in ``[-1, 1]``.
    sample_rate : int
        Audio sample rate.
    """
    # --- Attempt 1: sounddevice ---
    try:
        import sounddevice as sd

        print(f"Playing audio via sounddevice ({len(waveform)/sample_rate:.1f}s) ...")
        sd.play(waveform.astype(np.float32), samplerate=sample_rate)
        sd.wait()
        print("Playback finished.")
        return
    except Exception as sd_err:
        sd_message = str(sd_err)

    # --- Attempt 2: simpleaudio ---
    try:
        import simpleaudio as sa

        # Convert float [-1, 1] → int16 (same conversion as export.py)
        int16_data = (waveform * 32767).clip(-32768, 32767).astype(np.int16)

        print(f"Playing audio via simpleaudio ({len(waveform)/sample_rate:.1f}s) ...")
        play_obj = sa.play_buffer(
            int16_data.tobytes(),
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=sample_rate,
        )
        play_obj.wait_done()
        print("Playback finished.")
        return
    except Exception as sa_err:
        sa_message = str(sa_err)

    # --- Both failed ---
    raise RuntimeError(
        f"Could not play audio.\n"
        f"  sounddevice error: {sd_message}\n"
        f"  simpleaudio error: {sa_message}\n\n"
        f"Install one of:\n"
        f"  pip install sounddevice\n"
        f"  pip install simpleaudio\n"
    )
