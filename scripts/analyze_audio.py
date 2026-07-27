#!/usr/bin/env python
"""
Audio analysis for sonification output: quality metrics + spectrogram.

Replaces the three earlier one-off scripts (gen_spec.py, generate_spectrograms.py,
generate_spectrogram_fixed.py), which each read a WAV and wrote a spectrogram with
hardcoded filenames.

Usage
-----
    # Metrics table for one or more files
    py scripts/analyze_audio.py outputs/chime.wav outputs/ambient.wav

    # Metrics plus a spectrogram PNG per file
    py scripts/analyze_audio.py --spectrogram outputs/chime.wav

    # Compare against the reference files Dr. Malaska approved
    py scripts/analyze_audio.py --compare outputs/my_new_render.wav
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sonify.quality import describe  # noqa: E402


# Files Dr. Malaska has listened to and approved — the sound we must not regress from.
REFERENCE_FILES = [
    "outputs/chime_fixed.wav",
    "outputs/chime_full.wav",
    "outputs/chime_quiet_section.wav",
]


def load_wav(path: str) -> tuple[int, np.ndarray]:
    """Read a WAV as float64 in [-1, 1]."""
    sr, data = wavfile.read(path)
    x = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        x /= float(np.iinfo(data.dtype).max + 1)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x


def print_header() -> None:
    print(
        f"{'file':<38} {'dur':>7} {'rms':>7} {'crest':>7} "
        f"{'artic':>6} {'onset/s':>8} {'flat':>7} {'silence':>8}"
    )
    print("-" * 92)


def print_row(path: str, m: dict[str, float]) -> None:
    print(
        f"{os.path.basename(path):<38} "
        f"{m['duration_s']:>6.1f}s {m['rms']:>7.4f} {m['crest_db']:>6.1f}dB "
        f"{m['articulation']:>6.3f} {m['onset_rate']:>8.2f} "
        f"{m['spectral_flatness']:>7.4f} {100 * m['silence_fraction']:>7.1f}%"
    )


def save_spectrogram(path: str, sr: int, x: np.ndarray, max_hz: float = 2500.0) -> str:
    """Render a spectrogram PNG next to the source file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    f, t, Sxx = signal.spectrogram(x, sr, nperseg=2048)
    keep = f <= max_hz

    plt.figure(figsize=(14, 5))
    plt.pcolormesh(
        t, f[keep], 10 * np.log10(Sxx[keep] + 1e-10),
        shading="gouraud", cmap="plasma", vmin=-100, vmax=-20,
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(f"{os.path.basename(path)} — Spectrogram")
    plt.colorbar(label="Power (dB)")
    plt.ylim(0, max_hz)
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.dirname(path) or "."
    out_path = os.path.join(out_dir, f"spectrogram_{base}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Quality metrics and spectrograms for sonification output.",
    )
    p.add_argument("files", nargs="+", help="WAV file(s) to analyze")
    p.add_argument("--spectrogram", action="store_true",
                   help="Also write a spectrogram PNG per file")
    p.add_argument("--compare", action="store_true",
                   help="Also show the approved reference files for comparison")
    p.add_argument("--max-hz", type=float, default=2500.0,
                   help="Upper frequency limit for the spectrogram")
    args = p.parse_args()

    targets = list(args.files)
    if args.compare:
        targets += [f for f in REFERENCE_FILES if os.path.isfile(f)]

    print_header()
    for path in targets:
        if not os.path.isfile(path):
            print(f"{os.path.basename(path):<38} -- file not found --")
            continue

        sr, x = load_wav(path)
        print_row(path, describe(x, sr))

        if args.spectrogram:
            out = save_spectrogram(path, sr, x, max_hz=args.max_hz)
            print(f"{'':<38} -> {out}")

    if args.compare:
        print()
        print("Reference values from files Dr. Malaska approved:")
        print("  articulation >= 0.97  (notes separated by silence, not a drone)")
        print("  onset/s      ~= playback_speed  (one attack per row)")


if __name__ == "__main__":
    main()
