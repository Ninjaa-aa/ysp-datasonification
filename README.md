# Borehole Sonification Toolkit

**BMSIS YSP — Sounds of Deep Ice Fluorescence**

A generic, open-source Python toolkit that sonifies multi-channel tabular datasets and generates synchronized visual animations. The 32-channel ice borehole fluorescence dataset (WATSON instrument) is the primary example case, but the engine works on any CSV with band/channel-like columns.

---

## What It Does

This toolkit translates multi-channel depth-series or time-series data into both sound (sonification) and video (visualization). 

### 1. Audio Sonification (Phase 1)
- **Additive Synthesis**: Synthesizes a composite multi-tone waveform where each data channel controls the amplitude of a dedicated sine wave oscillator.
- **Dynamic Range Compression**: Supports `linear`, `log10`, and `ln` intensity scaling modes to bring out quiet background features without clipping loud peaks.
- **Flexible Frequency Mapping**:
  - `index` mode: Distributes frequencies logarithmically based on channel index.
  - `wavelength` mode: Map frequencies logarithmically based on actual physical wavelengths (e.g., UV/visible spectrometer bands).
- **Phase-Continuous Transitions**: Smoothly carries oscillator phases across row boundaries with raised-cosine crossfading to prevent acoustic clicks.
- **Per-Channel Normalization**: Independently normalizes each channel's range to ensure quiet bands remain audible at their peak locations.

### 2. Synchronized Visualization & Video Export (Phase 2)
- **Independent Visual Scaling**: Supports independent scaling (`linear`, `log10`, `ln`) for visual representation, decoupling visual intensity from audio volume.
- **Display Modes**:
  - `dots`: Horizontal line of circles where radius and brightness scale with channel intensity.
  - `circles`: Nested concentric rings representing channel intensities.
- **Flexible Styling**: Custom colormaps (via Matplotlib) and optional channel/wavelength label overlays.
- **Live Playback Window**: Optional live-rendering window synchronized with audio playback through local speakers.
- **Muxed Video Export**: Renders frames and merges them with the generated WAV audio track into a single `.mp4` or `.avi` video using OpenCV and MoviePy.

---

## Installation

Ensure you have Python 3.12 installed.

```bash
# Set up a virtual environment (using standard venv)
python -m venv .venv
.venv\Scripts\activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> [!NOTE]
> The heavy video dependencies (`opencv-python` and `moviepy`) are imported lazily. If you do not install them, the audio-only pipeline will still function.

---

## How to Run

The main entry point is `scripts/run_sonify.py`.

### 1. Audio-Only Examples

```bash
# Play sonification directly through speakers
python scripts/run_sonify.py --yes

# Export sonification to a WAV file
python scripts/run_sonify.py --yes --output outputs/borehole_audio.wav

# Run with wavelength-based frequency mapping and linear scaling
python scripts/run_sonify.py --yes --freq-mode wavelength --scale linear --output outputs/linear_wavelength.wav
```

### 2. Video & Live Display Examples

```bash
# Export a full video (.mp4) with default dots visualization and synchronized audio
python scripts/run_sonify.py --yes --video-output outputs/borehole_dots.mp4

# Export a video using circles mode, plasma colormap, and channel/wavelength labels
python scripts/run_sonify.py --yes --visual-mode circles --colormap plasma --show-labels --video-output outputs/borehole_circles.mp4

# Play audio through speakers and show a live, real-time animated display window
python scripts/run_sonify.py --yes --live-display
```

---

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Core Options** | | |
| `--input` | example dataset | Path to the input CSV file |
| `--row-start` | `None` (start) | First row index to include (0-based) |
| `--row-end` | `None` (end) | Last row index to include (exclusive) |
| `--n-bins` | detected count | Number of output frequency bins (channels to rebin to) |
| `--yes` | `False` | Skip interactive band-confirmation prompt |
| **Audio Options** | | |
| `--min-freq` | `150.0` | Lowest tone frequency in Hz |
| `--max-freq` | `2500.0` | Highest tone frequency in Hz |
| `--playback-speed` | `10.0` | Playback speed in rows per second |
| `--volume` | `0.8` | Master gain volume (0.0 to 1.0) |
| `--scale` | `log10` | Intensity scaling mode for audio: `linear`, `log10`, `ln` |
| `--freq-mode` | `index` | Frequency mapping mode: `index` or `wavelength` |
| `--wavelength-path`| reference table | Path to band-to-wavelength mapping CSV |
| `--sample-rate` | `44100` | Audio sample rate in Hz |
| `--output` | `None` (speakers) | Output .wav file path |
| **Visual Options** | | |
| `--visual-mode` | `dots` | Visual display mode: `dots`, `circles` |
| `--visual-scale` | `log10` | Intensity scaling mode for visuals: `linear`, `log10`, `ln` |
| `--colormap` | `plasma` | Matplotlib colormap name (e.g., `viridis`, `plasma`, `inferno`) |
| `--show-labels` | `False` | Show channel index or wavelength labels below elements |
| `--video-output` | `None` | Path to export the final .mp4 or .avi video |
| `--live-display` | `False` | Show a live matplotlib animation window during speaker playback |
| `--video-title` | "Sounds..." | Title text displayed at the top of the video frames |
| `--frame-width` | `1280` | Frame width in pixels |
| `--frame-height` | `720` | Frame height in pixels |

---

## Physical & Psychoacoustic Tuning Decisions

Below are the recommended defaults for analyzing the 4000-row borehole dataset:

- **Playback Speed (`10` rows/second)**: At 10 rows per second, each data row is mapped to a 100 ms acoustic segment. This allows the human ear to comfortably resolve both pitch (frequency) and relative volumes (timbre) of the channels. The full 4000-row scan is condensed into a digestible 6 minutes and 40 seconds.
- **Intensity Scale (`log10`)**: Tabular fluorescence values exhibit highly localized peaks with quiet background baseline fluctuations. Linear scaling forces the peaks to dominate, rendering baseline variations silent. Logarithmic scaling compresses the dynamic range, rendering subtle structural changes in the baseline clearly audible.
- **Frequency Mode (`wavelength`)**: Rather than arbitrary index-based spacing, this mode maps channels to log-spaced frequencies according to their actual physical wavelengths (275 nm to 446 nm). Longer wavelengths (near-UV/visible) map to higher pitches, and shorter wavelengths (deep UV) map to lower pitches, translating the physical spectrum directly to the audible range.

---

## Running Tests

Verify the installation and engine correctness by running the test suite:

```bash
py -m pytest tests/ -v
```

---

## Project Structure

```
sonify/              # Generic, reusable, dataset-agnostic engine
  config.py          # SonificationConfig dataclass and validations
  data_io.py         # CSV loader
  band_detect.py     # Auto-detect band/channel columns
  preprocess.py      # Clean, sort, rebin
  mapping.py         # Scale, normalize, frequency assignment
  synth.py           # Additive synthesis engine with phase continuity
  playback.py        # Speaker playback engine
  export.py          # WAV file export
  visualize.py       # Frame rendering and live display window
  video_export.py    # Silent video writer and audio muxer
scripts/
  run_sonify.py      # CLI entry point (dataset-specific defaults)
tests/               # Comprehensive unit tests
data/raw/            # Raw datasets
data/reference/      # Wavelength reference table
outputs/             # Generated .wav and .mp4 files
```
