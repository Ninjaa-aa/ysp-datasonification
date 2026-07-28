# Sounds of Deep Ice Fluorescence

**BMSIS Young Scientist Program — Sonification Toolkit**

An open-source Python toolkit that translates multi-channel tabular datasets into sound (sonification) and synchronized video (visualization). Built for the BMSIS Young Scientist Program under the guidance of Dr. Michael Malaska (NASA JPL), the primary test case is a 32-channel UV fluorescence scan captured by the WATSON instrument down a Greenland ice borehole — but the engine is fully generic and works on any CSV with band/channel-like columns.

---

## Install

```bash
# Clone and enter the repository
git clone https://github.com/Ninjaa-aa/ysp-datasonification.git
cd ysp-datasonification

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> `opencv-python` and `moviepy` are only needed for video export. If you only want audio output, the core pipeline works without them.

---

## Quick Start

```bash
# Wind-chime sonification of the example borehole dataset
.venv/bin/python3 scripts/run_sonify.py --yes --preset chime \
    --row-end 200 --output outputs/quick_start.wav

# Event-driven: only significant fluorescence peaks sound, each pitched by
# its dominant emission wavelength
.venv/bin/python3 scripts/run_sonify.py --yes --preset event \
    --output outputs/event.wav

# With synchronized video
.venv/bin/python3 scripts/run_sonify.py --yes --preset chime --row-end 200 \
    --marker-shape square --marker-size 150 --show-colorbar \
    --show-minimap --output-name quick_start
```

The last command produces both `outputs/quick_start.wav` and `outputs/quick_start.mp4`.

Check the result objectively at any point:

```bash
.venv/bin/python3 scripts/analyze_audio.py outputs/quick_start.wav --compare
```

---

## Presets

| Preset | What it is for |
|---|---|
| `chime` | Wind-chime aesthetic on 8 pentatonic channels. The main listening preset. |
| `ambient` | Slow, meditative, low register, heavy smoothing — good for the full descent. |
| `scientific` | All channels, pure sines, linear scale. Fidelity over musicality. |
| `event` | Only supra-threshold peaks sound, each pitched by its λmax. See below. |

### Event mode

Implements Dr. Malaska's two-function design: a **trigger** decides *which* peaks
sound; the **intensity encoding** separately decides *how loud* they are.

Because the dataset is 58.6% exact zeros, sonifying every row mostly amplifies
background. His threshold study (`data/threshold/`) quantified the trade-off — a
row is a "hit" if any band exceeds the threshold:

| Threshold | Signals |
|-----------|---------|
| 100       | 1498    |
| 300       | 75      |
| 500       | 34      |
| 700       | 19      |
| 900       | 10      |
| 1000      | 8       |

with the note *"How many tones would we like to hear? Probably above 10."*
`--target-tones` inverts this table for you:

```bash
# Ask for ~25 events; the threshold is solved from the data
.venv/bin/python3 scripts/run_sonify.py --yes --preset event --target-tones 25
```

---

## Full Command Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| **Core** | | | |
| `--input` | str | example dataset | Path to the input CSV file |
| `--row-start` | int | None (start) | First row index to include (0-based) |
| `--row-end` | int | None (end) | Last row index (exclusive) |
| `--n-bins` | int | detected count | Number of output frequency bins (rebins channels) |
| `--preset` | choice | none | `chime`, `ambient`, `scientific`, `event` |
| `--yes` | flag | False | Skip interactive band-confirmation prompt |
| **Trigger** — *which* peaks sound | | | |
| `--threshold` | float | 0.0 | Rows with no band above this stay silent (0 = off) |
| `--trigger-type` | choice | linear | Compare in `linear` or `log` domain |
| `--target-tones` | int | None | Solve `--threshold` from the data to yield ~N events |
| **Audio** — *how loud*, and what it sounds like | | | |
| `--min-freq` | float | 150.0 | Lowest tone frequency in Hz |
| `--max-freq` | float | 2500.0 | Highest tone frequency in Hz |
| `--playback-speed` | float | prompt/10.0 | Rows per second (prompts if omitted) |
| `--gain-mode` | choice | max_linear | Global normalization: `max_linear`, `max_log`, `pct90_linear`, etc. (8 modes) |
| `--timbre` | choice | chime | Synth voice: `sine`, `bell`, or `chime` |
| `--timbre-partition` / `--no-timbre-partition` | flag | True | Give each spectral third its own voice |
| `--adsr-shape` | choice | natural | Note envelope: `tight`, `natural`, `slow` |
| `--reverb-tail-ms` | float | 0.0 | Decaying tail so notes ring out (600–1500 suits event mode) |
| `--smoothing` | float | 0.3 | Temporal amplitude smoothing (0 = off) |
| `--scale` | choice | log10 | Intensity encoding: `linear`, `log10`, `ln` |
| `--freq-mode` | choice | index | Pitch layout: `index`, `wavelength`, `pentatonic` |
| `--pentatonic-root` | float | 220.0 | Root note in Hz for pentatonic mode |
| `--pentatonic-octaves` | int | 3 | Octaves spanned in pentatonic mode |
| `--wavelength-path` | str | reference table | Path to band → wavelength CSV |
| `--sample-rate` | int | 44100 | Audio sample rate in Hz |
| `--output` | str | None (speakers) | Output .wav file path |
| **Parameter Mapping** | | | |
| `--tone-source` | choice | band_index | Pitch driver: `band_index`, `wavelength`, `lambda_max`, `column` |
| `--tone-column` | str | None | Column name when `--tone-source column` |
| `--intensity-source` | choice | band_value | Volume driver: `band_value`, `column` |
| `--intensity-column` | str | None | Column name when `--intensity-source column` |
| **Visual** | | | |
| `--visual-mode` | choice | dots | Display style: `dots` or `circles` |
| `--visual-scale` | choice | log10 | Visual intensity scaling: `linear`, `log10`, `ln` |
| `--marker-shape` | choice | square | Shape of data points: `circle` or `square` |
| `--marker-size` | int | 120 | Size of markers in the plot |
| `--show-colorbar/--no-colorbar` | flag | True | Show or hide the intensity colorbar |
| `--colormap` | str | plasma | Matplotlib colormap name |
| `--show-labels` | flag | False | Show channel labels below each marker |
| `--video-output` | str | None | Output video path (.mp4 or .avi) |
| `--live-display` | flag | False | Show live animation during playback |
| `--video-title` | str | "Sounds..." | Title text in video frames |
| `--frame-width` | int | 1280 | Frame width in pixels |
| `--frame-height` | int | 720 | Frame height in pixels |
| `--trail-rows` | int | 5 | Trail rows visible simultaneously (1–20) |
| `--max-frames` | int | 500 | Max frames to render (safety cap) |
| `--show-minimap` | flag | False | Show overview minimap in video |
| **Output** | | | |
| `--output-name` | str | None | Base name → `outputs/NAME.wav` + `outputs/NAME.mp4` |
---

## Tuning Guide

Based on listening tests with the 4000-row borehole fluorescence dataset:

- **Playback Speed: `5-10` rows/sec for `chime`, `25` for `event`** — Each row maps to 100 ms, allowing comfortable pitch and timbre resolution. The full 4000-row scan compresses to ~6:40.
- **Gain Mode: `max_log`** — The global log modes (especially `max_log` and `pct90_log`) compress the dynamic range so quiet structure is audible without introducing the extreme noise amplification of the previous per-channel normalizations. The `max_linear` mode leaves the original dynamic range largely intact (often appearing mostly dark/silent except for major peaks).
- **Timbre: `chime`** — Phase-tracked inharmonic partials inspired by tubular bells provide a shimmering, clear tone that cuts through well at 10 rows/sec. `bell` (harmonic) is also available for a warmer sound.
- **Sustain: `0.3`** — Leaves a slight trailing echo on each channel so it doesn't sound entirely discrete, but recovers fast enough so rapid features don't blur.
- **Frequency Mode: `wavelength`** — Maps channels to log-spaced frequencies according to their physical wavelengths (275–446 nm). This translates the UV/visible spectrum directly into the audible range, making spectral shifts perceptually meaningful.

---

## Dataset Format

The toolkit expects a **CSV file** with:

1. **Band/channel columns** — Numeric columns with names matching one of these patterns:
   - `Band_1_bc`, `Band_2_bc`, ... (WATSON format)
   - `Channel_1`, `Channel_2`, ...
   - `Band_1`, `Band_2`, ...
   - `Band1`, `Band2`, ...
   - `Ch_1`, `Ch_2`, ...
   - `ch1`, `ch2`, ...

2. **Optional metadata columns** — `row_num` (for sorting), `depth` (for display), etc. These are auto-excluded from sonification.

3. **Noise/housekeeping columns** — Columns containing `std`, `sdt`, `max`, `min`, or `err` tokens are automatically excluded.

### Auto-detection

The engine scans all column names against the patterns above, sorts matches by their extracted numeric index, and presents them for confirmation. No hardcoded column count — works with 2 channels or 2048.

### Using with a non-borehole dataset

```bash
.venv/bin/python3 scripts/run_sonify.py --input your_data.csv --yes --output output.wav
```

If your columns don't match the recognized patterns, rename them to one of the supported formats (e.g., `Channel_1`, `Channel_2`, etc.).

---

## Project Structure

```
sonify/              # Generic, dataset-agnostic engine
  config.py          # SonificationConfig + ParameterMap dataclasses
  data_io.py         # CSV loader
  band_detect.py     # Auto-detect band columns + interactive confirmation UI
  preprocess.py      # Clean, sort, rebin
  events.py          # Trigger: which measurements sound (threshold, clusters)
  mapping.py         # Scale, normalize, frequency assignment, lambda-max
  synth.py           # Phase-continuous additive synthesis, ADSR, reverb tail
  quality.py         # Objective audio-quality metrics (regression guards)
  playback.py        # Speaker playback (sounddevice)
  export.py          # WAV file export
  visualize.py       # Trail-aware frame rendering (scatter + heatmap modes)
  video_export.py    # Video muxing (OpenCV + MoviePy)
scripts/
  run_sonify.py      # CLI entry point with structured logging
  analyze_audio.py   # Quality metrics + spectrograms for any WAV
docs/                # Investigations and design notes
tests/               # 183 unit tests
data/raw/            # Raw datasets
data/reference/      # Wavelength reference table
outputs/             # Generated .wav and .mp4 files
```

---

## Documentation

- [docs/roadmap.md](docs/roadmap.md) — status against every project-plan phase, plus blocked items
- [docs/frequency_investigation.md](docs/frequency_investigation.md) — where the sonified frequencies come from
- [CHANGELOG.md](CHANGELOG.md) — release history
- [docs/history/](docs/history/) — original phase specs and walkthroughs

---

## Running Tests

```bash
.venv/bin/python3 -m pytest tests/ -v
```

---

## Credits

### Team

- **Hammad Zahid** — Sonification engine, visualization, and CLI toolkit
- **Dayana** — Baseline subtraction and spectral peak characterization (companion toolkit)
- **Dr. Michael Malaska** — NASA Jet Propulsion Laboratory, project advisor

### Program & instrument

- **BMSIS Young Scientist Program** — Program framework and mentorship
- **WATSON Instrument** — UV fluorescence spectrometer; band wavelength data from:
  Eshelman, E., Daly, M.G., Slater, G., Bonaccorsi, R., & Pappalardo, R.T. (2019).
  *WATSON: a Wide-Angle Topographic Sensor for Organics at Night.*
  Astrobiology, 19(7), 885–905. DOI: [10.1089/ast.2018.1925](https://doi.org/10.1089/ast.2018.1925)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
