# Borehole Sonification Toolkit — Phase 1 Implementation Spec

**Project:** BMSIS YSP — Sounds of Deep Ice Fluorescence (Ice Borehole Sonification)
**Mentor:** Dr. Michael Malaska
**This phase:** Phase 1 of 6 — Quick standalone sonification script, V0
**Target runtime:** Python 3.10+, fully offline, no paid services, no GPU required

---

## 1. Context (read this before writing code)

The end goal of the overall YSP project is **not** a one-off sonification of one dataset.
It is an **open-source, general-purpose Python toolkit** that can sonify *any* multi-channel
tabular dataset (sensor arrays, spectral scans, time series with many columns, etc.). The
32-channel ice borehole fluorescence dataset is the **example/test case** used to build and
validate the tool — it is not a constant baked into the code.

This matters for implementation: **do not hardcode** `Band_1_bc ... Band_32_bc`, 32 channels,
or this file's specific column names anywhere in the core library. Auto-detect everything from
the CSV header. The CLI script that runs against the example dataset can have dataset-specific
defaults, but the underlying engine must work on a CSV with a different number of channels and
different column names.

### What the data physically is

Each row = one depth point as the instrument descended down the borehole. At each depth, the
instrument fired a DUV laser and recorded fluorescence intensity in 32 spectral bands
(channels), already background-subtracted and instrument-calibrated. So the table is structurally
a **spectrogram**: rows = time/depth axis, columns = frequency-like axis (channels), cell values =
intensity. That framing should drive the synthesis design (see §4).

### Example dataset (for testing only — must not be hardcoded)

`2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv`

Relevant columns:
- `row_num`, `depth`, `rot`, `timestamp` — housekeeping / position columns
- `Band_1_bc` … `Band_32_bc` — fluorescence intensity per channel, background-corrected (**these are the values to sonify**)
- `Band_1_std1_MAX_SDT` … `Band_32_std1_MAX_SDT` — max standard deviation per channel (noise estimate; **not used in Phase 1**, reserved for later signal-processing phases)
- `row_bc`, `depth_bc`, `rot_bc` — cross-check duplicates of `row_num`/`depth`/`rot`, ignore

A companion table maps band index → wavelength (nm), band 1 = 275 nm through band 32 = 446 nm,
roughly evenly spaced (~5.5 nm steps), deep UV → near-UV/violet. This file should be loaded
optionally (see §6.4) to allow wavelength-based frequency mapping instead of plain index-based
mapping, since the real physical mapping is the more scientifically defensible default.

---

## 2. Phase 1 deliverable, exactly as specified by the mentor

> Take 1 borehole worth of data (or small section). Create tool to read in file as .csv.
> User variables: volume, number of bins (columns) (allow rebinning), tone range, playback
> speed (rows per second). Tone intensity is based on value intensity. Figure out correct
> timing to make sound OK. Figure out tone range and set to make sound OK. Enable switches
> for log intensity scale (10, e), or linear scale. Identify spectral bands (ask user if
> correct). Assign frequencies in hearing range to each band.
> BONUS (next version): Output is a .wav file.

Translate that into concrete behavior:

| Requirement | Concrete behavior |
|---|---|
| Read CSV | Generic CSV loader, works on any tabular file |
| Identify spectral bands, confirm with user | Auto-detect band-like columns via regex, print detected list + count, ask for y/n confirmation (or accept via `--yes` flag for non-interactive runs) |
| Number of bins (rebinning) | User can request fewer output channels than detected channels; adjacent channels are merged (mean) |
| Tone range | User sets `--min-freq` / `--max-freq` in Hz; all channel tones live inside this window |
| Playback speed | User sets rows-per-second; this sets duration of each row's audio segment |
| Volume | Single master gain `--volume`, 0.0–1.0, applied after synthesis, before final normalization |
| Tone intensity from value intensity | Per-channel amplitude in a row's segment scales with that channel's (scaled) value |
| Log/linear intensity switch | `--scale {linear,log10,ln}` |
| Assign frequencies to bands | Implemented in `mapping.py`, log-spaced across the tone range (see §4.2) |
| BONUS .wav export | Build now so it's nearly free: synthesize to an in-memory float array first, write to speakers OR `.wav` from the same array — don't make playback the only output path |

---

## 3. Folder structure

Create exactly this structure. It separates the **generic reusable engine** (`sonify/`) from
the **dataset-specific runner** (`scripts/`), which is what keeps this extensible into Phase 2+
without a rewrite.

```
sonification-toolkit/
├── README.md                      # what this is, how to run it, examples
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── 2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv
│   └── reference/
│       └── watson_band_wavelengths.csv   # Band No. -> wavelength center (nm), from Malaska et al.
├── sonify/                        # generic, reusable, dataset-agnostic engine
│   ├── __init__.py
│   ├── config.py                  # SonificationConfig dataclass: all user-tunable params + validation
│   ├── data_io.py                 # generic CSV loading
│   ├── band_detect.py             # auto-detect channel/band columns, interactive confirmation
│   ├── preprocess.py              # cleaning (NaN/negative handling), rebinning
│   ├── mapping.py                 # value scaling (linear/log10/ln) + frequency assignment
│   ├── synth.py                   # additive synthesis engine, anti-click windowing, normalization
│   ├── playback.py                # speaker playback (sounddevice)
│   └── export.py                  # .wav export (scipy.io.wavfile) — bonus feature, build the hook now
├── scripts/
│   └── run_sonify.py              # CLI entry point specific to the borehole example dataset
├── tests/
│   ├── test_band_detect.py
│   ├── test_mapping.py
│   ├── test_preprocess.py
│   └── test_synth.py
└── outputs/
    └── (generated .wav files land here, gitignored except .gitkeep)
```

---

## 4. Core design decisions (the part that actually matters)

### 4.1 Row → time

```
seconds_per_row = 1.0 / playback_speed_rows_per_second
segment_samples = round(seconds_per_row * sample_rate)
```
Each row becomes one fixed-length audio segment. Segments are concatenated **in row order**,
not in arbitrary file order — sort by `row_num` (or `depth`, descending) before synthesis, since
`row_num` increments by 2 (only laser-hot reads are kept) and the file is not guaranteed sorted.

Do not map true elapsed depth-distance to time in Phase 1 — the spec explicitly asks for
"rows per second," not "meters per second." Keep it simple; depth-proportional timing can be a
Phase 2+ option.

### 4.2 Band → frequency

Use **log-spaced** frequency assignment across `[min_freq, max_freq]`, because human pitch
perception is logarithmic (equal-sounding steps require log spacing, not linear Hz steps — this
is also why musical scales are built this way). For `n` output channels (after rebinning):

```
freqs[i] = min_freq * (max_freq / min_freq) ** (i / (n - 1))   for i in 0..n-1
```

Support two modes, selectable via config:
- `index` mode (default, always available): channels spaced evenly in log-frequency by index, as above.
- `wavelength` mode (optional, only if `data/reference/watson_band_wavelengths.csv` is supplied and matches the detected bands): map each band's actual wavelength (nm) into the same log-frequency window via linear interpolation on wavelength, then log-placement in Hz. This is the scientifically defensible mapping when wavelength data is available, and should be exposed as `--freq-mode {index,wavelength}`.

Sensible default tone range: `min_freq=150`, `max_freq=2500` Hz — comfortably within normal
hearing, avoiding the most fatiguing high end and the muddy sub-150Hz region. Expose as CLI args
so the agent (and later users) can retune by ear; do not treat these defaults as fixed truth.

### 4.3 Value → amplitude (per channel, per row)

1. Clean: replace NaN with 0; for log scaling, clip negative values to 0 first (background
   subtraction can produce small negative noise) and add a small epsilon before taking the log
   so `log(0)` never occurs.
2. Scale, per the `--scale` switch:
   - `linear`: value used as-is after clipping
   - `log10`: `log10(value + epsilon)`
   - `ln`: `log(value + epsilon)`
3. Normalize each channel's scaled values to `[0, 1]` using the **min/max across the whole
   loaded slice** (not per-row), so relative intensity between rows is preserved. Per-row
   normalization would destroy exactly the signal you're trying to hear.
4. This normalized `[0,1]` value is the **amplitude** of that channel's sine tone within that
   row's segment. A channel that is silent (~0) the whole way down should be near-silent in the
   output; a channel with a sharp spike should produce an audible swell.

### 4.4 Rebinning (channel count reduction)

If the user requests `n_bins < n_detected_channels`, group adjacent channels into `n_bins`
contiguous groups (as close to equal-width as possible) and **average** (not sum, to keep scale
consistent) the values within each group, per row, before the scaling step in §4.3. Frequency
assignment (§4.2) then runs on the *rebinned* channel count, not the original.

### 4.5 Synthesis — avoiding clicks (this will be the first thing that sounds "wrong")

Naively concatenating independent fixed-length sine segments produces audible clicks/pops at
segment boundaries because each segment's sine starts at phase 0 and the previous one didn't
necessarily end at a zero-crossing, and amplitude can jump discontinuously between rows.

Required mitigation, both of these:
- **Phase continuity**: track running phase per channel/oscillator across segment boundaries
  rather than restarting each segment's sine at `t=0`. Generate each segment as
  `amplitude * sin(2*pi*freq*t + phase_offset)` where `phase_offset` carries over from the end
  of the previous segment for that same channel.
- **Short amplitude envelope**: apply a short (5–15 ms) linear or raised-cosine fade-in/fade-out
  at each segment boundary per channel as a second safety net against amplitude discontinuities,
  even with phase continuity.

Sum all channel oscillators per segment (additive synthesis), then concatenate segments in
order to build the full waveform.

### 4.6 Final gain staging

```
full_waveform *= volume               # user master gain, 0.0-1.0
full_waveform /= max(abs(full_waveform)) # peak-normalize so it never clips, guard divide-by-zero
```

### 4.7 Output paths

Build the engine so the synthesized waveform is a single in-memory numpy float array
(`sample_rate`, mono) returned by `synth.py`. Both `playback.py` (live speaker playback via
`sounddevice`) and `export.py` (`.wav` via `scipy.io.wavfile.write`) consume that same array.
This makes the bonus .wav-export feature essentially free rather than a separate code path.

---

## 5. CLI parameters (Phase 1 surface)

```
python scripts/run_sonify.py \
    --input data/raw/2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv \
    --row-start 0 --row-end 200          # optional slice of the file, default = whole file
    --n-bins 32                          # default = however many channels were detected
    --min-freq 150 --max-freq 2500
    --playback-speed 10                  # rows per second
    --volume 0.8
    --scale log10                        # linear | log10 | ln
    --freq-mode index                    # index | wavelength
    --sample-rate 44100
    --output outputs/borehole_phase1.wav # omit to just play through speakers
    --yes                                # skip interactive band-confirmation prompt
```

All of these are also fields on `SonificationConfig` in `config.py` with validation (e.g.
`max_freq > min_freq`, `playback_speed > 0`, `0.0 <= volume <= 1.0`, `n_bins >= 1`).

---

## 6. Module-by-module responsibilities

### 6.1 `sonify/data_io.py`
- `load_csv(path) -> pandas.DataFrame`
- Generic. No knowledge of band naming.

### 6.2 `sonify/band_detect.py`
- `detect_band_columns(df) -> list[str]`: regex-match columns like `Band_<int>_bc` (case
  insensitive, but generalized enough to also match plausible variants like `Channel_3` or
  `Band3` on a different dataset — don't over-fit to this one file's exact naming).
- Explicitly **exclude** columns containing `std`, `sdt`, `max`, or other housekeeping markers
  from being treated as data channels.
- `confirm_with_user(detected_columns) -> bool`: print the count and column names, prompt
  `Proceed with these N channels? [y/n]`, honor a `--yes` CLI flag to bypass for automated runs/tests.

### 6.3 `sonify/preprocess.py`
- `clean(df, band_cols) -> np.ndarray`: NaN→0, negative-value clipping for log-safety.
- `rebin(matrix, n_bins) -> np.ndarray`: contiguous-group averaging per §4.4.
- `sort_by_row_order(df) -> df`: ensure monotonic order per §4.1.

### 6.4 `sonify/mapping.py`
- `scale_values(matrix, mode: Literal["linear","log10","ln"]) -> np.ndarray`
- `normalize_global(matrix) -> np.ndarray` (whole-slice min/max → [0,1], per §4.3 step 3)
- `assign_frequencies(n_channels, min_freq, max_freq, mode="index", wavelengths=None) -> np.ndarray`
- `load_wavelength_table(path) -> dict[int, float]` (optional, only used in wavelength mode)

### 6.5 `sonify/synth.py`
- `synthesize(amplitude_matrix, freqs, seconds_per_row, sample_rate) -> np.ndarray`
  Implements phase-continuous additive synthesis with boundary fades, per §4.5–4.6.

### 6.6 `sonify/playback.py`
- `play(waveform, sample_rate)` via `sounddevice` (fallback to `simpleaudio` if `sounddevice`
  is unavailable on the agent's environment — try/except import, document whichever is used).

### 6.7 `sonify/export.py`
- `export_wav(waveform, sample_rate, path)` via `scipy.io.wavfile.write`. Convert float
  `[-1, 1]` to `int16` correctly (`* 32767`, clip, cast) — don't write raw floats.

### 6.8 `sonify/config.py`
- `SonificationConfig` dataclass holding every parameter in §5 with type hints and a
  `validate()` method raising clear `ValueError`s on bad input.

### 6.9 `scripts/run_sonify.py`
- `argparse` CLI wiring all of the above together in the pipeline order: load → detect/confirm
  → slice rows → clean/sort → rebin → scale → normalize → assign frequencies → synthesize →
  apply volume/normalize → play and/or export.

---

## 7. Required libraries

```
pandas
numpy
scipy
sounddevice      # primary playback; simpleaudio as a documented fallback if it fails to install
```

No MIDI library is needed for Phase 1 (MIDI quantizes to fixed notes; this spec calls for
continuous frequency mapping driven directly by data values, which additive sine synthesis
handles more faithfully than MIDI note mapping would).

---

## 8. Test checklist (what "done" looks like)

Build `tests/` against a small synthetic CSV (don't require the real dataset for unit tests —
generate a tiny fixture with 4 fake channels and 20 rows in the test file itself), covering:

- [ ] Band detection finds the right columns and correctly excludes `_std1_MAX_SDT` columns
- [ ] Rebinning from N channels to M < N channels produces exactly M output columns with
      plausible averaged values
- [ ] `linear`, `log10`, and `ln` scaling modes all run without error on data containing zeros
      and small negative values
- [ ] Frequency assignment returns `n_bins` frequencies, all within `[min_freq, max_freq]`,
      strictly increasing, log-spaced (verify ratio between consecutive freqs is constant)
- [ ] Synthesized waveform length matches `n_rows * seconds_per_row * sample_rate` (within
      rounding)
- [ ] Synthesized waveform has no sample exceeding `[-1, 1]` after final normalization
- [ ] `.wav` export produces a valid file `scipy.io.wavfile.read` can load back, matching the
      expected sample rate
- [ ] Manual listening check on the real example dataset: no audible clicking between rows at
      the default playback speed (this one is a human-ears check, not an automated test —
      flag it explicitly as a manual QA step in the README)

## 9. Manual tuning step (cannot be skipped)

After the pipeline runs end-to-end, the spec explicitly asks to "figure out correct timing" and
"figure out tone range... to make sound OK." This is a listening-based tuning pass, not something
solvable purely analytically:

1. Run on the full example file at a few `--playback-speed` values (e.g. 5, 10, 20, 40 rows/sec)
   and note which range sounds like distinct evolving tones rather than a stream of clicks or an
   uncomfortably slow drone.
2. Try both `--freq-mode index` and `--freq-mode wavelength` and compare.
3. Try all three `--scale` options on the same slice and note how much more structure becomes
   audible under `log10`/`ln` versus `linear`, given this dataset's intensity range (values from
   ~0 up to several hundred in places).
4. Record the chosen defaults and the reasoning in the README so Phase 2 inherits informed
   starting points instead of arbitrary ones.

## 10. Explicitly out of scope for Phase 1 (don't build yet)

- Using the `_std1_MAX_SDT` uncertainty columns for anything (reserved, per the mentor's notes, for "detailed signal processing" in a later phase)
- Depth-proportional (rather than row-count-proportional) timing
- Stereo / spatial placement of channels
- Any GUI — CLI only for Phase 1
- MIDI/musicxml export — wav only, per the bonus note

---

## 11. References consulted for synthesis design choices

- Malaska et al., *Astrobiology* — WATSON instrument band/wavelength correspondence table (the wavelength reference table for §4.2 wavelength mode)
- STRAUSS (Sonification Tools and Resources for Analysis Using Sound Synthesis), Trayford & Harrison — general-purpose Python sonification package design patterns (modular mapping of data → sound parameters)
- Standard additive-synthesis formulation: S(t) = Σᵢ aᵢ(t)·sin(2πfᵢ(t)·t + φᵢ(t)) — basis for §4.5
- General data-sonification tutorials (e.g. Matt Russo / SYSTEM Sounds "Sonification 101") confirming the standard pipeline: understand data → define scale/mapping functions → convert to audio parameters → synthesize
