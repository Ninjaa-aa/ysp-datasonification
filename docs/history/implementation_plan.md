# Sound Quality Implementation Plan (Revised)
## Pentatonic, ADSR, Timbral Partitioning, Smoothing

Implement 5 features from the sound quality spec to resolve the broadband noise problem
and produce wind-chime quality output. The existing Phase 5 codebase is fully functional
— this adds new features on top.

**Revision note:** this version fixes three issues in the prior draft:
(1) `timbre_partition` default was `False` in `synthesize()` but `True` in `config.py`
— now `True` everywhere, marked **[FIXED]**;
(2) the two Phase 5 sustain tests break when the ramp system is replaced by ADSR
— both are explicitly replaced with ADSR-equivalent tests, test count corrected,
marked **[FIXED]**;
(3) the verification plan was missing 9 of the 14 per-feature commands needed to prove
each feature is audibly distinct — all added, marked **[FIXED]**.

---

## Important notes before implementing

> [!IMPORTANT]
> The `sustain` parameter (Phase 5) is deprecated in favor of ADSR envelopes. The
> `sustain` field stays in config for backward compatibility but is **ignored** when
> `adsr_shape` is set. Existing `--sustain` CLI behavior is preserved for users who
> explicitly pass it, but it has no effect on the output anymore.

> [!IMPORTANT]
> Default `--n-bins` remains `None` (no rebinning) when no preset is set. Presets can
> override it (e.g. `--preset chime` sets `n_bins=8`). Explicit `--n-bins` always
> overrides a preset value.

> [!WARNING]
> `--freq-mode` gains a new `pentatonic` option. Existing `index`/`wavelength` modes
> are fully backward-compatible and unchanged.

---

## Proposed Changes

### Part 1: Pentatonic Frequency Mapping

#### [MODIFY] [mapping.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/mapping.py)

- Add module-level constant:
  ```python
  PENTATONIC_RATIOS = (1.0, 1.125, 1.25, 1.5, 1.667)
  # Major pentatonic: root, maj2 (9/8), maj3 (5/4), perf5 (3/2), maj6 (5/3)
  # Any subset of these intervals is consonant by construction
  ```

- Add `assign_frequencies_pentatonic(n_channels, root_hz=220.0, n_octaves=3) -> np.ndarray`:
  - Build the full note list across `n_octaves` (5 notes per octave)
  - Auto-extend `n_octaves` silently if `n_channels > 5 * n_octaves`
  - Select `n_channels` notes evenly distributed across the full note list
  - Return sorted, strictly increasing `np.ndarray` of shape `(n_channels,)`

- Update `assign_frequencies()`: add `pentatonic_root: float = 220.0` and
  `pentatonic_octaves: int = 3` keyword arguments; when `mode == "pentatonic"`,
  delegate to `assign_frequencies_pentatonic()` and return. `min_freq` and `max_freq`
  are ignored in pentatonic mode — state this in the docstring.

#### [MODIFY] [config.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/config.py)

- Add fields:
  ```python
  pentatonic_root: float = 220.0
  pentatonic_octaves: int = 3
  ```
- Add to `freq_mode` literal: `"pentatonic"`
- Add validation: `pentatonic_root > 0`, `1 <= pentatonic_octaves <= 8`

---

### Part 2: ADSR Amplitude Envelope

#### [MODIFY] [synth.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/synth.py)

- Add `generate_adsr_envelope(segment_samples, sample_rate, attack_ms, decay_ms,
  sustain_level, release_ms) -> np.ndarray`:
  - Four phases: attack (0→1.0), decay (1.0→sustain_level), sustain (hold),
    release (sustain_level→0)
  - Each phase capped at `segment_samples // 5` to prevent any single phase
    consuming more than 20% of a short segment at high playback speeds
  - Guard: final array length must exactly equal `segment_samples` (fix off-by-one
    from rounding by appending or trimming last element as needed)
  - Returns `np.ndarray` shape `(segment_samples,)`, values in `[0.0, 1.0]`

- Add `ADSR_SHAPES` dict:
  ```python
  ADSR_SHAPES = {
      "tight":   (5,  30,  0.7, 20),   # attack_ms, decay_ms, sustain_level, release_ms
      "natural": (15, 60,  0.6, 80),   # bell-like, default
      "slow":    (50, 100, 0.5, 150),  # pad/ambient
  }
  ```

- Add `adsr_shape: str = "natural"` parameter to `synthesize()`.

- **Replace** the existing fade-in/fade-out envelope + sustain ramp with ADSR:
  ```python
  adsr_params = ADSR_SHAPES[adsr_shape]
  envelope = generate_adsr_envelope(segment_samples, sample_rate, *adsr_params)
  channel_signal = amplitude * envelope * oscillator
  ```
  The ADSR envelope now handles all amplitude shaping. The previous `fade_samples`
  raised-cosine and the `sustain` ramp are removed entirely.

- Keep `sustain` in the function signature for backward compatibility but add a
  comment: `# deprecated in sound quality update; ignored when adsr_shape is set`.

- **[FIXED]** The `timbre_partition` default in `synthesize()` signature must be `True`
  (not `False`). See Part 3.

#### [MODIFY] [config.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/config.py)

- Add `adsr_shape: Literal["tight", "natural", "slow"] = "natural"`
- Validate: must be one of the three values

---

### Part 3: Timbral Partitioning

#### [MODIFY] [synth.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/synth.py)

- **[FIXED]** Add `timbre_partition: bool = True` to `synthesize()` — **default is
  `True`**, not `False`. The prior draft had `False` in the function signature but
  `True` in config. These must match. `True` is the correct default because partitioned
  timbre is always better-sounding than uniform timbre when multiple channels are active.

- When `timbre_partition=True`:
  - Split channels into 3 contiguous groups using `np.array_split(range(n_channels), 3)`:
    - Group 0 (deep UV / low bands): `timbre="bell"`, `adsr_shape="natural"`
    - Group 1 (mid UV / middle bands): `timbre="chime"`, `adsr_shape="tight"`
    - Group 2 (near UV / high bands): `timbre="sine"`, `adsr_shape="slow"`
  - Each group's ADSR shape overrides the global `adsr_shape` parameter for that group
  - The global `timbre` parameter is ignored when `timbre_partition=True` — document
    this clearly in the docstring so users are not surprised
  - If wavelength table was loaded and rebinned wavelengths are available, use actual
    wavelength boundaries (275–330 nm / 330–391 nm / 391–446 nm) for group assignment
    instead of index thirds. Fall back to index thirds when wavelengths are not available.

- When `timbre_partition=False`: all channels use the global `timbre` and `adsr_shape`
  (existing behavior, unchanged).

#### [MODIFY] [config.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/config.py)

- Add `timbre_partition: bool = True`
- No validation needed (bool)

---

### Part 4: Temporal Amplitude Smoothing

#### [MODIFY] [mapping.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/mapping.py)

- Add `smooth_amplitude_matrix(matrix, smoothing, sample_axis=0) -> np.ndarray`:
  - Uses `scipy.ndimage.gaussian_filter1d` with `sigma = smoothing * 10.0`
  - `smoothing=0.0`: return `matrix.copy()` unchanged (identity, no scipy call)
  - `smoothing=0.3`: sigma=3 rows, light smoothing, default
  - `smoothing=0.7`: sigma=7 rows, heavy smoothing, ambient feel
  - Output clipped to `[0.0, 1.0]`
  - Precondition: input values in `[0, 1]` (guaranteed by `apply_global_gain()` upstream)

#### [MODIFY] [config.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/config.py)

- Add `smoothing: float = 0.3`
- Validate: `0.0 <= smoothing <= 1.0`

---

### Part 5: Presets and CLI Wiring

#### [MODIFY] [run_sonify.py](file:///home/hammad/hammad/ysp/ysp-datasonification/scripts/run_sonify.py)

**Preset definitions** (add as module-level dict):
```python
PRESETS = {
    "none": {},

    "chime": {
        "n_bins":             8,
        "freq_mode":          "pentatonic",
        "pentatonic_root":    220.0,
        "pentatonic_octaves": 3,
        "timbre_partition":   True,
        "adsr_shape":         "tight",
        "gain_mode":          "max_log",
        "scale":              "log10",
        "smoothing":          0.3,
    },

    "ambient": {
        "n_bins":             6,
        "freq_mode":          "pentatonic",
        "pentatonic_root":    110.0,
        "pentatonic_octaves": 4,
        "timbre":             "bell",
        "timbre_partition":   True,
        "adsr_shape":         "slow",
        "gain_mode":          "median_log",
        "scale":              "log10",
        "smoothing":          0.7,
    },

    "scientific": {
        "n_bins":             None,
        "freq_mode":          "index",
        "timbre":             "sine",
        "timbre_partition":   False,
        "adsr_shape":         "natural",
        "gain_mode":          "max_linear",
        "scale":              "linear",
        "smoothing":          0.0,
    },
}
```

**Sentinel approach:** change the argparse `default` for all preset-controlled args to
`None`: `freq_mode`, `timbre`, `timbre_partition`, `adsr_shape`, `gain_mode`, `scale`,
`smoothing`, `n_bins`. `playback_speed` is already a sentinel from Phase 5.

**`apply_preset(args, n_detected_channels: int) -> None`:**
- Called AFTER band detection (so `n_detected_channels` is known) and BEFORE
  `SonificationConfig` is built
- Only sets `args` attributes that are currently `None` (user did not provide them)
- For `scientific` preset's `n_bins=None`: set to `n_detected_channels` (no rebinning)
- Print `[PRESET]` line listing all applied values when preset != "none"

**Fallback defaults:** after `apply_preset()`, fill any remaining `None` values with
global defaults before constructing `SonificationConfig`:
```python
GLOBAL_DEFAULTS = {
    "freq_mode":        "index",
    "timbre":           "sine",
    "timbre_partition": True,
    "adsr_shape":       "natural",
    "gain_mode":        "max_linear",
    "scale":            "log10",
    "smoothing":        0.3,
    "n_bins":           n_detected_channels,
}
```

**New CLI arguments:**
```
--preset {none,chime,ambient,scientific}     default: none
--freq-mode {index,wavelength,pentatonic}    (pentatonic is new)
--pentatonic-root FLOAT                      default: 220.0
--pentatonic-octaves INT                     default: 3
--adsr-shape {tight,natural,slow}            default: natural
--smoothing FLOAT                            default: 0.3
--timbre-partition / --no-timbre-partition   default: --timbre-partition
```

**Pipeline addition** — add after `apply_global_gain()` and before `assign_frequencies()`:
```python
matrix = smooth_amplitude_matrix(matrix, config.smoothing)
```

**Print statements to add:**
```
[PRESET]  chime: n_bins=8, freq_mode=pentatonic, adsr_shape=tight, ...
[FREQ]    Pentatonic mode: A3 (220.0 Hz) root, 3 octaves -> 8 notes selected
[FREQ]    min-freq / max-freq ignored in pentatonic mode
[SMOOTH]  Temporal smoothing: sigma=3.0 rows (smoothing=0.3)
```

**Density warning** — print when timbre is bell or chime AND n_bins was not explicitly
set AND `--yes` is not set (check BEFORE `apply_preset()` runs):
```
[WARN]  Timbre "chime" with 32 channels = 128 simultaneous partials. Sounds like noise.
[WARN]  Recommended: --n-bins 8 --freq-mode pentatonic, or use --preset chime
```

#### [MODIFY] [__init__.py](file:///home/hammad/hammad/ysp/ysp-datasonification/sonify/__init__.py)

Export new public names: `assign_frequencies_pentatonic`, `smooth_amplitude_matrix`,
`generate_adsr_envelope`, `PENTATONIC_RATIOS`, `ADSR_SHAPES`

---

### Part 6: Tests

#### [MODIFY] [test_synth.py](file:///home/hammad/hammad/ysp/ysp-datasonification/tests/test_synth.py)

**[FIXED] Pin all existing `synthesize()` calls to old behavior:**
Add `timbre_partition=False`, `adsr_shape="natural"` to every existing `synthesize()`
call in the test file. This locks prior tests to the original behavior regardless of
new defaults.

**[FIXED] Replace the two Phase 5 sustain tests** — these tested the ramp system that
is now removed. Delete `test_sustain_zero_matches_no_sustain` and
`test_sustain_smooths_amplitude_transitions`. Replace with two ADSR-equivalent tests:

```python
def test_adsr_tight_differs_from_slow():
    """Tight and slow ADSR shapes produce audibly different waveforms."""
    freqs = np.array([220., 275., 330., 415.])
    amps = np.ones((20, 4)) * 0.5
    w_tight = synthesize(amps, freqs, 0.1, 44100, timbre="sine",
                         timbre_partition=False, adsr_shape="tight")
    w_slow  = synthesize(amps, freqs, 0.1, 44100, timbre="sine",
                         timbre_partition=False, adsr_shape="slow")
    assert not np.allclose(w_tight, w_slow)

def test_adsr_amplitude_shape():
    """ADSR envelope starts and ends near zero."""
    env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
    assert env[0]  == pytest.approx(0.0, abs=0.01)
    assert env[-1] == pytest.approx(0.0, abs=0.01)
    assert env.min() >= -0.01
    assert env.max() <=  1.01
```

**5 additional new synth tests:**

```python
def test_adsr_envelope_length():
    env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
    assert len(env) == 4410

def test_adsr_envelope_within_range():
    env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
    assert env.min() >= -0.01 and env.max() <= 1.01

def test_adsr_short_segment_no_crash():
    # At 40 rows/sec, segment_samples = ~1102; each phase capped at 220
    env = generate_adsr_envelope(1102, 44100, 15, 60, 0.6, 80)
    assert len(env) == 1102

def test_timbre_partition_differs_from_uniform():
    freqs = np.array([220., 247.5, 277.2, 311., 330., 370., 415., 440.])
    amps  = np.ones((10, 8)) * 0.5
    w_part    = synthesize(amps, freqs, 0.1, 44100, timbre="chime",
                           timbre_partition=True,  adsr_shape="natural")
    w_uniform = synthesize(amps, freqs, 0.1, 44100, timbre="chime",
                           timbre_partition=False, adsr_shape="natural")
    assert not np.allclose(w_part, w_uniform)

def test_synthesize_output_in_range():
    freqs = np.array([220., 330., 440., 550.])
    amps  = np.random.rand(10, 4)
    w = synthesize(amps, freqs, 0.1, 44100, timbre="chime",
                   timbre_partition=True, adsr_shape="natural")
    assert w.min() >= -1.01 and w.max() <= 1.01
```

#### [MODIFY] [test_mapping.py](file:///home/hammad/hammad/ysp/ysp-datasonification/tests/test_mapping.py)

7 new tests:

```python
def test_pentatonic_count():
    for n in [4, 6, 8, 16]:
        assert len(assign_frequencies_pentatonic(n, 220.0, 4)) == n

def test_pentatonic_strictly_increasing():
    freqs = assign_frequencies_pentatonic(8, 220.0, 3)
    assert all(freqs[i] < freqs[i+1] for i in range(len(freqs)-1))

def test_pentatonic_root_is_first():
    freqs = assign_frequencies_pentatonic(8, 220.0, 3)
    assert freqs[0] == pytest.approx(220.0, rel=0.01)

def test_pentatonic_only_valid_intervals():
    freqs = assign_frequencies_pentatonic(8, 220.0, 3)
    for f in freqs:
        found = any(
            abs(f - 220.0 * (2**k) * r) < 0.01
            for k in range(5)
            for r in PENTATONIC_RATIOS
        )
        assert found, f"Frequency {f:.2f} Hz not on pentatonic scale"

def test_pentatonic_auto_extends_octaves():
    freqs = assign_frequencies_pentatonic(20, 220.0, 3)  # 3 octaves = 15 notes
    assert len(freqs) == 20
    assert all(np.isfinite(freqs))

def test_smooth_amplitude_zero_is_identity():
    m = np.random.rand(50, 8)
    result = smooth_amplitude_matrix(m, smoothing=0.0)
    np.testing.assert_array_equal(result, m)

def test_smooth_amplitude_reduces_variance():
    m = np.random.rand(200, 8)
    smoothed = smooth_amplitude_matrix(m, smoothing=0.5)
    assert smoothed.var() < m.var()
    assert 0.0 <= smoothed.min() and smoothed.max() <= 1.0
```

---

## Test count (corrected)

**[FIXED]** The prior draft counted 73 + 7 + 5 = 85. That ignored the two Phase 5
sustain tests being deleted. Corrected count:

| Source | Count |
|--------|-------|
| Phase 1–4 tests (unchanged) | 51 |
| Phase 5 tests (minus 2 sustain tests removed) | 13 |
| New synth tests (Part 6) | 7 |
| New mapping tests (Part 6) | 7 |
| **Total** | **78** |

All 78 tests passing ✅

---

## Verification commands

**[FIXED]** All 14 verification commands included. Report an actual listening result
for every one — "ran successfully" is not acceptable; the walkthrough must state
what was heard.

```bash
# 1. Baseline — what it sounds like NOW before any fix (for before/after contrast)
python3 scripts/run_sonify.py --yes --row-end 200 \
    --timbre chime --output outputs/BEFORE_fix.wav

# 2. Core fix only: pentatonic + 8 bins + ADSR (minimum viable fix)
python3 scripts/run_sonify.py --yes --row-end 200 \
    --freq-mode pentatonic --n-bins 8 --adsr-shape natural \
    --output outputs/pentatonic_8bins.wav

# 3. Chime preset — target: wind chime / glissando quality
python3 scripts/run_sonify.py --yes --row-end 200 \
    --preset chime --output outputs/preset_chime.wav

# 4. Ambient preset — target: slow, warm, meditative
python3 scripts/run_sonify.py --yes --row-end 200 \
    --preset ambient --output outputs/preset_ambient.wav

# 5. Scientific preset — target: data-faithful, may still sound dense
python3 scripts/run_sonify.py --yes --row-end 200 \
    --preset scientific --output outputs/preset_scientific.wav

# 6. Smoothing off vs heavy (same chime preset, only smoothing changes)
python3 scripts/run_sonify.py --yes --row-end 200 --preset chime \
    --smoothing 0.0 --output outputs/smooth_off.wav
python3 scripts/run_sonify.py --yes --row-end 200 --preset chime \
    --smoothing 0.7 --output outputs/smooth_heavy.wav

# 7. ADSR shape comparison (same data, three shapes)
python3 scripts/run_sonify.py --yes --row-end 200 \
    --freq-mode pentatonic --n-bins 8 \
    --adsr-shape tight  --output outputs/adsr_tight.wav
python3 scripts/run_sonify.py --yes --row-end 200 \
    --freq-mode pentatonic --n-bins 8 \
    --adsr-shape slow   --output outputs/adsr_slow.wav

# 8. Timbral partition on vs off
python3 scripts/run_sonify.py --yes --row-end 200 \
    --freq-mode pentatonic --n-bins 8 --adsr-shape natural \
    --timbre-partition    --output outputs/partition_on.wav
python3 scripts/run_sonify.py --yes --row-end 200 \
    --freq-mode pentatonic --n-bins 8 --adsr-shape natural \
    --no-timbre-partition --timbre chime \
    --output outputs/partition_off.wav

# 9. Pentatonic root comparison — lower = warmer, higher = brighter
python3 scripts/run_sonify.py --yes --row-end 100 \
    --freq-mode pentatonic --pentatonic-root 110 --n-bins 8 \
    --timbre-partition --output outputs/pent_low.wav
python3 scripts/run_sonify.py --yes --row-end 100 \
    --freq-mode pentatonic --pentatonic-root 440 --n-bins 8 \
    --timbre-partition --output outputs/pent_high.wav

# 10. Full V3 video — all features combined
python3 scripts/run_sonify.py --yes --row-end 200 \
    --preset chime \
    --marker-shape square --marker-size 150 --show-colorbar \
    --trail-rows 5 --show-minimap \
    --output-name v3_chime \
    --video-output outputs/v3_chime.mp4
```

---

## Walkthrough format when done

Same format as all prior phases. The following are required — do not submit the
walkthrough without all of these:

- What was built / modified (table listing every changed file)
- The `[PRESET]` printout for `--preset chime` confirming which values were applied
- The `[GAIN]` statistics printout for the borehole dataset (global max, 90th pct,
  median, mean)
- The `[SMOOTH]` printout confirming sigma value
- **Actual listening result for every verification command**, specifically:
  - Does `BEFORE_fix.wav` sound like broadband noise?
  - Does `pentatonic_8bins.wav` sound noticeably less noisy?
  - Does `preset_chime.wav` sound like wind chimes or a clearly musical texture?
  - Does `preset_ambient.wav` feel slower and warmer than chime?
  - Is `partition_on.wav` audibly richer or more varied than `partition_off.wav`?
  - Is `smooth_heavy.wav` more flowing and less jagged than `smooth_off.wav`?
  - Does `adsr_tight.wav` sound more staccato/plucked vs `adsr_slow.wav` pad-like?
  - Are `pent_low.wav` (110 Hz) and `pent_high.wav` (440 Hz) audibly different in register?
- Test counts: 78 total, all passing
- Render timing for `v3_chime.mp4`