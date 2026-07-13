# Sound Quality Implementation Plan
## Pentatonic Scale, ADSR Envelopes, Timbral Partitioning, Amplitude Smoothing

Based on Dr. Malaska's feedback and a review of the data sonification literature
(Sonification Handbook, NASA SYSTEM Sounds methodology, psychoacoustics research on
simultaneous masking). This plan resolves the broadband noise problem completely and
brings the output to the quality level of professional sci-art sonification work.

**Revision note vs. the prior draft:** the prior plan correctly identified the
pentatonic fix and the preset system. This plan adds three additional improvements
grounded in the literature: proper ADSR amplitude envelopes (what makes synthetic
sound feel natural rather than electronic), timbral partitioning by spectral region
(the core technique NASA SYSTEM Sounds uses — different instruments for different
wavelength ranges), and temporal amplitude smoothing (suppresses row-to-row amplitude
jumps that contribute to the noise texture). All four are required together; the
pentatonic fix alone will improve things but the ADSR and partitioning are what get
it to "wind chimes."

---

## Step 0: Read the codebase before writing anything

Read these files in full before touching anything:

```
sonify/config.py
sonify/mapping.py
sonify/synth.py
scripts/run_sonify.py
tests/test_mapping.py
tests/test_synth.py
```

After reading, confirm in one paragraph: what `assign_frequencies()` currently does,
how `synthesize()` currently builds its amplitude envelope (the fade-in/fade-out +
sustain ramp), what `timbre` modes exist, and what `--n-bins` controls. Then proceed.

---

## The problem (understand this before implementing)

With 32 channels × 4 inharmonic chime partials, 128 simultaneous sine waves play
at arbitrary frequencies across 150–2500 Hz. The psychoacoustics literature is clear:
spectral crowding begins to occur as more tones with bigger footprints are sounded
simultaneously, making it more difficult to discern attributes of any one particular
sound. The result is perceived as broadband noise, not music.

Four independent problems compound this. All four must be fixed:

1. **Too many simultaneous tones** (32 channels → needs to default to 8 bins)
2. **Arbitrary non-musical frequencies** (log-spaced Hz → needs pentatonic constraint)
3. **All channels use identical timbre** (128 identical partials → needs partitioning)
4. **No proper note envelope** (flat amplitude → sounds electronic, not instrument-like)
5. **Jagged amplitude jumps between rows** (abrupt loudness changes → needs smoothing)

---

## Implementation order

Do these in sequence. Run all tests after each part before moving to the next.

```
Part 1: Pentatonic frequency mapping        (mapping.py)
Part 2: ADSR amplitude envelope             (synth.py)
Part 3: Timbral partitioning                (synth.py + config.py)
Part 4: Temporal amplitude smoothing        (mapping.py or preprocess.py)
Part 5: Presets and CLI wiring              (config.py + run_sonify.py)
Part 6: Tests
```

---

## Part 1: Pentatonic Frequency Mapping

### New function in `sonify/mapping.py`

```python
PENTATONIC_RATIOS = (1.0, 1.125, 1.25, 1.5, 1.667)
# Major pentatonic: root, major 2nd (9/8), major 3rd (5/4),
# perfect 5th (3/2), major 6th (5/3).
# Any subset of these intervals is consonant by construction —
# no dissonant minor 2nds or tritones exist in this set.

def assign_frequencies_pentatonic(
    n_channels: int,
    root_hz: float = 220.0,
    n_octaves: int = 3,
) -> np.ndarray:
    """Map n_channels to notes of a major pentatonic scale.

    Builds the full note list across n_octaves (5 notes per octave),
    then selects n_channels notes distributed evenly across that list.
    If n_channels > 5 * n_octaves, auto-extends octaves to accommodate
    (silently, no error).

    Args:
        n_channels: number of output frequencies needed
        root_hz:    root note in Hz (default 220.0 = A3, warm register)
        n_octaves:  number of octaves to span (default 3 = 15 available notes)

    Returns:
        np.ndarray shape (n_channels,), dtype float64, strictly increasing,
        all values >= root_hz.

    Example: n_channels=8, root_hz=220.0, n_octaves=3
        Available notes: 15 (3 octaves × 5 notes)
        Returns 8 notes evenly distributed across those 15.
    """
    # Auto-extend if needed
    while 5 * n_octaves < n_channels:
        n_octaves += 1

    # Build full note list across all octaves
    notes = []
    for octave in range(n_octaves):
        octave_root = root_hz * (2 ** octave)
        for ratio in PENTATONIC_RATIOS:
            notes.append(octave_root * ratio)
    notes = sorted(set(notes))  # deduplicate, sort ascending

    # Select n_channels evenly distributed notes
    if len(notes) <= n_channels:
        indices = list(range(len(notes)))
    else:
        indices = [round(i * (len(notes) - 1) / (n_channels - 1))
                   for i in range(n_channels)]

    return np.array([notes[i] for i in indices], dtype=np.float64)
```

### Update `assign_frequencies()` in `sonify/mapping.py`

Add `pentatonic_root` and `pentatonic_octaves` keyword arguments. When
`mode == "pentatonic"`, call `assign_frequencies_pentatonic()` and return.
`min_freq` and `max_freq` are ignored in pentatonic mode — state this in
the docstring. The function remains backward-compatible: existing callers
passing `mode="index"` or `mode="wavelength"` are unaffected.

### New fields in `sonify/config.py`

```python
pentatonic_root: float = 220.0      # root frequency Hz; default A3
pentatonic_octaves: int = 3         # octaves to span
```

Validation: `pentatonic_root > 0`, `1 <= pentatonic_octaves <= 8`.

### New CLI args in `scripts/run_sonify.py`

```
--freq-mode {index,wavelength,pentatonic}
--pentatonic-root FLOAT    default: 220.0
--pentatonic-octaves INT   default: 3
```

When pentatonic mode is used, print:
```
[FREQ]  Pentatonic mode: A3 (220.0 Hz) root, 3 octaves, 15 available notes -> 8 selected
[FREQ]  min-freq / max-freq ignored in pentatonic mode
```

---

## Part 2: ADSR Amplitude Envelope

This is the single most impactful change for making the sound feel like an instrument
rather than an oscillator. Every natural instrument has a characteristic amplitude
envelope. The introduction of amplitude envelopes is what makes synthesizers sound
natural. The current code has a linear ramp (simplified sustain) and short fade windows
but no proper attack-decay-release shape.

### Replace the current fade+ramp system in `sonify/synth.py`

Remove: the current `fade_samples` raised-cosine fade-in/fade-out and the
`sustain` linear ramp from `effective_start_amp` to `new_amp`.

Replace with: a proper ADSR envelope applied per-channel per-segment.

```python
def generate_adsr_envelope(
    segment_samples: int,
    sample_rate: int,
    attack_ms: float,
    decay_ms: float,
    sustain_level: float,
    release_ms: float,
) -> np.ndarray:
    """Generate a normalized ADSR amplitude envelope for one row segment.

    The four phases:
      Attack:  ramp from 0 → 1.0 over attack_ms milliseconds
      Decay:   ramp from 1.0 → sustain_level over decay_ms milliseconds
      Sustain: hold at sustain_level for the remaining middle of the segment
      Release: ramp from sustain_level → 0 over release_ms milliseconds

    Each phase is capped so the four phases together never exceed segment_samples.
    Cap rule: each phase gets at most segment_samples // 5 samples, so no single
    phase can consume more than 20% of the segment even at very short playback speeds.

    Args:
        segment_samples: total samples in this row segment
        sample_rate:     samples per second (44100)
        attack_ms:       attack time in milliseconds
        decay_ms:        decay time in milliseconds
        sustain_level:   amplitude level during sustain (0.0–1.0)
        release_ms:      release time in milliseconds

    Returns:
        np.ndarray shape (segment_samples,), values in [0.0, 1.0]
    """
    cap = segment_samples // 5

    a = min(int(attack_ms  * sample_rate / 1000), cap)
    d = min(int(decay_ms   * sample_rate / 1000), cap)
    r = min(int(release_ms * sample_rate / 1000), cap)
    s = max(0, segment_samples - a - d - r)

    attack  = np.linspace(0.0,           1.0,           a) if a > 0 else np.array([])
    decay   = np.linspace(1.0,           sustain_level, d) if d > 0 else np.array([])
    sustain = np.full(s, sustain_level)
    release = np.linspace(sustain_level, 0.0,           r) if r > 0 else np.array([])

    envelope = np.concatenate([attack, decay, sustain, release])

    # Guard: length must match segment_samples exactly (rounding can cause off-by-one)
    if len(envelope) < segment_samples:
        envelope = np.append(envelope, np.full(segment_samples - len(envelope),
                                               envelope[-1] if len(envelope) > 0 else 0.0))
    return envelope[:segment_samples]
```

### ADSR parameters per timbre in `sonify/config.py`

Different timbres need different ADSR shapes. Hard-code sensible defaults per
timbre — do not expose all four ADSR parameters as CLI args (too many knobs).
Instead, expose a single `--adsr-shape {tight,natural,slow}` parameter that
adjusts all four values proportionally:

| Shape     | Attack | Decay | Sustain level | Release | Character |
|-----------|--------|-------|---------------|---------|-----------|
| `tight`   | 5ms    | 30ms  | 0.7           | 20ms    | Plucked, staccato |
| `natural` | 15ms   | 60ms  | 0.6           | 80ms    | Bell-like, default |
| `slow`    | 50ms   | 100ms | 0.5           | 150ms   | Pad-like, ambient |

Add `adsr_shape: Literal["tight", "natural", "slow"] = "natural"` to `SonificationConfig`.

In `synthesize()`, look up the four ADSR values from the shape name before the
synthesis loop. `generate_adsr_envelope()` is called once per segment per channel,
then the oscillator output is multiplied by the envelope:

```python
# Inside the per-row, per-channel synthesis loop:
envelope = generate_adsr_envelope(
    segment_samples, sample_rate,
    attack_ms, decay_ms, sustain_level, release_ms
)
# Scale envelope by the channel's normalized amplitude for this row:
channel_signal = amplitude * envelope * sin_wave
```

**Remove the `sustain` parameter from Phase 5.** The ADSR release phase replaces it
and is more musically accurate. The `sustain_level` in the ADSR shape is the new
equivalent. Keep `sustain` in the config for backward compatibility but document it
as deprecated and ignore it when `adsr_shape` is set.

---

## Part 3: Timbral Partitioning by Spectral Region

This is the core technique used by NASA SYSTEM Sounds. Each wavelength range is
assigned to a different instrument or synthesized sound to create a symphony rather
than a wall of identical sound. For this dataset, the WATSON instrument covers
275–446 nm with 32 bands. Partition into three spectral regions:

| Region     | Bands     | Wavelength (nm) | Timbre  | Character |
|------------|-----------|-----------------|---------|-----------|
| Deep UV    | 1–11      | 275–330         | `bell`  | Warm, harmonic partials |
| Mid UV     | 12–22     | 335–391         | `chime` | Bright, inharmonic shimmer |
| Near UV    | 23–32     | 396–446         | `sine`  | Clean, pure, simple |

After rebinning to N bins, the band indices are scaled proportionally into these
three regions. So with `--n-bins 8`: bins 1-3 → bell, bins 4-6 → chime, bins 7-8 → sine.
With `--n-bins 32` (no rebinning): bins 1-11 → bell, 12-22 → chime, 23-32 → sine.

### Implementation in `sonify/synth.py`

Add `timbre_partition: bool = True` parameter to `synthesize()`. When True:

1. Compute the partition boundary indices based on `n_channels`:
   ```python
   third = n_channels // 3
   partition = {
       "bell":  list(range(0, third)),
       "chime": list(range(third, 2 * third)),
       "sine":  list(range(2 * third, n_channels)),
   }
   ```
2. For each channel in the synthesis loop, look up which group it belongs to,
   and use that group's partial ratios and amplitudes instead of the global `timbre`.
3. Each group also gets its own ADSR shape: bell → `natural`, chime → `tight`,
   sine → `slow`. This further differentiates the three regions.

When `timbre_partition=False`, all channels use the single global `timbre` setting
(existing behavior).

Add to `SonificationConfig`:
```python
timbre_partition: bool = True   # new default: True
```

Add to CLI:
```
--timbre-partition / --no-timbre-partition    default: --timbre-partition
```

If wavelength table is loaded and channels have known wavelengths (either original or
rebinned), use actual wavelength boundaries (275–330 / 330–391 / 391–446 nm) instead
of index thirds. Fall back to index thirds when wavelengths are not available.

---

## Part 4: Temporal Amplitude Smoothing

Row-to-row amplitude jumps — where a channel is bright in row N and near-zero in
row N+1 — produce jarring loudness discontinuities that contribute to the noise
texture. The fix is to smooth the amplitude matrix along the time axis before
synthesis, suppressing rapid changes while preserving gradual structure.

### New function in `sonify/mapping.py`

```python
def smooth_amplitude_matrix(
    matrix: np.ndarray,
    smoothing: float,
    sample_axis: int = 0,
) -> np.ndarray:
    """Apply Gaussian temporal smoothing along the row axis per channel.

    Smoothing is applied AFTER global gain normalization and BEFORE synthesis.
    This suppresses row-to-row amplitude jumps while preserving gradual
    depth-dependent structure.

    Args:
        matrix:      (n_rows, n_channels) amplitude matrix, values in [0, 1]
        smoothing:   0.0 = no smoothing (identity), 1.0 = maximum (sigma=10 rows)
        sample_axis: axis to smooth along (0 = rows/time axis)

    Returns:
        np.ndarray same shape as matrix, values clipped to [0, 1]

    Uses scipy.ndimage.gaussian_filter1d. sigma = smoothing * 10.
    At smoothing=0.3: sigma=3 rows, light smoothing, preserves sharp transitions.
    At smoothing=0.7: sigma=7 rows, heavy smoothing, very flowing transitions.
    """
    if smoothing <= 0.0:
        return matrix.copy()
    from scipy.ndimage import gaussian_filter1d
    sigma = smoothing * 10.0
    smoothed = gaussian_filter1d(matrix.astype(np.float64), sigma=sigma, axis=sample_axis)
    return np.clip(smoothed, 0.0, 1.0)
```

### New config field and CLI arg

```python
smoothing: float = 0.3    # default: light smoothing
```

Validation: `0.0 <= smoothing <= 1.0`.

```
--smoothing FLOAT    default: 0.3
                     0.0 = no smoothing (original behavior)
                     0.3 = light (recommended default)
                     0.7 = heavy (ambient/meditative feel)
```

### Pipeline placement in `run_sonify.py`

Add after `apply_global_gain()` and before `assign_frequencies()`:

```python
matrix = smooth_amplitude_matrix(matrix, config.smoothing)
```

Print:
```
[SMOOTH] Temporal smoothing: sigma=3.0 rows (smoothing=0.3)
```

---

## Part 5: Presets and CLI Wiring

### Sentinel approach for preset override detection

Change the argparse `default` for all preset-controlled arguments to `None`:
`freq_mode`, `timbre`, `timbre_partition`, `adsr_shape`, `gain_mode`, `scale`,
`smoothing`, `sustain` (deprecated), `playback_speed` (already sentinel from Phase 5).
`apply_preset()` fills in `None` values from the preset definition. Remaining `None`
values after preset application are filled from the global defaults table before
building `SonificationConfig`.

### Preset definitions

```python
PRESETS = {
    "none": {},   # no overrides, all global defaults apply

    "chime": {
        # Target: wind chime aesthetic, the primary Dr. Malaska goal
        "n_bins":           8,
        "freq_mode":        "pentatonic",
        "pentatonic_root":  220.0,
        "pentatonic_octaves": 3,
        "timbre":           "chime",       # fallback if partition disabled
        "timbre_partition": True,
        "adsr_shape":       "tight",
        "sustain":          0.0,           # ADSR handles this now
        "gain_mode":        "max_log",
        "scale":            "log10",
        "smoothing":        0.3,
        # Result: 8 pentatonic-tuned channels, partitioned bell/chime/sine timbres,
        # tight ADSR (plucked feel), log-compressed gain. Should sound like wind chimes.
    },

    "ambient": {
        # Target: slow, meditative, full dataset playback
        "n_bins":           6,
        "freq_mode":        "pentatonic",
        "pentatonic_root":  110.0,         # A2, one octave lower = warmer
        "pentatonic_octaves": 4,
        "timbre":           "bell",
        "timbre_partition": True,
        "adsr_shape":       "slow",
        "sustain":          0.0,
        "gain_mode":        "median_log",
        "scale":            "log10",
        "smoothing":        0.7,           # heavy smoothing = very flowing
        # Result: deep, warm, slow-evolving soundscape. Good for full 4000-row run.
    },

    "scientific": {
        # Target: data fidelity over aesthetics, all channels, no musical processing
        "n_bins":           None,          # no rebinning, use all detected channels
        "freq_mode":        "index",
        "timbre":           "sine",
        "timbre_partition": False,
        "adsr_shape":       "natural",
        "sustain":          0.0,
        "gain_mode":        "max_linear",
        "scale":            "linear",
        "smoothing":        0.0,           # no smoothing, raw amplitude
        # Result: scientifically accurate, may still sound noisy at 32 channels.
        # Use --n-bins 8 manually if pleasantness is desired with this preset.
    },
}
```

### `apply_preset(args, n_detected_channels)` in `run_sonify.py`

```python
def apply_preset(args, n_detected_channels: int) -> None:
    """Fill None values on args namespace from preset definition.

    Called AFTER band detection (so n_detected_channels is known) and BEFORE
    SonificationConfig is built. Only sets values where args attribute is None
    (i.e., user did not explicitly provide that argument).
    """
    preset_name = args.preset or "none"
    preset = PRESETS.get(preset_name, {})

    for key, value in preset.items():
        if getattr(args, key, None) is None:
            if key == "n_bins" and value is None:
                setattr(args, key, n_detected_channels)  # no rebinning
            else:
                setattr(args, key, value)

    if preset_name != "none":
        applied = {k: getattr(args, k) for k in preset}
        parts = ", ".join(f"{k}={v}" for k, v in applied.items())
        print(f"[PRESET] {preset_name}: {parts}")
```

### Density warning

When `timbre` is bell or chime AND `n_bins` was not explicitly set (still None before
preset application) AND `--yes` is not set, print:

```
[WARN]  Timbre "chime" with 32 channels = 128 simultaneous partials. Sounds like noise.
[WARN]  Recommended: --n-bins 8 --freq-mode pentatonic, or use --preset chime
```

Only print this before `apply_preset()` runs, so presets suppress it naturally.

### New `--preset` CLI arg

```
--preset {none,chime,ambient,scientific}    default: none
```

---

## Part 6: Tests

### `tests/test_mapping.py` — 6 new pentatonic tests

```python
def test_pentatonic_count():
    for n in [4, 6, 8, 16]:
        freqs = assign_frequencies_pentatonic(n, root_hz=220.0, n_octaves=4)
        assert len(freqs) == n

def test_pentatonic_strictly_increasing():
    freqs = assign_frequencies_pentatonic(8, root_hz=220.0, n_octaves=3)
    assert all(freqs[i] < freqs[i+1] for i in range(len(freqs)-1))

def test_pentatonic_root_is_first():
    freqs = assign_frequencies_pentatonic(8, root_hz=220.0, n_octaves=3)
    assert freqs[0] == pytest.approx(220.0, rel=0.01)

def test_pentatonic_only_valid_intervals():
    """Every frequency must equal root * 2^k * ratio for some k>=0 and ratio in PENTATONIC_RATIOS."""
    freqs = assign_frequencies_pentatonic(8, root_hz=220.0, n_octaves=3)
    for f in freqs:
        found = False
        for octave in range(5):
            for ratio in (1.0, 1.125, 1.25, 1.5, 1.667):
                if abs(f - 220.0 * (2**octave) * ratio) < 0.01:
                    found = True
        assert found, f"Frequency {f} not on pentatonic scale"

def test_pentatonic_auto_extends_octaves():
    freqs = assign_frequencies_pentatonic(20, root_hz=220.0, n_octaves=3)
    # 3 octaves = 15 notes, 20 requested — should auto-extend without error
    assert len(freqs) == 20
    assert all(np.isfinite(freqs))

def test_smooth_amplitude_matrix_zero_smoothing():
    m = np.random.rand(50, 8)
    result = smooth_amplitude_matrix(m, smoothing=0.0)
    np.testing.assert_array_equal(result, m)

def test_smooth_amplitude_matrix_reduces_variance():
    # High-variance matrix should have lower variance after smoothing
    m = np.random.rand(200, 8)
    smoothed = smooth_amplitude_matrix(m, smoothing=0.5)
    assert smoothed.var() < m.var()
    assert smoothed.min() >= 0.0
    assert smoothed.max() <= 1.0
```

### `tests/test_synth.py` — 4 new ADSR + partition tests

```python
def test_adsr_envelope_length():
    env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
    assert len(env) == 4410

def test_adsr_envelope_starts_at_zero():
    env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
    assert env[0] == pytest.approx(0.0, abs=0.01)

def test_adsr_envelope_ends_at_zero():
    env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
    assert env[-1] == pytest.approx(0.0, abs=0.01)

def test_adsr_envelope_within_range():
    env = generate_adsr_envelope(4410, 44100, 15, 60, 0.6, 80)
    assert env.min() >= -0.01
    assert env.max() <= 1.01

def test_timbre_partition_differs_from_uniform():
    # With partition enabled, output should differ from uniform timbre
    freqs = np.array([220., 247.5, 277.2, 311., 330., 370., 415., 440.])
    amps = np.ones((10, 8)) * 0.5
    w_partition = synthesize(amps, freqs, 0.1, 44100, timbre="chime",
                             timbre_partition=True, adsr_shape="natural")
    w_uniform   = synthesize(amps, freqs, 0.1, 44100, timbre="chime",
                             timbre_partition=False, adsr_shape="natural")
    assert not np.allclose(w_partition, w_uniform)
```

### All prior tests must still pass

Pin all existing `synthesize()` calls in the test file to the old defaults:
`timbre="sine"`, `timbre_partition=False`, `adsr_shape="natural"`, `smoothing=0.0`.
This locks prior tests to the original behavior regardless of new defaults.

**Expected total: 73 prior + 7 mapping + 5 synth = 85 tests, all passing.**

---

## Verification commands

Run all of these. Report what you hear for each — subjective listening result is
required in the walkthrough, not just "ran successfully."

```bash
# Baseline: what it sounds like before this fix (for before/after comparison)
python3 scripts/run_sonify.py --yes --row-end 200 \
    --timbre chime --output outputs/BEFORE_fix.wav

# Core fix: pentatonic + 8 bins + ADSR (the minimum viable fix)
python3 scripts/run_sonify.py --yes --row-end 200 \
    --freq-mode pentatonic --n-bins 8 --timbre chime \
    --adsr-shape natural \
    --output outputs/pentatonic_8bins.wav

# Chime preset: should sound like wind chimes
python3 scripts/run_sonify.py --yes --row-end 200 \
    --preset chime --output outputs/preset_chime.wav

# Ambient preset: slower, warmer, meditative
python3 scripts/run_sonify.py --yes --row-end 200 \
    --preset ambient --output outputs/preset_ambient.wav

# Scientific preset: raw, all channels, no musical processing
python3 scripts/run_sonify.py --yes --row-end 200 \
    --preset scientific --output outputs/preset_scientific.wav

# Smoothing comparison: hear the difference
python3 scripts/run_sonify.py --yes --row-end 200 --preset chime \
    --smoothing 0.0 --output outputs/smooth_off.wav
python3 scripts/run_sonify.py --yes --row-end 200 --preset chime \
    --smoothing 0.7 --output outputs/smooth_heavy.wav

# ADSR shape comparison
python3 scripts/run_sonify.py --yes --row-end 200 --preset chime \
    --adsr-shape tight  --output outputs/adsr_tight.wav
python3 scripts/run_sonify.py --yes --row-end 200 --preset chime \
    --adsr-shape slow   --output outputs/adsr_slow.wav

# Timbral partition on vs off (hear the difference)
python3 scripts/run_sonify.py --yes --row-end 200 \
    --freq-mode pentatonic --n-bins 8 --adsr-shape natural \
    --timbre-partition     --output outputs/partition_on.wav
python3 scripts/run_sonify.py --yes --row-end 200 \
    --freq-mode pentatonic --n-bins 8 --adsr-shape natural \
    --no-timbre-partition  --timbre chime \
    --output outputs/partition_off.wav

# Full V3 video — chime preset + all visual features
python3 scripts/run_sonify.py --yes --row-end 200 \
    --preset chime \
    --marker-shape square --marker-size 150 --show-colorbar \
    --trail-rows 5 --show-minimap \
    --output-name v3_chime \
    --video-output outputs/v3_chime.mp4
```

---

## Walkthrough format when done

Same format as all prior phases:

**Required in the walkthrough:**
- What was built / modified (table with all changed files)
- The `[PRESET]` printout for `--preset chime` confirming values applied
- The `[GAIN]` statistics for the borehole dataset (printed on every run)
- **Listening results for every verification command**, specifically:
  - Does `BEFORE_fix.wav` sound like noise? (it should, for before/after contrast)
  - Does `pentatonic_8bins.wav` sound noticeably less noisy? (the core fix)
  - Does `preset_chime.wav` sound like wind chimes or a similar pleasant texture?
  - Does `preset_ambient.wav` feel slower and warmer?
  - Is `partition_on.wav` audibly richer/more varied than `partition_off.wav`?
  - Is `smooth_heavy.wav` more flowing and less jagged than `smooth_off.wav`?
  - Does `adsr_tight.wav` sound plucked/staccato vs `adsr_slow.wav` feeling pad-like?
- Test counts: 73 prior + 7 new mapping + 5 new synth = **85 total, all passing**
- Render timing for `v3_chime.mp4`