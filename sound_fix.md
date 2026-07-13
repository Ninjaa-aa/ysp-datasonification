# Sound Fix — Three Spectrogram-Identified Issues

## Step 0: Read before touching anything

Read these files in full:

```
sonify/synth.py          — find ADSR_SHAPES and chime partial definitions
sonify/config.py         — find adsr_shape, sustain, timbre_partition fields
scripts/run_sonify.py    — find PRESETS dict, specifically the "chime" preset
tests/test_synth.py      — find all ADSR-related tests
```

After reading, confirm in one paragraph: what the current `ADSR_SHAPES["tight"]`
values are, how many partials the chime timbre currently generates per channel, and
what `playback_speed` the chime preset currently sets (if any). Then proceed.

---

## Context: what the spectrograms showed

Two spectrograms were analyzed (BEFORE_fix.wav and preset_chime.wav) and they
revealed three specific, diagnosable problems that the current code has:

**Problem 1 — ADSR sustain level too high.**
The chime spectrogram shows horizontal frequency bands that are bright and continuous
for the full 20-second duration. They never dim. A wind chime makes a sound then goes
silent before the next note. The current `tight` ADSR shape has `sustain_level=0.7`,
meaning 70% amplitude is held continuously throughout every row. The note never decays
to silence. It sounds like a buzzing organ, not a chime.

**Problem 2 — Too many chime partials filling the gaps between pentatonic notes.**
Between the 8 main frequency bands in the chime spectrogram, the space is orange
(occupied) rather than purple (near-silent). It should be nearly silent between
pentatonic notes. The cause: 8 pentatonic fundamentals × 4 inharmonic partials =
32 frequencies distributed almost continuously across the spectrum, recreating the
original noise problem at a smaller scale. The partials land exactly in the gaps
between the pentatonic notes.

**Problem 3 — Playback speed too high for ADSR to breathe.**
At 10 rows/sec, each row segment is 100ms. The `tight` ADSR is
5ms + 30ms + hold + 20ms = 100ms total, meaning the note reaches zero exactly when
the next note starts. There is no silence, no gap, no contrast between notes — just
continuous sound. At 5 rows/sec, each segment is 200ms, giving the decay 90ms to
reach zero and leaving ~100ms of near-silence before the next note. That gap is what
makes it sound like a chime rather than a drone.

---

## The three changes to make

These are surgical fixes to specific values only. Do not restructure or rewrite
anything else. Three changes, three files, no more.

---

### Change 1: Fix `ADSR_SHAPES` in `sonify/synth.py`

Find the `ADSR_SHAPES` dict and replace it with:

```python
ADSR_SHAPES = {
    #              attack_ms  decay_ms  sustain_level  release_ms
    "tight":   (   5,         90,       0.0,           5   ),
    "natural": (   15,        60,       0.3,           80  ),
    "slow":    (   50,        100,      0.2,           150 ),
}
```

Changes from current values:
- `tight`: decay extended 30ms → 90ms; sustain_level dropped 0.7 → **0.0**;
  release shortened 20ms → 5ms. The note now hits, decays fully to silence within
  90ms, and stops. No sustained buzz.
- `natural`: sustain_level dropped 0.6 → 0.3. Less drone, more natural decay.
- `slow`: sustain_level dropped 0.5 → 0.2. Same reasoning.

The `sustain_level=0.0` for `tight` is the critical fix. This is what creates the
percussive, staccato "plink" quality of a wind chime rather than a continuous tone.

### Change 2: Reduce chime partial count in `sonify/synth.py`

Find where the chime timbre generates its inharmonic partials. Currently it generates
4 partials. Replace with 2 partials only:

```python
# Chime timbre: 2 partials only (was 4)
CHIME_PARTIALS = [
    (1.000, 1.00),   # fundamental
    (2.756, 0.35),   # first inharmonic overtone, reduced amplitude (was 0.50)
]
# Removed: (5.404, 0.25) and (8.933, 0.12)
# These were landing in the gaps between pentatonic notes and filling the
# spectrum with noise. 2 partials per channel × 8 bins = 16 frequencies,
# which is sparse enough for the gaps to be audible.
```

Normalization: the sum of partial amplitudes is now 1.00 + 0.35 = 1.35.
Normalize by dividing by 1.35 (not 1.87 as before) to keep output in the
same amplitude range.

If the chime partial ratios are not defined as a named constant but are instead
inline in a synthesis loop, find that loop and reduce to the first two entries
only. The principle is the same regardless of how it is structured.

### Change 3: Add `playback_speed` to the chime preset in `scripts/run_sonify.py`

Find the `PRESETS["chime"]` dict and add one line:

```python
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
    "playback_speed":     5.0,      # ADD THIS LINE — was missing
},
```

5 rows/sec means 200ms per row. With a 90ms decay to silence, each row has ~100ms
of near-silence before the next note. This gap is audible and is what distinguishes
a wind chime from a drone.

`apply_preset()` already handles `playback_speed` as a sentinel (it is already `None`
by default from Phase 5), so no changes to `apply_preset()` are needed. The preset
value will correctly be overridden if the user explicitly passes `--playback-speed`.

---

## Tests to update

The ADSR shape changes mean the two tests that check specific ADSR behavior need
updating. Find and update these:

**`test_adsr_tight_differs_from_slow`** — this test currently passes because tight
and slow produce different waveforms. After the change they still will (different
values), so it should still pass without modification. Run it and confirm.

**`test_adsr_amplitude_shape`** — this test checks `env[0] ≈ 0.0` and
`env[-1] ≈ 0.0`. With `sustain_level=0.0` in `tight`, the sustain phase is silent
and the release is 5ms — at 44100 Hz, 5ms = ~220 samples. Confirm the test still
passes. If the release is so short it gets capped by `segment_samples // 5` at the
test's segment length, adjust the test's segment length to something longer (e.g.
44100 samples = 1 second) so the 5ms release is not capped.

**`test_adsr_short_segment_no_crash`** — still valid, no changes needed.

**Do not add new tests.** These are value-only changes, not structural ones. The
existing 78 tests should still all pass. If any fail due to the value changes, fix
the test fixture values to match the new ADSR parameters — do not revert the ADSR
changes.

---

## Verification — generate these four files and report what you hear

```bash
# 1. Chime preset with all three fixes (the main output)
python3 scripts/run_sonify.py --yes --preset chime \
    --row-end 200 --output outputs/chime_fixed.wav

# 2. Same but at 10 rows/sec so you can hear the silence gap difference
python3 scripts/run_sonify.py --yes --preset chime \
    --row-end 200 --playback-speed 10 \
    --output outputs/chime_fixed_10rps.wav

# 3. Slow ADSR with the new lower sustain — should feel ambient, not droning
python3 scripts/run_sonify.py --yes --preset ambient \
    --row-end 200 --output outputs/ambient_fixed.wav

# 4. Generate updated spectrogram for chime_fixed.wav using the same script
#    as before, and upload it — the horizontal bands should now have dark gaps
#    between them and each band should pulse brighter/dimmer rather than
#    staying continuously bright
```

```python
# Spectrogram generation script (same as before, just change the filename)
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

sr, data = wavfile.read("outputs/chime_fixed.wav")
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0

f, t, Sxx = signal.spectrogram(data, sr, nperseg=1024)
plt.figure(figsize=(14, 5))
plt.pcolormesh(t, f[:200], 10*np.log10(Sxx[:200]+1e-10),
               shading='gouraud', cmap='plasma')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('chime_fixed.wav — Spectrogram')
plt.colorbar(label='Power (dB)')
plt.ylim(0, 2500)
plt.tight_layout()
plt.savefig("outputs/spectrogram_chime_fixed.png", dpi=150)
```

Upload `spectrogram_chime_fixed.png` alongside the walkthrough. The spectrogram
is the objective proof that the fixes worked:
- The gaps between horizontal bands should now be clearly **purple** (near-silent),
  not orange
- Each band should show visible **pulsing** (brighter peaks with darker valleys)
  rather than a flat continuous bright line
- The region above ~1800 Hz should be significantly darker (fewer high partials)

---

## Walkthrough format

Short and focused — this is a three-value fix, not a full phase:

- Confirm the three exact values changed and in which files
- Report what `chime_fixed.wav` sounds like vs the old `preset_chime.wav`
- Report what `chime_fixed_10rps.wav` sounds like to confirm the silence-gap effect
- Upload `spectrogram_chime_fixed.png` — describe what changed vs the old spectrogram
- Test count: same as before (78), all passing