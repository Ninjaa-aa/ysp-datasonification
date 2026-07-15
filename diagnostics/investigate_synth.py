import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os
os.makedirs("diagnostics/outputs", exist_ok=True)

# Load the actual output file
sr, data = wavfile.read("outputs/chime_fixed.wav")
waveform = data.astype(np.float32) / 32768.0

print("=" * 60)
print("INVESTIGATION 3: SYNTHESIS OUTPUT ANALYSIS")
print("=" * 60)
print(f"Sample rate:     {sr} Hz")
print(f"Duration:        {len(waveform)/sr:.2f} s")
print(f"Total samples:   {len(waveform)}")
print()
print("AMPLITUDE STATISTICS:")
print(f"  Peak:          {np.abs(waveform).max():.6f}")
print(f"  RMS:           {np.sqrt(np.mean(waveform**2)):.6f}")
print(f"  Mean absolute: {np.abs(waveform).mean():.6f}")
print(f"  DC offset:     {waveform.mean():.6f}")
print()

# Dynamic range
peak_db = 20 * np.log10(np.abs(waveform).max() + 1e-10)
rms_db  = 20 * np.log10(np.sqrt(np.mean(waveform**2)) + 1e-10)
print(f"  Peak (dB):     {peak_db:.1f} dBFS")
print(f"  RMS (dB):      {rms_db:.1f} dBFS")
print(f"  Crest factor:  {peak_db - rms_db:.1f} dB  "
      f"(high = sparse/bursty; typical music ~10-15 dB)")
print()

# Clipping check
clip_threshold = 0.99
n_clipped = (np.abs(waveform) > clip_threshold).sum()
print(f"  Clipped samples (>{clip_threshold}): {n_clipped}  "
      f"({100*n_clipped/len(waveform):.3f}%)")
print()

# Silence analysis
silence_threshold = 0.001
silence_mask = np.abs(waveform) < silence_threshold
pct_silence = 100 * silence_mask.mean()
print(f"  Near-silent samples (<{silence_threshold}): "
      f"{pct_silence:.1f}%")
print()

# Row-level analysis: what does each 200ms row segment look like?
samples_per_row = int(sr / 5.0)   # 5 rows/sec
n_rows = len(waveform) // samples_per_row
print(f"ANALYSIS PER ROW SEGMENT ({samples_per_row} samples each):")
print(f"{'Row':>5} {'Peak':>8} {'RMS':>8} {'%Silent':>10}")
for r in range(min(n_rows, 20)):
    seg = waveform[r*samples_per_row:(r+1)*samples_per_row]
    peak = np.abs(seg).max()
    rms  = np.sqrt(np.mean(seg**2))
    pct_sil = 100 * (np.abs(seg) < 0.001).mean()
    print(f"{r+1:>5} {peak:>8.4f} {rms:>8.4f} {pct_sil:>9.1f}%")
print()

# ENVELOPE SHAPE: zoom into 3 individual row segments to see the ADSR shape
fig, axes = plt.subplots(3, 2, figsize=(16, 10))

# Left column: time-domain waveform of rows 1, 10, 20
for row_idx, ax in zip([0, 10, 19], axes[:, 0]):
    if row_idx < n_rows:
        seg = waveform[row_idx*samples_per_row:(row_idx+1)*samples_per_row]
        t_ms = np.arange(len(seg)) / sr * 1000
        ax.plot(t_ms, seg, linewidth=0.5, color='steelblue')
        ax.set_title(f"Row {row_idx+1} waveform (time domain)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color='gray', linewidth=0.5)

# Right column: amplitude envelope (abs) of same rows
for row_idx, ax in zip([0, 10, 19], axes[:, 1]):
    if row_idx < n_rows:
        seg = waveform[row_idx*samples_per_row:(row_idx+1)*samples_per_row]
        envelope = np.abs(seg)
        # Smooth for visibility
        from scipy.ndimage import uniform_filter1d
        env_smooth = uniform_filter1d(envelope, size=50)
        t_ms = np.arange(len(seg)) / sr * 1000
        ax.plot(t_ms, env_smooth, linewidth=1.5, color='tomato')
        ax.set_title(f"Row {row_idx+1} amplitude envelope")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("|Amplitude|")
        ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("diagnostics/outputs/investigation_3_row_envelopes.png", dpi=150)
print("Saved: diagnostics/outputs/investigation_3_row_envelopes.png")

# Full spectrogram
f_spec, t_spec, Sxx = signal.spectrogram(waveform, sr, nperseg=2048)
fig, ax = plt.subplots(figsize=(16, 5))
ax.pcolormesh(t_spec, f_spec[:300], 10*np.log10(Sxx[:300]+1e-10),
              shading='gouraud', cmap='plasma', vmin=-100, vmax=-20)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_title('chime_fixed.wav — Full Spectrogram (0–3000 Hz)')
ax.set_ylim(0, 3000)
plt.colorbar(plt.cm.ScalarMappable(
    norm=plt.Normalize(-100, -20), cmap='plasma'), ax=ax, label='Power (dB)')
plt.tight_layout()
plt.savefig("diagnostics/outputs/investigation_3_spectrogram.png", dpi=150)
print("Saved: diagnostics/outputs/investigation_3_spectrogram.png")
