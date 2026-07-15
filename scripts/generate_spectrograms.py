import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

sr, data = wavfile.read("outputs/chime_bugfixed.wav")
d = data.astype(np.float32) / 32768.0
print(f"Peak: {np.abs(d).max():.4f}")
print(f"RMS:  {np.sqrt(np.mean(d**2)):.4f}")
print(f"% silent (<0.001): {100*(np.abs(d)<0.001).mean():.1f}%")

f, t, Sxx = signal.spectrogram(d, sr, nperseg=2048)
plt.figure(figsize=(14, 5))
plt.pcolormesh(t, f[:200], 10*np.log10(Sxx[:200]+1e-10),
               shading='gouraud', cmap='plasma', vmin=-100, vmax=-20)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('chime_bugfixed.wav — Spectrogram')
plt.colorbar(label='Power (dB)')
plt.ylim(0, 2500)
plt.tight_layout()
plt.savefig("outputs/spectrogram_bugfixed.png", dpi=150)
