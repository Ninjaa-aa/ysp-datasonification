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
