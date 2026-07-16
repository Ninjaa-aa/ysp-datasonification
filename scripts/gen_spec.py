import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

for file_path in sys.argv[1:]:
    print(f"Processing {file_path}")
    sr, data = wavfile.read(file_path)
    d = data.astype(np.float32) / 32768.0
    print(f"  Peak: {np.abs(d).max():.4f}")
    print(f"  RMS:  {np.sqrt(np.mean(d**2)):.4f}")
    print(f"  % silent (<0.001): {100*(np.abs(d)<0.001).mean():.1f}%")

    f, t, Sxx = signal.spectrogram(d, sr, nperseg=2048)
    
    # We plot the first 200 frequency bins up to 2500Hz
    # If the file is very long (like full dataset), plotting might be heavy. We'll plot the whole thing.
    plt.figure(figsize=(14, 5))
    
    # Clip frequencies above 2500Hz (roughly first 120 bins out of 1025 depending on sr/nperseg)
    # The resolution is sr/nperseg = 44100/2048 = 21.5Hz per bin
    # 2500Hz / 21.5 = 116 bins
    n_bins = 120
    plt.pcolormesh(t, f[:n_bins], 10*np.log10(Sxx[:n_bins]+1e-10),
                   shading='gouraud', cmap='plasma', vmin=-100, vmax=-20)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    title_name = os.path.basename(file_path)
    plt.title(f'{title_name} — Spectrogram')
    plt.colorbar(label='Power (dB)')
    plt.ylim(0, 2500)
    plt.tight_layout()
    
    out_name = os.path.splitext(title_name)[0]
    out_path = f"outputs/spectrogram_{out_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Saved {out_path}")
