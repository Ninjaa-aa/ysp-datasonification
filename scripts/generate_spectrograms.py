import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

def generate_spectrogram(wav_path, out_png_path, title):
    print(f"Generating spectrogram for {wav_path}...")
    sr, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    f, t, Sxx = signal.spectrogram(data, sr, nperseg=1024)
    plt.figure(figsize=(14, 5))
    plt.pcolormesh(t, f[:200], 10*np.log10(Sxx[:200]+1e-10), 
                   shading='gouraud', cmap='plasma')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.colorbar(label='Power (dB)')
    plt.ylim(0, 2500)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()
    print(f"Saved to {out_png_path}")

def main():
    base_dir = "/home/hammad/hammad/ysp/ysp-datasonification"
    chime_wav = os.path.join(base_dir, "outputs", "preset_chime.wav")
    before_wav = os.path.join(base_dir, "outputs", "BEFORE_fix.wav")
    
    generate_spectrogram(chime_wav, os.path.join(base_dir, "outputs", "spectrogram_chime.png"), "preset_chime.wav — Spectrogram")
    generate_spectrogram(before_wav, os.path.join(base_dir, "outputs", "spectrogram_before.png"), "BEFORE_fix.wav — Spectrogram")

if __name__ == "__main__":
    main()
