import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("diagnostics/outputs", exist_ok=True)

from sonify.data_io import load_csv
from sonify.band_detect import detect_band_columns
from sonify.preprocess import sort_by_row_order, clean, rebin
from sonify.mapping import (scale_values, apply_global_gain,
                            smooth_amplitude_matrix, assign_frequencies)

# --- Reproduce the chime preset pipeline exactly ---
CSV_PATH   = "data/raw/2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv"
N_BINS     = 8
SCALE      = "log10"
GAIN_MODE  = "max_log"
SMOOTHING  = 0.0
ROW_END    = 200

df = load_csv(CSV_PATH)
band_cols, band_indices = detect_band_columns(df)
df = sort_by_row_order(df)
df = df.iloc[:ROW_END]
matrix_raw = clean(df, band_cols)
matrix_rebinned = rebin(matrix_raw, N_BINS)

# Save checkpoint
clean_matrix = matrix_rebinned.copy()

matrix_scaled  = scale_values(matrix_rebinned.copy(), SCALE)
matrix_gained  = apply_global_gain(matrix_scaled.copy(), GAIN_MODE)
matrix_smoothed = smooth_amplitude_matrix(matrix_gained.copy(), SMOOTHING)

stages = {
    "1_after_clean":   matrix_rebinned,
    "2_after_scale":   matrix_scaled,
    "3_after_gain":    matrix_gained,
    "4_after_smooth":  matrix_smoothed,
}

print("=" * 60)
print("INVESTIGATION 2: PIPELINE INTERMEDIATE VALUES")
print("=" * 60)
for stage_name, m in stages.items():
    print(f"\n{stage_name}:")
    print(f"  Shape:    {m.shape}")
    print(f"  Min:      {m.min():.6f}")
    print(f"  Max:      {m.max():.6f}")
    print(f"  Mean:     {m.mean():.6f}")
    print(f"  Std:      {m.std():.6f}")
    print(f"  % zero:   {100*(m==0).mean():.1f}%")
    print(f"  % < 0.01: {100*(m<0.01).mean():.1f}%")
    print(f"  % > 0.99: {100*(m>0.99).mean():.1f}%")
    nan_count = np.isnan(m).sum()
    inf_count = np.isinf(m).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  *** WARNING: {nan_count} NaN, {inf_count} Inf values ***")

# Plot: heatmap at each stage
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, (name, m) in zip(axes.flat, stages.items()):
    im = ax.imshow(m.T, aspect='auto', cmap='plasma', origin='lower',
                   vmin=0, vmax=m.max())
    ax.set_title(name)
    ax.set_xlabel("Row")
    ax.set_ylabel("Channel (bin)")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("diagnostics/outputs/investigation_2_pipeline_stages.png", dpi=150)
print("\nSaved: diagnostics/outputs/investigation_2_pipeline_stages.png")

# Plot: per-channel amplitude over time (final matrix)
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
for i, ax in enumerate(axes.flat):
    if i < N_BINS:
        ax.plot(matrix_smoothed[:, i], linewidth=0.8, color='steelblue')
        ax.set_title(f"Channel {i+1} amplitude over rows")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Amplitude [0,1]")
        ax.set_xlabel("Row")
        ax.axhline(0, color='red', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig("diagnostics/outputs/investigation_2_per_channel_amplitude.png", dpi=150)
print("Saved: diagnostics/outputs/investigation_2_per_channel_amplitude.png")

# Print the actual frequency values being assigned
freqs = assign_frequencies(N_BINS, 150, 2500, mode="pentatonic",
                           pentatonic_root=220.0, pentatonic_octaves=3)
print("\nFREQUENCIES ASSIGNED TO EACH BIN:")
for i, f in enumerate(freqs):
    print(f"  Bin {i+1}: {f:.2f} Hz")
