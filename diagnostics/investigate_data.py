import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.makedirs("diagnostics/outputs", exist_ok=True)

# --- Load ---
df = pd.read_csv(
    "data/raw/2019_07_04_10_59_16_pointcloud_fE_BACKGROUND_SUBTRACTED_corrected_data.csv"
)
band_cols = [c for c in df.columns if c.startswith("Band_") and "_bc" in c
             and "std" not in c.lower() and "sdt" not in c.lower()]
df_sorted = df.sort_values("row_num").reset_index(drop=True)
matrix = df_sorted[band_cols].values.astype(np.float64)

print("=" * 60)
print("INVESTIGATION 1: RAW DATA QUALITY")
print("=" * 60)
print(f"Shape:           {matrix.shape}  ({matrix.shape[0]} rows, {matrix.shape[1]} channels)")
print(f"Global min:      {matrix.min():.4f}")
print(f"Global max:      {matrix.max():.4f}")
print(f"Global mean:     {matrix.mean():.4f}")
print(f"Global median:   {np.median(matrix):.4f}")
print(f"Global std:      {matrix.std():.4f}")
print()

# Sparsity analysis
zero_count = (matrix == 0).sum()
near_zero  = (np.abs(matrix) < 1.0).sum()
total      = matrix.size
print(f"Exact zeros:     {zero_count} / {total}  ({100*zero_count/total:.1f}%)")
print(f"Near-zero (<1):  {near_zero} / {total}  ({100*near_zero/total:.1f}%)")
print()

# Per-channel stats
print("PER-CHANNEL STATISTICS:")
print(f"{'Chan':>5} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'%Zero':>8} {'%Near0':>8}")
for i, col in enumerate(band_cols):
    ch = matrix[:, i]
    pz  = 100 * (ch == 0).mean()
    pn0 = 100 * (np.abs(ch) < 1.0).mean()
    print(f"{i+1:>5} {ch.min():>8.2f} {ch.max():>8.2f} "
          f"{ch.mean():>8.2f} {ch.std():>8.2f} {pz:>7.1f}% {pn0:>7.1f}%")

# Value distribution
print()
percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
vals = matrix.flatten()
vals_pos = vals[vals > 0]
print("DISTRIBUTION OF POSITIVE VALUES:")
for p in percentiles:
    print(f"  {p:>3}th pct: {np.percentile(vals_pos, p):.3f}")

# Plot 1: Heatmap of raw data (all rows x all channels)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

im = axes[0,0].imshow(matrix.T, aspect='auto', cmap='plasma', origin='lower')
axes[0,0].set_title("Raw Band Values (rows x channels)")
axes[0,0].set_xlabel("Row (depth)")
axes[0,0].set_ylabel("Channel (band)")
plt.colorbar(im, ax=axes[0,0])

# Plot 2: Histogram of all values
axes[0,1].hist(vals[vals > 0], bins=100, color='steelblue', edgecolor='none')
axes[0,1].set_title("Distribution of Positive Values")
axes[0,1].set_xlabel("Intensity value")
axes[0,1].set_ylabel("Count")
axes[0,1].set_yscale('log')

# Plot 3: Per-channel mean intensity
ch_means = matrix.mean(axis=0)
ch_stds  = matrix.std(axis=0)
axes[1,0].bar(range(len(band_cols)), ch_means, yerr=ch_stds,
              color='steelblue', alpha=0.7)
axes[1,0].set_title("Per-Channel Mean ± Std")
axes[1,0].set_xlabel("Channel index")
axes[1,0].set_ylabel("Mean intensity")

# Plot 4: Fraction of zero values per channel
pct_zeros = [(matrix[:, i] == 0).mean() * 100 for i in range(matrix.shape[1])]
axes[1,1].bar(range(len(band_cols)), pct_zeros, color='tomato')
axes[1,1].set_title("Percentage of Zero Values per Channel")
axes[1,1].set_xlabel("Channel index")
axes[1,1].set_ylabel("% zero rows")
axes[1,1].axhline(50, color='black', linestyle='--', linewidth=1, label='50% threshold')
axes[1,1].legend()

plt.tight_layout()
plt.savefig("diagnostics/outputs/investigation_1_data_quality.png", dpi=150)
print("\nSaved: diagnostics/outputs/investigation_1_data_quality.png")
