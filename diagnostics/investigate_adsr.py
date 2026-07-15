import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("diagnostics/outputs", exist_ok=True)

from sonify.synth import generate_adsr_envelope, ADSR_SHAPES

SR = 44100
SEGMENT_SAMPLES = int(44100 / 5)  # 200ms at 5 rows/sec

print("=" * 60)
print("INVESTIGATION 4: ADSR ENVELOPE SHAPE VERIFICATION")
print("=" * 60)
print(f"Segment samples: {SEGMENT_SAMPLES} ({SEGMENT_SAMPLES/SR*1000:.0f}ms)")
print()

for shape_name, params in ADSR_SHAPES.items():
    env = generate_adsr_envelope(SEGMENT_SAMPLES, SR, *params)
    print(f"Shape '{shape_name}':")
    print(f"  Params:        {params}")
    print(f"  Length:        {len(env)} samples (expected {SEGMENT_SAMPLES})")
    print(f"  First sample:  {env[0]:.6f}  (expected ~0.0)")
    print(f"  Peak value:    {env.max():.6f}  at sample {env.argmax()}")
    print(f"  Last sample:   {env[-1]:.6f}  (expected ~0.0)")
    print(f"  Mid-point:     {env[len(env)//2]:.6f}")
    print(f"  75% point:     {env[int(len(env)*0.75)]:.6f}")

    # Is it exponential or linear after the attack?
    peak_idx = env.argmax()
    decay_portion = env[peak_idx:]
    if len(decay_portion) > 10:
        # Fit linear: if R² > 0.99 → linear; if R² < 0.95 → exponential
        x = np.arange(len(decay_portion))
        lin_coeffs = np.polyfit(x, decay_portion, 1)
        lin_fit = np.polyval(lin_coeffs, x)
        ss_res = np.sum((decay_portion - lin_fit)**2)
        ss_tot = np.sum((decay_portion - decay_portion.mean())**2)
        r2_linear = 1 - ss_res/ss_tot if ss_tot > 0 else 1.0
        print(f"  Decay R² (linear fit): {r2_linear:.4f}  "
              f"{'← LOOKS LINEAR (bad)' if r2_linear > 0.95 else '← curved/exponential (good)'}")
    print()

# Plot all three shapes
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (shape_name, params) in zip(axes, ADSR_SHAPES.items()):
    env = generate_adsr_envelope(SEGMENT_SAMPLES, SR, *params)
    t_ms = np.arange(len(env)) / SR * 1000
    ax.plot(t_ms, env, linewidth=2, color='steelblue', label='actual')
    ax.set_title(f"ADSR shape: '{shape_name}'\nParams: {params}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0, color='red', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig("diagnostics/outputs/investigation_4_adsr_shapes.png", dpi=150)
print("Saved: diagnostics/outputs/investigation_4_adsr_shapes.png")
