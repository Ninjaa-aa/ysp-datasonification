# Phase 3 & 4 — Walkthrough

## What Was Built / Modified

| File | Change Summary |
|------|---------------|
| [config.py](file:///d:/Freelancing/YSP/sonify/config.py) | Added `ParameterMap` dataclass, `trail_rows`, `max_frames`, `show_minimap`, `output_name` fields + validation |
| [visualize.py](file:///d:/Freelancing/YSP/sonify/visualize.py) | Full rewrite: trail display (stacked N rows with fading opacity/size), `SCATTER_CHANNEL_LIMIT=256` auto-switch to heatmap, minimap panel support |
| [band_detect.py](file:///d:/Freelancing/YSP/sonify/band_detect.py) | Formatted table confirmation UI with Index/Column/Band/Wavelength columns, excluded/housekeeping summary, interactive edit mode (add/remove channels) |
| [mapping.py](file:///d:/Freelancing/YSP/sonify/mapping.py) | Added `map_tone_from_column()` and `apply_intensity_column()` for parameter mapping |
| [synth.py](file:///d:/Freelancing/YSP/sonify/synth.py) | Support 2-D `freqs` array `(n_rows, n_channels)` for per-row frequency assignment |
| [run_sonify.py](file:///d:/Freelancing/YSP/scripts/run_sonify.py) | Full rewrite: structured `log()` output, `--trail-rows`, `--max-frames`, `--tone-source/column`, `--intensity-source/column`, `--show-minimap`, `--output-name` |
| [README.md](file:///d:/Freelancing/YSP/README.md) | Complete V1 rewrite: mission statement, install, quick start, full CLI table, tuning guide, dataset format, credits, MIT license |
| [test_visualize.py](file:///d:/Freelancing/YSP/tests/test_visualize.py) | Updated for 2-D signature + 3 new trail tests |
| [test_mapping.py](file:///d:/Freelancing/YSP/tests/test_mapping.py) | 2 new parameter mapping tests |
| [test_cli.py](file:///d:/Freelancing/YSP/tests/test_cli.py) | **[NEW]** CLI help output test verifying all new args present |

---

## Key Design Decisions

### Scatter → Heatmap Threshold
- **`SCATTER_CHANNEL_LIMIT = 256`** — Below this, scatter dots/circles are readable as individual markers. Above, 10,240+ scatter points per frame would be painfully slow and unreadable; a 2-D `imshow()` heatmap is both faster and more informative.
- The switch is automatic based on `n_channels` — no user flag needed.

### Minimap Pre-Render Strategy
- The full-dataset heatmap is rendered **once** as a numpy RGB array before the frame loop using matplotlib's colormap applied directly to the amplitude matrix.
- Per frame, only the position line is drawn on top. This keeps per-frame cost low (no re-rendering the full heatmap).
- The minimap takes the right 20% of the frame via `GridSpec(1, 2, width_ratios=[4, 1])`.

### Trail Display
- Each trail row gets a y-position from `linspace(0.75, 0.25, n_trail)` — oldest at top, newest at bottom.
- Opacity: `0.2 + 0.8 * age_frac` (20% → 100%)
- Marker size: `0.4 + 0.6 * age_frac` (40% → 100%)
- Partial fill: when fewer rows exist than `trail_rows`, only available rows are shown (no empty padding).

### `confirm_with_user()` Return Type Change
- Changed from `bool` to `list[str]` to support the edit workflow. Empty list signals abort (equivalent to old `False`). The caller in `run_sonify.py` checks `if not confirmed_cols`.

### Parameter Mapping
- `map_tone_from_column()`: normalizes column values to [0,1], maps to [min_freq, max_freq] in log-space. Returns `(n_rows,)` array, then broadcast to `(n_rows, n_channels)` so all channels share the same frequency per row.
- `apply_intensity_column()`: normalizes column to [0,1], multiplies each row's amplitude vector by its scalar. Preserves relative channel differences while modulating global volume per row.

---

## Test Counts

| Source | Count |
|--------|-------|
| Phase 1 (unchanged) | 39 |
| Phase 2 (updated signature, behavior unchanged) | 6 |
| Trail tests (new) | 3 |
| Parameter mapping tests (new) | 2 |
| CLI test (new) | 1 |
| **Total** | **51** |

All 51 tests passing ✅
