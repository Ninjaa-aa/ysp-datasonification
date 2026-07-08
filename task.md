# Phase 3 & 4 — Task Tracker

## Part 1: Trail Display Fix
- [x] Add `trail_rows` to `SonificationConfig` + validation
- [x] Rewrite `render_frame()` for trail (2-D amplitude_history)
- [x] Update `render_all_frames()` with sliding window
- [x] Update `live_display()` with trail support
- [x] Add `--trail-rows` to CLI runner
- [x] Update existing visualize tests for new signature
- [x] Add 3 new trail tests
- [x] Run tests — all 48 pass ✅

## Part 2: Phase 3 — Expanded Channels & Parameter Mapping
- [x] Add `SCATTER_CHANNEL_LIMIT` and heatmap mode in `render_frame()`
- [x] Add `max_frames` to config + `render_all_frames()` + CLI
- [x] Rewrite `confirm_with_user()` with table + edit mode
- [x] Add `ParameterMap` dataclass to config
- [x] Add `map_tone_from_column()` and `apply_intensity_column()` to mapping.py
- [x] Support 2-D freqs in `synth.py`
- [x] Wire parameter mapping into CLI runner
- [x] Add 2 new parameter mapping tests
- [x] Run tests — all 50 pass ✅

## Part 3: Phase 4 — Minimap, CLI Polish, Output Naming, README
- [x] Add minimap rendering to `visualize.py`
- [x] Add `show_minimap` to config + CLI
- [x] Add `log()` helper and polished CLI output
- [x] Add `--output-name` to config + CLI
- [x] Rewrite README.md for GitHub V1
- [x] Add test_cli.py with help output test
- [x] Run tests — all 51 pass ✅
