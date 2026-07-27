"""Tests for CLI argument presence (Phase 4)."""

import subprocess
import sys


class TestCLIHelp:
    """Verify that all new CLI arguments are present in --help output."""

    def test_help_includes_new_args(self):
        """Run run_sonify.py --help and confirm new args are present."""
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "scripts/run_sonify.py", "--help"],
            capture_output=True,
            text=True,
            cwd=".",
            env=env,
        )
        # Should exit cleanly
        assert result.returncode == 0, f"--help failed: {result.stderr}"

        help_text = result.stdout

        # Phase 3 args
        assert "--trail-rows" in help_text, "Missing --trail-rows"
        assert "--max-frames" in help_text, "Missing --max-frames"
        assert "--tone-source" in help_text, "Missing --tone-source"
        assert "--tone-column" in help_text, "Missing --tone-column"
        assert "--intensity-source" in help_text, "Missing --intensity-source"
        assert "--intensity-column" in help_text, "Missing --intensity-column"

        # Phase 4 args
        assert "--show-minimap" in help_text, "Missing --show-minimap"
        assert "--output-name" in help_text, "Missing --output-name"

        # Display + auto-gain (Dr. Malaska's 2026-07-09 feedback)
        assert "--marker-size" in help_text, "Missing --marker-size"
        assert "--marker-shape" in help_text, "Missing --marker-shape"
        assert "--show-colorbar" in help_text, "Missing --show-colorbar"
        assert "--gain-mode" in help_text, "Missing --gain-mode"
        assert "--timbre" in help_text, "Missing --timbre"

        # Trigger + lambda-max (Dr. Malaska's 2026-07-24 two-function design)
        assert "--threshold" in help_text, "Missing --threshold"
        assert "--trigger-type" in help_text, "Missing --trigger-type"
        assert "--target-tones" in help_text, "Missing --target-tones"
        assert "lambda_max" in help_text, "Missing lambda_max tone source"

        # Sustain, as a decaying tail
        assert "--reverb-tail-ms" in help_text, "Missing --reverb-tail-ms"

    def test_sustain_flag_is_gone(self):
        """--sustain was never applied; it is removed rather than left misleading."""
        result = subprocess.run(
            [sys.executable, "scripts/run_sonify.py", "--help"],
            capture_output=True, text=True, cwd=".",
        )
        assert "--sustain" not in result.stdout

    def test_event_preset_is_available(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_sonify.py", "--help"],
            capture_output=True, text=True, cwd=".",
        )
        assert "event" in result.stdout, "Missing 'event' preset choice"


class TestTriggerPrecedence:
    """An explicit --threshold must beat a preset's --target-tones.

    Regression: `--preset event --threshold 400` used to silently render the
    preset's 25-tone target instead of the threshold asked for, so every
    threshold produced byte-identical audio.
    """

    @staticmethod
    def _run(*extra):
        import os
        return subprocess.run(
            [sys.executable, "scripts/run_sonify.py", "--yes", "--preset", "event",
             "--output", os.devnull, *extra],
            capture_output=True, text=True, cwd=".",
        )

    def test_explicit_threshold_overrides_preset_target(self):
        out = self._run("--threshold", "900").stdout
        assert "threshold 900" in out, out[-600:]

    def test_explicit_target_tones_still_solves_threshold(self):
        out = self._run("--target-tones", "10").stdout
        assert "Target 10 tones" in out, out[-600:]

    def test_different_thresholds_give_different_event_counts(self):
        a = self._run("--threshold", "400").stdout
        b = self._run("--threshold", "900").stdout
        assert "50 of 4000 rows sound" in a, a[-600:]
        assert "10 of 4000 rows sound" in b, b[-600:]
