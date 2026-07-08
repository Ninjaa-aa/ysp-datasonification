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
