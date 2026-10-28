"""Tests for the resnet_accuracy experiment."""

import subprocess
import sys
from pathlib import Path

MAIN = (
    Path(__file__).resolve().parents[2]
    / "src/workloads/experiments/resnet_accuracy/main.py"
)


def test_help_runs() -> None:
    """``--help`` parses without loading data or weights."""
    result = subprocess.run(
        [sys.executable, str(MAIN), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
