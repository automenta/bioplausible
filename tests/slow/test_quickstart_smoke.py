"""Slow-tier smoke for the canonical quickstart entry point (TODO8 R5.3).

Runs ``scripts/quickstart.py`` (Backprop vs Forward-Forward on MNIST) with a
1-epoch budget via ``QUICKSTART_EPOCHS`` and asserts the documented output
contract: clean exit, both accuracy lines rendered, SUMMARY block present.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

QUICKSTART = Path(__file__).resolve().parents[2] / "scripts" / "quickstart.py"


@pytest.mark.timeout(600)
def test_quickstart_end_to_end() -> None:
    env = {**os.environ, "QUICKSTART_EPOCHS": "1"}
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed argv
        [sys.executable, str(QUICKSTART)],
        env=env,
        capture_output=True,
        text=True,
        timeout=540,
        cwd=QUICKSTART.parents[1],
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "Backprop:" in proc.stdout
    assert "Forward-Forward:" in proc.stdout
    assert "SUMMARY" in proc.stdout
    assert "accuracy" in proc.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
