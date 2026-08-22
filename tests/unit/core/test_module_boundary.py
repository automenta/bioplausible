"""Sprint 0.5 module-boundary hardening.

Ensures a *light* submodule import (``bioplausible.core.registry``) does NOT
drag in the heavy stack (torch, the whole zoo) or trigger side-effect model
registration. Consumers that need the zoo must import it explicitly.

Run in a fresh subprocess so the parent pytest process's already-imported
modules don't mask the boundary.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

_SCRIPT = """
import sys
import bioplausible.core.registry
import bioplausible.core.registry as r
registered = bool(r.list_models())
print("torch_loaded", "torch" in sys.modules)
print("zoo_loaded", "bioplausible.zoo" in sys.modules)
print("registered", registered)
"""


def _run(script: str) -> dict[str, str]:
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return {k: v for k, v in (line.split() for line in out.stdout.splitlines())}


def test_light_import_does_not_load_torch_or_zoo():
    r = _run(_SCRIPT)
    assert r["torch_loaded"] == "False"
    assert r["zoo_loaded"] == "False"
    assert r["registered"] == "False"


def test_light_import_exposes_registry_symbols_lazily():
    script = """
import bioplausible.core as core
assert core.Registry is not None
assert core.ComponentCategory is not None
print("ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_top_level_import_is_lazy_and_attr_access_loads():
    script = """
import bioplausible
assert "SystemTrainer" not in vars(bioplausible)  # not loaded on plain import
_ = bioplausible.SystemTrainer  # triggers lazy load
assert "SystemTrainer" in vars(bioplausible)
print("ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
