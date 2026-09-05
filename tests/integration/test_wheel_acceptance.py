"""Wheel acceptance test (R11.4.1 / CP-5).

Builds the wheel, installs it into a fresh venv (torch inherited from the
parent environment via ``--system-site-packages``), and runs the RESEARCH3
PR-sequence smoke: import, compose a 5-D system through ``SystemModule``,
forward under ``no_grad``, and one ``fit_step`` — the stranger's first
minute with a pip-installed computronium.
"""

import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_SMOKE = """
import torch
from computronium import (
    BackpropCredit, CreditAssignmentConfig, DigitalSubstrate, EuclideanUpdate,
    FeedforwardGeometry, GeometryConfig, InstantaneousDynamics,
    ParameterUpdateConfig, StateDynamicsConfig, SubstrateConfig, SystemModule,
    compose_system,
)
model = SystemModule(compose_system(
    substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
    geometry=FeedforwardGeometry(GeometryConfig.feedforward(
        input_dim=8, output_dim=4, hidden_dims=(6,))),
    dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
    credit=BackpropCredit(CreditAssignmentConfig.gradient()),
    update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
))
x = torch.randn(4, 8)
with torch.no_grad():
    logits = model(x)
assert logits.shape == (4, 4)
metrics = model.fit_step(x, torch.randint(0, 4, (4,)))
assert {"loss", "energy"} <= set(metrics)
print("acceptance-ok")
"""


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, (
        f"{' '.join(cmd)} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    return result.stdout


@pytest.mark.timeout(300)
def test_wheel_installs_and_runs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "dist"
        _run(["uv", "build", "--wheel", "--out-dir", str(out_dir)])
        wheels = list(out_dir.glob("*.whl"))
        assert len(wheels) == 1

        env_dir = Path(tmp) / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env_dir)
        pip = str(env_dir / "bin" / "pip")
        python = str(env_dir / "bin" / "python")
        _run([pip, "install", "--no-deps", "--no-index", str(wheels[0])])

        result = subprocess.run(
            [python, "-c", _SMOKE],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, (
            f"smoke failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
        assert "acceptance-ok" in result.stdout


def test_wheel_contains_subpackages() -> None:
    """The wheel ships the full package tree, not just the top level."""
    import zipfile

    out_dir = Path(tempfile.mkdtemp())
    _run(["uv", "build", "--wheel", "--out-dir", str(out_dir)])
    wheel = next(out_dir.glob("*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    for sub in ("computronium/nn/__init__.py", "computronium/ontology/__init__.py"):
        assert sub in names, f"{sub} missing from wheel — flat package list regression"
    sys.stdout.write(f"wheel: {wheel.name}\n")
