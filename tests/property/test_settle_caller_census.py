"""R7 probe #5 (imp-47): settle() caller census — no caller may ignore the
returned state.

Canonical mutation contract (pinned on the ``StateDynamics`` protocol):
``settle()`` always returns the state to use; implementations may rebuild
rather than mutate, so a caller reading the input state after the call reads
pre-settle activations — the defect class that manufactured the fidelity
probe's false "no-effect" verdict (imp-27 ancestor).

Enforcement: AST scan of ``computronium/`` rejecting bare expression
statements whose value is a ``.settle(...)`` call. The scan self-checks:
it must see a floor number of real call sites (can't go silently blind)
and must flag a planted violation (probe-the-probe).
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "computronium"
CALLEE_NAME = "settle"
MIN_KNOWN_CALL_SITES = 20

VIOLATION_SNIPPET = """
def bad(dynamics, state, geometry, substrate):
    dynamics.settle(state, geometry, substrate)
    return state.activations
"""

CLEAN_SNIPPET = """
def good(dynamics, state, geometry, substrate):
    settled = dynamics.settle(state, geometry, substrate, target=None)
    return settled.activations
"""


def _flagged_lines(source: str) -> list[int]:
    """Line numbers of bare Expr statements calling ``.settle(...)``."""
    tree = ast.parse(source)
    flagged: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == CALLEE_NAME:
                flagged.append(node.lineno)
    return flagged


def _settle_call_sites() -> list[tuple[Path, int]]:
    """All ``.settle(...)`` call sites under the package (census coverage)."""
    sites: list[tuple[Path, int]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == CALLEE_NAME
            ):
                sites.append((path, node.lineno))
    return sites


def test_census_scan_is_not_blind() -> None:
    assert len(_settle_call_sites()) >= MIN_KNOWN_CALL_SITES


def test_detector_flags_planted_violation() -> None:
    assert _flagged_lines(VIOLATION_SNIPPET) == [3]


def test_detector_passes_bound_caller() -> None:
    assert _flagged_lines(CLEAN_SNIPPET) == []


def test_no_settle_call_discards_return_value() -> None:
    offenders = []
    for path, _line in _settle_call_sites():
        flagged = _flagged_lines(path.read_text(encoding="utf-8"))
        offenders.extend(f"{path.relative_to(PACKAGE_ROOT)}:{n}" for n in flagged)
    assert not offenders, (
        "settle() return value ignored at: "
        + ", ".join(offenders)
        + " — bind and use the returned state (imp-47 contract)"
    )
