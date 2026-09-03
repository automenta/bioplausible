"""Unit tests for the θ-invariance audit harness (RESEARCH3 PR-2)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from computronium.core.plasticity.theta_audit import (
    ThetaAuditReport,
    ThetaInvarianceAudit,
    require_frozen,
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.randn(4, 4))
        self.psi = nn.Parameter(torch.randn(4), requires_grad=False)


def _select_theta(name: str, _p: nn.Parameter) -> bool:
    return name == "theta"


class TestThetaInvarianceAudit:
    def test_invariant_when_untouched(self):
        model = _Model()
        model.theta.requires_grad_(False)
        with ThetaInvarianceAudit(model, selector=_select_theta) as audit:
            _ = model.psi * 2
        report = audit.report
        assert report is not None
        assert report.invariant
        assert report.max_abs_change == 0.0
        assert report.changed_keys == ()
        assert report.frozen_on_entry

    def test_detects_drift(self):
        model = _Model()
        model.theta.requires_grad_(False)
        with ThetaInvarianceAudit(model, selector=_select_theta) as audit:  # ruff: ignore[multiple-with-statements]
            with torch.no_grad():
                model.theta.add_(1e-3)
        report = audit.report
        assert report is not None
        assert not report.invariant
        assert report.changed_keys == ("theta",)
        assert report.max_abs_change == pytest.approx(1e-3, abs=1e-6)
        assert not report.is_within(1e-6)
        assert report.is_within(1e-2)

    def test_entry_raises_on_trainable_selection(self):
        model = _Model()
        with pytest.raises(RuntimeError, match="trainable"):  # ruff: ignore[multiple-with-statements]
            with ThetaInvarianceAudit(model, selector=_select_theta):
                pass

    def test_expect_frozen_false_allows_trainable(self):
        model = _Model()
        with ThetaInvarianceAudit(
            model, selector=_select_theta, expect_frozen=False
        ) as audit:
            pass
        report = audit.report
        assert report is not None
        assert not report.frozen_on_entry

    def test_no_report_on_exception(self):
        model = _Model()
        model.theta.requires_grad_(False)
        audit_holder = []

        def _run() -> None:
            with ThetaInvarianceAudit(model, selector=_select_theta) as audit:
                audit_holder.append(audit)
                raise ValueError("boom")

        with pytest.raises(ValueError):
            _run()
        assert audit_holder[0].report is None

    def test_report_defaults(self):
        report = ThetaAuditReport(
            frozen_on_entry=True, max_abs_change=0.0, changed_keys=()
        )
        assert report.invariant and report.is_within(0.0)


def test_require_frozen_raises_on_trainable():
    params: dict[str, nn.Parameter] = {
        "ok": nn.Parameter(torch.zeros(1), requires_grad=False)
    }
    require_frozen(params)
    params["bad"] = nn.Parameter(torch.zeros(1))
    with pytest.raises(RuntimeError, match="bad"):
        require_frozen(params)
