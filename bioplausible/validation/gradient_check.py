"""Gradient-equivalence gate (architecture §7#2, RESEARCH §5.2).

Promoted from the finite-difference equivalence helpers originally embedded in
``tests/integration/test_gradient_equivalence.py``. This module is the single
source for the P0 gradient-equivalence gate: every gradient-aligned propagator
must align its local learning-rule update direction with a finite-difference
gradient of its loss before it is admitted to the parity tier.

The direction of a local rule is validated against a central-difference FD
gradient of the *task loss the rule actually descends*: cross-entropy for
backprop/FA/MEP-backprop families, and the MSE energy for equilibrium rules
(EqProp/MEP-EP/CHL). Spiking/STDP and forward-only rules (FF, PEPITA) have no
defined gradient direction vs. the task loss and are excluded by design.

The integration test consumes this module (zero behavioural change); the nightly
``biopl-repro-check`` gate invokes :func:`check_family` per family.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.utils import seed_everything

__all__ = [
    "GradientCheckError",
    "GradientEquivalenceMLP",
    "MetricRule",
    "check_family",
    "check_gradient_equivalence",
    "finite_diff_gradient",
    "local_direction",
    "loss_ce",
    "loss_mse",
]

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
PropBuilder = Callable[[list[nn.Parameter], nn.Module], object]
Driver = Callable[[object, nn.Module, torch.Tensor, torch.Tensor], None]


class GradientCheckError(RuntimeError):
    """Raised when a family's local learning-rule gradient fails the gate."""


class MetricRule:
    """Aggregate of (name, builder, driver, loss, threshold) for one family."""

    __slots__ = ("build", "driver", "loss_fn", "name", "threshold")

    def __init__(
        self,
        name: str,
        build: PropBuilder,
        driver: Driver,
        loss_fn: LossFn,
        threshold: float,
    ) -> None:
        self.name = name
        self.build = build
        self.driver = driver
        self.loss_fn = loss_fn
        self.threshold = threshold


class GradientEquivalenceMLP(nn.Module):
    """Small bias-free MLP; all params are transition weights for EP/CHL."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 8, output_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def transition_modules(self):
        return [m for m in self.net if isinstance(m, nn.Linear)]


def finite_diff_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: LossFn,
    eps: float = 1e-2,
) -> torch.Tensor:
    """Central-difference finite-difference gradient of ``loss_fn(model(x), y)``.

    Args:
        model: Model whose parameters define the FD gradient.
        x: Input batch.
        y: Target batch.
        loss_fn: Loss the gradient is taken of.
        eps: Central-difference step.

    Returns:
        Flattened FD gradient vector over all parameters.
    """
    params = list(model.parameters())
    grads: list[torch.Tensor] = []
    for p in params:
        g = torch.zeros_like(p)
        flat = p.data.view(-1)
        for i in range(p.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps
            loss_plus = loss_fn(model(x), y)
            flat[i] = orig - eps
            loss_minus = loss_fn(model(x), y)
            flat[i] = orig
            g.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)
        grads.append(g)
    return torch.cat([v.reshape(-1) for v in grads])


def local_direction(model: nn.Module) -> torch.Tensor:
    """Flatten the in-place learning-rule gradients accumulated on ``model``."""
    return torch.cat([
        p.grad.reshape(-1) for p in model.parameters() if p.grad is not None
    ])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def loss_ce(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(out, y)


def loss_mse(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out, F.one_hot(y, num_classes=out.shape[1]).float())


def check_gradient_equivalence(
    name: str,
    build_opt: PropBuilder,
    driver: Driver,
    loss_fn: LossFn,
    threshold: float,
) -> tuple[float, float, float]:
    """Assert one family's local update direction aligns with the FD gradient.

    Trains the rule on a seeded tiny MLP, computes the local direction and the
    autograd reference gradient, verifies the FD machinery reproduces autograd,
    then asserts the local rule aligns with the FD-verified gradient.

    Args:
        name: Family label (for diagnostics).
        build_opt: Builds the propagator from ``(parameters, model)``.
        driver: Runs one update given ``(opt, model, x, y)``.
        loss_fn: Loss the rule descends (CE or MSE-energy).
        threshold: Minimum cosine similarity between rule and FD gradient.

    Returns:
        ``(fd_cosine, rule_cosine, threshold)`` triple for reporting.

    Raises:
        GradientCheckError: If FD machinery diverges from autograd or the rule
            direction drifts below ``threshold``.
    """
    seed_everything(0)
    x = torch.randn(16, 8)
    y = torch.randint(0, 5, (16,))

    model = GradientEquivalenceMLP()
    sd = model.state_dict()
    twin = GradientEquivalenceMLP()
    twin.load_state_dict(sd)

    opt = build_opt(list(model.parameters()), model)
    driver(opt, model, x, y)
    d = local_direction(model)

    twin.zero_grad()
    g = torch.cat(
        torch.autograd.grad(loss_fn(twin(x), y), list(twin.parameters()))
    ).reshape(-1)
    gf = finite_diff_gradient(twin, x, y, loss_fn)

    fd_cos = _cosine(g, gf)
    if fd_cos <= 0.99:  # ruff: ignore[magic-value-comparison]  (FD-vs-autograd sanity bound)
        raise GradientCheckError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            f"{name}: FD machinery diverged (cos={fd_cos:.3f})"
        )
    rule_cos = _cosine(d, gf)
    if rule_cos < threshold:
        raise GradientCheckError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            f"{name}: local gradient direction drifted "
            f"(cos={rule_cos:.3f} < {threshold})"
        )
    return fd_cos, rule_cos, threshold


def check_family(
    metric: MetricRule,
) -> tuple[float, float]:
    """Run the equivalence gate for one preconfigured family rule.

    Args:
        metric: An assembled :class:`MetricRule`.

    Returns:
        ``(fd_cosine, rule_cosine)`` for the family.
    """
    _, rule_cos, _ = check_gradient_equivalence(
        metric.name,
        metric.build,
        metric.driver,
        metric.loss_fn,
        metric.threshold,
    )
    return rule_cos, metric.threshold
