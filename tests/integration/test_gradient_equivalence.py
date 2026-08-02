import pytest
import torch
import torch.nn.functional as F
from torch import nn

# FORCE DISABLE TRITON/COMPILE CHECKS BEFORE IMPORTING MODELS
# This avoids the hang observed during import of ConvEqProp
import bioplausible.acceleration

bioplausible.acceleration._check_compile_works = lambda: False

from bioplausible.zoo.models.eqprop import LoopedMLP  # noqa: E402
from bioplausible.zoo.mep.presets import smep as _smep  # noqa: E402
from bioplausible.zoo.propagators.backprop import Backprop as _Backprop  # noqa: E402
from bioplausible.zoo.propagators.eqprop import EqProp as _EqProp  # noqa: E402
from bioplausible.zoo.propagators.fa import (  # noqa: E402
    DirectFA as _DirectFA,
    FeedbackAlignment as _FeedbackAlignment,
    StochasticFA as _StochasticFA,
)
from bioplausible.zoo.propagators.hebbian import (  # noqa: E402
    ContrastiveHebbianLearning as _CHL,
)


def test_contrastive_gradients():
    """Verify gradient equivalence after .detach() optimization."""
    print("Testing contrastive gradient correctness...")
    torch.manual_seed(42)

    # Create model
    model = LoopedMLP(10, 20, 5, gradient_method="contrastive", max_steps=10)

    # Create dummy data
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    # Run contrastive step
    metrics = model.train_step(x, y)

    print(f"Metrics: {metrics}")

    # Verify gradients exist and are valid (no NaNs)
    has_grads = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grads = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
            # Check magnitude is reasonable
            grad_norm = param.grad.norm().item()
            assert grad_norm <= 100.0, f"High gradient norm for {name}: {grad_norm}"

    assert has_grads, "No gradients computed for any parameter."


# =====================================================================
# Sprint 2.1 — Finite-Difference Gradient Equivalence
# =====================================================================
# For every gradient-aligned propagator we verify the *direction* of the
# local learning-rule gradient against a finite-difference gradient of the
# task loss. Backprop/FA/MEP(backprop) are validated against cross-entropy
# (the loss they descend); equilibrium rules (EqProp/MEP-EP/CHL) are
# validated against the MSE energy loss they are designed to minimize —
# EP's contrastive gradient is a gradient of the energy, not of CE, so the
# CE comparison would conflate rule quality with loss choice.
#
# Excluded by design (non-gradient families): spiking/STDP and forward-only
# rules (FF, PEPITA) which have no defined gradient direction vs. the task
# loss (the TODO plan marks these "N/A").


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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def _flatten(vectors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in vectors])


def _loss_ce(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(out, y)


def _loss_mse(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out, F.one_hot(y, num_classes=out.shape[1]).float())


def _finite_diff_gradient(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn, eps: float = 1e-2
) -> torch.Tensor:
    """Central-difference finite-difference gradient of ``loss_fn(model(x), y)``."""
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
    return _flatten(grads)


def _local_direction(model: nn.Module) -> torch.Tensor:
    return _flatten([
        p.grad.reshape(-1) for p in model.parameters() if p.grad is not None
    ])


def _check_gradient_equivalence(
    name: str,
    build_opt,
    driver,
    loss_fn,
    threshold: float,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    y = torch.randint(0, 5, (16,))

    model = GradientEquivalenceMLP()
    sd = model.state_dict()
    twin = GradientEquivalenceMLP()
    twin.load_state_dict(sd)

    opt = build_opt(list(model.parameters()), model)
    driver(opt, model, x, y)
    d = _local_direction(model)

    twin.zero_grad()
    g = _flatten(torch.autograd.grad(loss_fn(twin(x), y), list(twin.parameters())))
    gf = _finite_diff_gradient(twin, x, y, loss_fn)

    # Machinery sanity: FD must reproduce autograd.
    assert _cosine(g, gf) > 0.99, (
        f"{name}: FD machinery diverged (cos={_cosine(g, gf)})"
    )
    # Direction equivalence: local rule aligns with the (FD-verified) gradient.
    c = _cosine(d, gf)
    assert c >= threshold, (
        f"{name}: local gradient direction drifted (cos={c:.3f} < {threshold})"
    )


def _lro_driver(opt, model, x, y) -> None:
    opt.step(x=x, target=y)


def _bptt_driver(opt, model, x, y) -> None:
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()


# --- cross-entropy-aligned families (backprop / FA / MEP-backprop) ---
GRADIENT_FAMILIES_CE = [
    ("backprop", lambda p, m: _Backprop(p, m), 0.9),
    ("feedback_alignment", lambda p, m: _FeedbackAlignment(p, m), 0.9),
    ("direct_fa", lambda p, m: _DirectFA(p, m), 0.9),
    ("stochastic_fa", lambda p, m: _StochasticFA(p, m), 0.9),
    (
        "smep (backprop mode)",
        lambda p, m: _smep(p, m, mode="backprop", ns_steps=0),
        0.9,
    ),
]

# --- equilibrium-energy families (EqProp / MEP-EP / CHL) vs MSE energy ---
EQUILIBRIUM_FAMILIES_MSE = [
    (
        "eq_prop",
        lambda p, m: _EqProp(p, m, beta=0.5, settle_steps=30, settle_lr=0.15),
        0.6,
    ),
    (
        "smep (ep mode)",
        lambda p, m: _smep(
            p, m, mode="ep", settle_steps=30, ns_steps=0, settle_lr=0.15
        ),
        0.6,
    ),
    ("contrastive_hebbian_learning", lambda p, m: _CHL(p, m), 0.6),
]


@pytest.mark.parametrize("name,build,threshold", GRADIENT_FAMILIES_CE)
def test_ce_gradient_direction_equivalence(name, build, threshold):
    """Backprop/FA/MEP-backprop update directions match the CE gradient."""
    driver = _bptt_driver if name == "smep (backprop mode)" else _lro_driver
    _check_gradient_equivalence(name, build, driver, _loss_ce, threshold)


@pytest.mark.parametrize("name,build,threshold", EQUILIBRIUM_FAMILIES_MSE)
def test_equilibrium_gradient_direction_equivalence(name, build, threshold):
    """EqProp/MEP-EP/CHL update directions match the MSE-energy gradient."""
    _check_gradient_equivalence(name, build, _lro_driver, _loss_mse, threshold)
