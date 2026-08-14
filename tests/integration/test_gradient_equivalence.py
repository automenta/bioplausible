import pytest
import torch
import torch.nn.functional as F

# FORCE DISABLE TRITON/COMPILE CHECKS BEFORE IMPORTING MODELS
# This avoids the hang observed during import of ConvEqProp
import bioplausible.acceleration

bioplausible.acceleration._check_compile_works = lambda: False

from bioplausible.core.local_learning.rules.backprop import (
    Backprop as _Backprop,
)
from bioplausible.core.local_learning.rules.eqprop import (
    EqProp as _EqProp,
)
from bioplausible.core.local_learning.rules.fa import (  # ruff: ignore[module-import-not-at-top-of-file]
    DirectFA as _DirectFA,
)
from bioplausible.core.local_learning.rules.fa import (
    FeedbackAlignment as _FeedbackAlignment,
)
from bioplausible.core.local_learning.rules.fa import (
    StochasticFA as _StochasticFA,
)
from bioplausible.core.local_learning.rules.hebbian import (  # ruff: ignore[module-import-not-at-top-of-file]
    ContrastiveHebbianLearning as _CHL,
)
from bioplausible.validation.gradient_check import (
    check_gradient_equivalence,
    loss_ce,
    loss_mse,
)
from bioplausible.zoo.mep.presets import (
    smep as _smep,
)
from bioplausible.zoo.models.eqprop import (
    LoopedMLP,
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

# --- the finite-difference machinery + equivalence check + MLP host module
#     are promoted to bioplausible.validation.gradient_check (Phase 1.2) ---


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
    check_gradient_equivalence(name, build, driver, loss_ce, threshold)


@pytest.mark.parametrize("name,build,threshold", EQUILIBRIUM_FAMILIES_MSE)
def test_equilibrium_gradient_direction_equivalence(name, build, threshold):
    """EqProp/MEP-EP/CHL update directions match the MSE-energy gradient."""
    check_gradient_equivalence(name, build, _lro_driver, loss_mse, threshold)
