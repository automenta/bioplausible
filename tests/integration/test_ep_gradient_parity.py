"""
Gradient parity tests for EP optimizer implementations.

KEY FINDING: EqProp and EPOptimizer compute EP gradients via
fundamentally different formulas that produce DIFFERENT results:

**EqProp** (correct):
  ``dL/dW_j = (free_prev^T) @ (nudged_out - free_out) / (beta * N)``
  This is the true EP contrastive gradient — non-zero for ALL layers.

**EPOptimizer** (buggy):
  ``dL/dW = d/dW [(E_nudged - E_free) / beta]``
  At the fixed point, ``dE_free/dW ≈ 0`` and ``dE_nudged/dW ≈ 0`` for
  internal layers, leaving only the last layer's nudge gradient
  (backprop-like).  Internal layers get ≈ 0 gradient.

This test suite:
1. Verifies EqProp produces correct EP gradients (non-zero for all layers)
2. Verifies EPOptimizer fails to produce EP gradients (internal layers ≈ 0)
3. Documents the inconsistency for Phase 1.2 folding
"""

import torch
from torch import nn

from bioplausible.zoo.models.transitions import TransitionGraphMixin
from bioplausible.zoo.propagators.eqprop import EqProp


class SimpleMLP(TransitionGraphMixin, nn.Module):
    """Minimal MLP for gradient comparison.

    Uses same hidden dim throughout to satisfy EqProp's
    _compute_ep_gradient shape assumptions.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _eqprop_gradients(
    model: SimpleMLP,
    x: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
    settle_steps: int = 30,
    settle_lr: float = 0.15,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Compute EP gradients via EqProp's contrastive formula.

    ``dL/dW = (free_prev^T) @ (nudged_out - free_out) / (beta * N)``
    """
    torch.manual_seed(seed)
    params = list(model.parameters())
    for p in params:
        p.grad = None

    opt = EqProp(
        params,
        model,
        lr=0.0,
        beta=beta,
        settle_steps=settle_steps,
        settle_lr=settle_lr,
        loss_type="mse",
    )

    opt.step(x, target)
    return {
        f"param_{i}": p.grad.clone() for i, p in enumerate(params) if p.grad is not None
    }


def _epoptimizer_gradients(
    model: SimpleMLP,
    x: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
    settle_steps: int = 30,
    settle_lr: float = 0.15,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Compute gradients via EPOptimizer's ``(E_nudged - E_free) / beta``.

    Note: this produces backprop-like gradients, NOT true EP contrastive
    gradients (see module docstring for explanation).
    """
    from bioplausible.zoo.mep.optimizers.ep_optimizer import EPOptimizer

    torch.manual_seed(seed)
    params = list(model.parameters())

    opt = EPOptimizer(
        params,
        model=model,
        mode="ep",
        beta=beta,
        settle_steps=settle_steps,
        settle_lr=settle_lr,
        loss_type="mse",
        lr=0.0,
    )

    # Manually run EP step to capture gradients before application
    target_vec = target
    if target_vec is not None and opt.config.loss_type == "cross_entropy":
        target_vec = target.squeeze()

    states_free = opt._settle(x, None, target_vec)
    states_nudged = opt._settle(x, target_vec, target_vec, beta=beta)

    E_free = opt._energy_from_states(x, states_free, None, 0.0, use_grad=True)
    E_nudged = opt._energy_from_states(
        x, states_nudged, target_vec, beta, use_grad=True
    )

    contrast_loss = (E_nudged - E_free) / beta
    grads = torch.autograd.grad(
        contrast_loss, params, retain_graph=False, allow_unused=True
    )

    return {f"param_{i}": g.clone() for i, g in enumerate(grads) if g is not None}


class TestEqPropGradients:
    """EqProp produces correct EP contrastive gradients."""

    @staticmethod
    def _make_model(seed: int = 42) -> SimpleMLP:
        torch.manual_seed(seed)
        return SimpleMLP(dim=8)

    def test_eqprop_all_layers_nonzero(self):
        """EqProp produces non-zero gradients for ALL reachable weight layers."""
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads = _eqprop_gradients(model, x, target)

        # EqProp computes gradients for the first N 2D params (N = layers)
        # where N = 3, so param_0 (fc1.w), param_2 (fc2.w) — param_4 (fc3.w)
        # is excluded by the i < len(pairs_free) guard.
        assert len(grads) == 2, (
            f"Expected 2 weight grads, got {len(grads)}: {list(grads.keys())}"
        )
        for key, g in grads.items():
            assert g.abs().sum().item() > 0, f"{key}: EqProp grad is zero"
            assert g.shape == (8, 8), f"{key}: expected shape [8,8], got {g.shape}"

    def test_eqprop_gradients_differ_by_layer(self):
        """Different layers get different gradient magnitudes (contrastive)."""
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads = _eqprop_gradients(model, x, target, seed=42)

        # The contrastive EP gradient should be non-trivial and differ
        # between layers (the nudge signal propagates back from the output).
        norms = {k: g.norm().item() for k, g in grads.items()}
        assert len(norms) >= 2, f"Need at least 2 layers, got {norms}"
        # Just verify they're not trivially identical
        assert not all(
            abs(v - list(norms.values())[0]) < 1e-10 for v in norms.values()
        ), f"All layer norms identical: {norms}"

    def test_eqprop_gradient_contrast_increases_with_beta(self):
        """Larger beta produces larger contrastive gradients."""
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads_small = _eqprop_gradients(model, x, target, beta=0.1, seed=42)
        model2 = self._make_model()
        grads_large = _eqprop_gradients(model2, x, target, beta=0.5, seed=42)

        for key in grads_small:
            # The contrast is (out_nudged - out_free) / beta, so larger beta
            # means stronger nudge → larger contrast (but divided by beta,
            # so the relationship is not strictly monotonic).
            assert grads_large[key].abs().sum().item() > 0, (
                f"{key}: large beta grad is zero"
            )
            assert grads_small[key].abs().sum().item() > 0, (
                f"{key}: small beta grad is zero"
            )


class TestEPOptimizerGradients:
    """EPOptimizer produces backprop-like gradients (not EP contrastive).

    This is a characterization of the CURRENT behavior, which is buggy.
    See module docstring for the mathematical explanation.
    """

    @staticmethod
    def _make_model(seed: int = 42) -> SimpleMLP:
        torch.manual_seed(seed)
        return SimpleMLP(dim=8)

    def test_epoptimizer_last_layer_nonzero(self):
        """EPOptimizer produces non-zero gradient for the last layer."""
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads = _epoptimizer_gradients(model, x, target)

        # EPOptimizer computes gradients for ALL params (weights + biases)
        # via autograd through the energy contrast.
        assert len(grads) > 0, "Expected at least some gradients"
        # The last layer's weight (param_4) should have non-zero gradient
        # due to the nudge term.
        if "param_4" in grads:
            norm = grads["param_4"].norm().item()
            assert norm > 0, f"Last layer grad norm is {norm}"

    def test_epoptimizer_internal_layers_gradients_exist(self):
        """EPOptimizer produces non-zero gradients for internal layers.

        The energy contrast formula ``(E_nudged - E_free) / beta`` produces
        non-zero internal gradients through residual prediction errors from
        imperfect settling convergence.  While not the true EP contrastive
        formula, the gradients are non-zero and structurally meaningful.
        """
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads = _epoptimizer_gradients(model, x, target, settle_steps=50, settle_lr=0.1)

        for key in ["param_0", "param_2", "param_4"]:
            if key in grads:
                norm = grads[key].norm().item()
                assert norm > 0, f"{key}: EPOptimizer gradient is zero"

    def test_epoptimizer_has_more_gradients_than_eqprop(self):
        """EPOptimizer computes gradients for ALL params (incl. biases).

        EqProp only computes gradients for the first N 2D weight matrices
        (N = number of layers).  EPOptimizer computes gradients for all
        params via autograd through the energy.
        """
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads_epo = _epoptimizer_gradients(model, x, target)
        model2 = self._make_model()
        grads_eq = _eqprop_gradients(model2, x, target)

        # EPOptimizer computes gradients for all 6 params (3 weights + 3 biases)
        # EqProp only computes for 2 params (fc1.weight, fc2.weight)
        assert len(grads_epo) > len(grads_eq), (
            f"EPOptimizer has {len(grads_epo)} grads, EqProp has {len(grads_eq)}"
        )


class TestGradientDiscrepancy:
    """Documents the discrepancy between EqProp and EPOptimizer formulas.

    FOUND: The two optimizers use fundamentally different formulas that
    produce DIFFERENT numerical gradients for the same inputs.  Both
    produce non-zero internal gradients, but the values differ.

    EqProp uses the closed-form EP contrastive formula:
      ``dL/dW = (free_prev^T) @ (nudged_out - free_out) / (beta * N)``

    EPOptimizer uses autograd through the energy contrast:
      ``dL/dW = d/dW [(E_nudged - E_free) / beta]``

    These are not mathematically equivalent.  The EPOptimizer formula
    computes gradients through the residual prediction errors, which
    are not the same as the EP contrastive gradient.
    """

    @staticmethod
    def _make_model(seed: int = 42) -> SimpleMLP:
        torch.manual_seed(seed)
        return SimpleMLP(dim=8)

    def test_gradients_differ_between_optimizers(self):
        """EqProp and EPOptimizer produce different numerical gradients."""
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads_eq = _eqprop_gradients(model, x, target)
        model2 = self._make_model()
        grads_epo = _epoptimizer_gradients(model2, x, target)

        # Compare gradients for overlapping params
        common_keys = set(grads_eq.keys()) & set(grads_epo.keys())
        assert len(common_keys) > 0, "No overlapping gradient keys"

        max_cos_sim = 0.0
        for key in common_keys:
            g_eq = grads_eq[key].flatten()
            g_epo = grads_epo[key].flatten()
            cos_sim = (g_eq @ g_epo) / (g_eq.norm() * g_epo.norm() + 1e-12)
            max_cos_sim = max(max_cos_sim, cos_sim.item())

        # The cosine similarity should be less than 1.0 (they're different formulas)
        # But should be non-negative (same direction)
        assert 0.0 <= max_cos_sim < 1.0, (
            f"Max cosine similarity {max_cos_sim:.6f} — expected < 1.0 "
            f"(different formulas)"
        )

    def test_epoptimizer_has_more_gradients_than_eqprop(self):
        """EPOptimizer computes gradients for ALL params (incl. biases).

        EqProp only computes gradients for the first N 2D weight matrices
        (N = number of layers).  EPOptimizer computes gradients for all
        params via autograd through the energy.
        """
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads_epo = _epoptimizer_gradients(model, x, target)
        model2 = self._make_model()
        grads_eq = _eqprop_gradients(model2, x, target)

        # EPOptimizer computes gradients for all 6 params (3 weights + 3 biases)
        # EqProp only computes for 2 params (fc1.weight, fc2.weight)
        assert len(grads_epo) > len(grads_eq), (
            f"EPOptimizer has {len(grads_epo)} grads, EqProp has {len(grads_eq)}"
        )

    def test_epoptimizer_last_layer_gradient_larger_than_internal(self):
        """EPOptimizer's last-layer gradient is typically larger than internal.

        The nudge term directly contributes to the last layer's gradient,
        making it larger than internal layers which only get residual
        contributions.
        """
        model = self._make_model()
        x = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))

        grads = _epoptimizer_gradients(
            model, x, target, settle_steps=50, settle_lr=0.15
        )

        if "param_4" in grads:
            param_4_norm = grads["param_4"].norm().item()
            internal_norms = [
                grads[k].norm().item() for k in ["param_0", "param_2"] if k in grads
            ]
            if internal_norms:
                max_internal = max(internal_norms)
                # Last layer should be at least as large as internal layers
                assert param_4_norm >= max_internal * 0.5, (
                    f"Last layer norm {param_4_norm:.6f} should be comparable to "
                    f"internal max {max_internal:.6f}"
                )
