"""Deep local Hebbian chain with per-layer activity normalization (R11.3.14).

Plain Hebbian chains die at depth: per-layer gain != 1 compounds
super-exponentially (runaway to inf, or decay to zero), so layers past ~10
receive no meaningful signal and never learn. The fix is structural gain
control — the same role μPC init plays for error-based rules:

    spectral renorm (unit layer gain at init, bounds W)
    + tanh (bounded nonlinearity)
    + Oja decay (stability: Δw = η(y·a − y²·w), weights stay bounded)
    + activity renorm (unit RMS after every layer — amplitude is a
      constant carrier, direction carries the information)
    + linear readout for evaluation (standard local-feature demo)

Legacy recipe reference: `spectral_norm` + `tanh_()` + Oja
`addcmul_(y_sq, W, -lr)` in `computronium/zoo/models/hebbian.py@8d8de04b^`
(zoo deleted). Probe evidence: `scripts/probes/deep_hebbian_chain.py`
(hebbian tile chain explodes at depth 50+, one local_update NaNs it).

Measured regime (2026-09-04, synthetic 32→32×L, batch 64, 32 steps):

    per-layer signal norms O(1) at depth 10/50/100 — the runaway-gain /
    NaN pathology is gone; unnormalized control decays to ~1e-14.
    Dominant-direction task (2-class, ±v means): readout 1.000 at every
    depth — the chain transmits its dominant direction indefinitely.
    Direction-coded 10-class task (orthogonal means): L1 1.00 → L10
    0.52 → L100 0.20 (> 0.1 chance) — activity covariance effective
    rank collapses 5.1 (L2) → 1.5 (L10) under compounding tanh
    distortion + renormalization + Oja spectral sharpening. Sanger
    (tril) variant, gain scaling, and spectral renorm during training
    do NOT rescue the subspace (all ≈ 0.2 at L100): renorm amplifies
    whatever the spectrum favors each layer, so local Hebbian chains
    carry the dominant direction to any depth and progressively discard
    the rest. This is the Hebbian instance of the depth boundary
    (error-based rules die by telescoping decay, unnormalized local
    chains by runaway gain, normalized Oja chains by subspace collapse).

"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_EPS = 1e-8


def _unit_rms(x: Tensor) -> Tensor:
    return x / x.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(_EPS)


def _spectral_renorm(w: Tensor) -> Tensor:
    s = torch.linalg.svdvals(w)[0]
    return w / s.clamp_min(_EPS)


class DeepHebbianChain(nn.Module):
    """Feedforward Hebbian chain that carries signal at any depth.

    Each layer applies `tanh(W @ a)` where `a` is the previous layer's
    unit-RMS activity; the output is renormalized to unit RMS. Weights are
    spectrally normalized at init and updated in place with batch Oja's
    rule on the normalized activities (pure Hebbian strengthening + decay,
    no backprop, no nudging).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        learning_rate: float = 1e-3,
        normalize: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.normalize = normalize
        dims = (input_dim, *([hidden_dim] * num_layers))
        self.weights = nn.ParameterList(
            nn.Parameter(
                _spectral_renorm(torch.randn(hidden_dim, d, generator=generator)),
                requires_grad=False,
            )
            for d in dims[:-1]
        )

    @torch.no_grad()
    def forward(self, x: Tensor) -> list[Tensor]:
        """Per-layer activations (each unit-RMS when `normalize`)."""
        acts = [x]
        a = x
        for w in self.weights:
            if self.normalize:
                a = _unit_rms(a)
            y = torch.tanh(a @ w.T)
            if self.normalize:
                y = _unit_rms(y)
            acts.append(y)
            a = y
        return acts

    @torch.no_grad()
    def local_update(self, x: Tensor) -> None:
        """One batch Oja update per layer, in place."""
        acts = self(x)
        n = x.shape[0]
        for i in range(len(self.weights)):
            w = self.weights[i]
            a, y = acts[i], acts[i + 1]
            if self.normalize:
                a, y = _unit_rms(a), _unit_rms(y)
            w += self.learning_rate * (
                y.T @ a / n - y.pow(2).mean(dim=0).unsqueeze(1) * w
            )

    @torch.no_grad()
    def layer_norms(self, x: Tensor) -> list[float]:
        """Mean per-sample L2 norm of each layer's pre-renormalization output.

        The gain-control diagnostic: with normalization the carrier is
        restored to unit RMS every layer, so O(1) norms here mean the
        learned weights neither amplify nor decay the signal they pass on.
        """
        norms: list[float] = []
        a = x
        for w in self.weights:
            if self.normalize:
                a = _unit_rms(a)
            y = torch.tanh(a @ w.T)
            norms.append(y.norm(dim=1).mean().item())
            if self.normalize:
                y = _unit_rms(y)
            a = y
        return norms
