"""Energy Landscape Visualization — Sprint 2.2.

Produces a 2D slice of the energy surface ``E(w)`` around a trained model's
weights, rendered as a contour plot with gradient-flow arrows. This lets
researchers see local minima / saddle geometry for energy-based models
(EqProp, EquiTile) at a glance.

Approach
--------
We cannot plot the full parameter space, so we restrict to a 2D subspace
spanned by two orthonormal directions through the trained weight vector
``w*``:

1. ``d1`` — the direction of steepest descent (negative gradient) at ``w*``.
2. ``d2`` — a normalized component of the *second* principal direction or a
   random orthogonal vector, so the plane is not degenerate.

We then sweep ``E(w* + α·d1 + β·d2)`` over an ``(α, β)`` grid. The energy
``E`` is either the model's own ``model.energy(x, y)`` when it implements the
:class:`EnergyModel` protocol, else the cross-entropy loss on a fixed batch
(a faithful scalar proxy for energy-based learning rules).

Integration
-----------
``:func:`plot_energy_landscape`` writes an ``energy_landscape_{model}_{task}.png``
file and returns the figure path. Callable from the CLI or the NiceGUI demo.
"""

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from bioplausible.core.energy_model import is_energy_model

__all__ = [
    "EnergyLandscape",
    "compute_energy_landscape",
    "plot_energy_landscape",
]


@dataclass(frozen=True, slots=True)
class EnergyLandscape:
    """The evaluated 2D energy slice plus the plotting metadata."""

    model_name: str
    task_name: str
    alphas: np.ndarray  # (N,)
    betas: np.ndarray  # (M,)
    energy: np.ndarray  # (N, M) — row = α, col = β
    d1_norm: float  # norm of the gradient direction used for the α axis
    param_count: int

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Persist the energy grid to an ``.npz`` archive."""
        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            model_name=self.model_name,
            task_name=self.task_name,
            alphas=self.alphas,
            betas=self.betas,
            energy=self.energy,
            d1_norm=self.d1_norm,
            param_count=self.param_count,
        )
        return out


def _orthonormal_directions(
    params: Sequence[torch.Tensor], grad: list[torch.Tensor], seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two orthonormal flat directions through the parameter vector.

    ``d1`` is the unit-norm concatenated gradient; ``d2`` is the unit-norm
    *random* direction orthogonalized against ``d1``.
    """

    def flat(tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in tensors])

    g = flat(grad)
    d1 = g / (g.norm() + 1e-8)

    rng = np.random.default_rng(seed)
    r = rng.standard_normal(d1.shape)
    r = torch.from_numpy(r).to(dtype=g.dtype, device=g.device)
    d2 = r - (r @ d1) * d1
    d2 = d2 / (d2.norm() + 1e-8)
    return d1, d2


def _energy_at(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    params: Sequence[torch.Tensor],
    deltas: list[torch.Tensor],
    use_model_energy: bool,
) -> float:
    """Evaluate scalar energy at ``params + deltas`` without autograd pollution."""
    with torch.no_grad():
        for p, d in zip(params, deltas):
            p.add_(d)
        total = 0.0
        try:
            if use_model_energy:
                total = float(model.energy(x, y).item())
            else:
                logits = model(x)
                total = float(nn.functional.cross_entropy(logits, y).item())
        finally:
            # Restore original weights regardless of exceptions.
            for p, d in zip(params, deltas):
                p.sub_(d)
    return total


def compute_energy_landscape(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    task_name: str,
    radius: float = 1.0,
    grid: int = 21,
    seed: int = 0,
) -> EnergyLandscape:
    """Evaluate the energy surface on a 2D plane through ``model``'s weights.

    Args:
        model: Trained model whose parameters define the landscape origin.
        x, y: Fixed evaluation batch.
        model_name, task_name: Metadata for the output filename.
        radius: Max perturbation magnitude (in units of the gradient norm)
            along each axis.
        grid: Number of points per axis (odd → includes origin).
        seed: RNG seed for the orthogonal second direction.

    Returns:
        A populated :class:`EnergyLandscape`.
    """
    model.eval()
    params = list(p for p in model.parameters() if p.requires_grad)
    if not params:
        raise ValueError("model has no trainable parameters")

    # Gradient direction at w* via a single backward pass.
    xb, yb = x[:64], y[:64]
    model.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = nn.functional.cross_entropy(logits, yb)
    loss.backward()
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    model.zero_grad(set_to_none=True)

    use_model_energy = is_energy_model(model)
    d1, d2 = _orthonormal_directions(params, grads, seed=seed)
    d1_norm = torch.cat([g.reshape(-1) for g in grads]).norm().item()

    p_flat = list(params)
    alphas = np.linspace(-radius, radius, grid)
    betas = np.linspace(-radius, radius, grid)
    energy = np.zeros((grid, grid))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            deltas = _flat_to_params(d1, d2, params, alpha, beta)
            energy[i, j] = _energy_at(model, xb, yb, p_flat, deltas, use_model_energy)

    return EnergyLandscape(
        model_name=model_name,
        task_name=task_name,
        alphas=alphas,
        betas=betas,
        energy=energy,
        d1_norm=d1_norm,
        param_count=sum(p.numel() for p in params),
    )


def _flat_to_params(
    d1: torch.Tensor,
    d2: torch.Tensor,
    params: Sequence[torch.Tensor],
    alpha: float,
    beta: float,
) -> list[torch.Tensor]:
    """Map ``(alpha, beta)`` onto per-parameter delta tensors."""
    splits = [p.numel() for p in params]
    flat = alpha * d1 + beta * d2
    deltas = []
    offset = 0
    for n in splits:
        deltas.append(flat[offset : offset + n].reshape(params[len(deltas)].shape))
        offset += n
    return deltas


def plot_energy_landscape(
    landscape: EnergyLandscape,
    output_dir: str | pathlib.Path = "results/figures",
    cmap: str = "viridis",
) -> pathlib.Path:
    """Render the energy slice as a contour plot with gradient-flow arrows.

    Matplotlib is imported lazily so this module stays import-cheap for
    headless / CI usage (AGENTS.md import hygiene).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"energy_landscape_{landscape.model_name}_{landscape.task_name}.png"
    fig_path = out / fname

    fig, ax = plt.subplots(figsize=(7, 6))
    X, Y = np.meshgrid(landscape.alphas, landscape.betas)

    # Gradient flow arrows: negative gradient of the energy surface.
    gx, gy = np.gradient(landscape.energy)
    skip = slice(None, None, max(1, len(landscape.alphas) // 10))

    ax.contourf(X, Y, landscape.energy.T, levels=30, cmap=cmap)
    cf = ax.contour(X, Y, landscape.energy.T, levels=12, colors="k", linewidths=0.5)
    ax.clabel(cf, inline=True, fontsize=6)
    ax.quiver(
        X[skip, skip],
        Y[skip, skip],
        -gx[skip, skip],
        -gy[skip, skip],
        color="white",
        alpha=0.7,
        scale=20,
    )
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.scatter([0], [0], marker="*", s=180, c="red", label="trained weights w*")
    ax.set_xlabel(r"perturbation along −∇E (units of $|\nabla E|$)")
    ax.set_ylabel(r"perturbation along orthogonal dir")
    ax.set_title(
        f"Energy landscape — {landscape.model_name} ({landscape.task_name})\n"
        f"{landscape.param_count} params"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path
