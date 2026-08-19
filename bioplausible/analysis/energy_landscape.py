"""Energy Landscape Visualization — Enhanced.

Produces 2D/3D slices of the energy surface E(w) around a trained model's
weights, rendered as contour plots, 3D surfaces, with gradient-flow arrows.
Supports multiple direction selection methods and curvature analysis.
"""

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch import nn

from bioplausible.core.ebm import is_energy_model

__all__ = [
    "DirectionMethod",
    "EnergyLandscape",
    "LandscapeSlice",
    "analyze_landscape_curvature",
    "compute_energy_landscape",
    "compute_hessian_spectrum",
    "find_minima",
    "plot_energy_landscape",
    "plot_energy_landscape_3d",
]


class DirectionMethod(str, Enum):
    """Method for selecting the 2D slice directions."""

    GRADIENT = "gradient"  # Steepest descent direction
    GRADIENT_PCA = "gradient_pca"  # Gradient + top PCA of Hessian
    GRADIENT_RANDOM = "gradient_random"  # Gradient + random orthogonal
    TOP_EIGEN = "top_eigen"  # Top 2 Hessian eigenvectors
    PCA = "pca"  # Top 2 PCA directions of parameter covariance


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
    direction_method: DirectionMethod = DirectionMethod.GRADIENT_RANDOM
    d1: np.ndarray | None = None  # Flattened direction 1
    d2: np.ndarray | None = None  # Flattened direction 2

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
            direction_method=self.direction_method.value,
            d1=self.d1,
            d2=self.d2,
        )
        return out

    @classmethod
    def load(cls, path: str | pathlib.Path) -> EnergyLandscape:
        """Load energy landscape from ``.npz`` archive."""
        data = np.load(path)
        return cls(
            model_name=str(data["model_name"]),
            task_name=str(data["task_name"]),
            alphas=data["alphas"],
            betas=data["betas"],
            energy=data["energy"],
            d1_norm=float(data["d1_norm"]),
            param_count=int(data["param_count"]),
            direction_method=DirectionMethod(data.get("direction_method", "gradient_random")),
            d1=data.get("d1"),
            d2=data.get("d2"),
        )


@dataclass(frozen=True, slots=True)
class LandscapeSlice:
    """A single 1D or 2D slice through the energy landscape."""

    name: str
    alphas: np.ndarray
    energy: np.ndarray  # 1D for line, 2D for surface
    direction: np.ndarray | tuple[np.ndarray, np.ndarray]
    metadata: dict


def _orthonormal_directions(
    params: Sequence[torch.Tensor],
    grad: list[torch.Tensor],
    method: DirectionMethod = DirectionMethod.GRADIENT_RANDOM,
    seed: int = 0,
    hessian_evecs: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two orthonormal flat directions through the parameter vector."""

    def flat(tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in tensors])

    g = flat(grad)
    g_norm = g.norm()
    if g_norm < 1e-8:
        # Zero gradient - use random directions
        rng = np.random.default_rng(seed)
        d1 = torch.from_numpy(rng.standard_normal(g.shape)).to(dtype=g.dtype, device=g.device)
        d1 = d1 / (d1.norm() + 1e-8)
        d2 = torch.from_numpy(rng.standard_normal(g.shape)).to(dtype=g.dtype, device=g.device)
        d2 = d2 - (d2 @ d1) * d1
        d2 = d2 / (d2.norm() + 1e-8)
        return d1, d2

    d1 = g / g_norm

    match method:
        case DirectionMethod.GRADIENT_RANDOM:
            rng = np.random.default_rng(seed)
            r = rng.standard_normal(d1.shape)
            r = torch.from_numpy(r).to(dtype=g.dtype, device=g.device)
            d2 = r - (r @ d1) * d1
            d2 = d2 / (d2.norm() + 1e-8)

        case DirectionMethod.GRADIENT_PCA:
            # Use top eigenvector of Hessian if available, else random
            if hessian_evecs is not None and len(hessian_evecs) > 0:
                d2 = flat([hessian_evecs[0]])
                d2 = d2 - (d2 @ d1) * d1
                d2 = d2 / (d2.norm() + 1e-8)
            else:
                rng = np.random.default_rng(seed)
                r = rng.standard_normal(d1.shape)
                r = torch.from_numpy(r).to(dtype=g.dtype, device=g.device)
                d2 = r - (r @ d1) * d1
                d2 = d2 / (d2.norm() + 1e-8)

        case DirectionMethod.TOP_EIGEN:
            if hessian_evecs is not None and len(hessian_evecs) >= 2:
                d1 = flat([hessian_evecs[0]])
                d1 = d1 / (d1.norm() + 1e-8)
                d2 = flat([hessian_evecs[1]])
                d2 = d2 - (d2 @ d1) * d1
                d2 = d2 / (d2.norm() + 1e-8)
            else:
                # Fallback to gradient + random
                rng = np.random.default_rng(seed)
                r = rng.standard_normal(d1.shape)
                r = torch.from_numpy(r).to(dtype=g.dtype, device=g.device)
                d2 = r - (r @ d1) * d1
                d2 = d2 / (d2.norm() + 1e-8)

        case DirectionMethod.PCA:
            # Would need parameter covariance - fallback to gradient + random
            rng = np.random.default_rng(seed)
            r = rng.standard_normal(d1.shape)
            r = torch.from_numpy(r).to(dtype=g.dtype, device=g.device)
            d2 = r - (r @ d1) * d1
            d2 = d2 / (d2.norm() + 1e-8)

        case _:
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


def compute_hessian_spectrum(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    top_k: int = 10,
    use_lanczos: bool = True,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """
    Compute top-k eigenvalues and eigenvectors of the Hessian.

    Uses Lanczos algorithm for efficiency on large models.

    Returns:
        eigenvalues: Top-k eigenvalues (ascending)
        eigenvectors: Corresponding eigenvectors (flattened)
    """
    model.eval()
    params = list(p for p in model.parameters() if p.requires_grad)
    if not params:
        raise ValueError("model has no trainable parameters")

    # Flatten parameters
    p_flat = torch.cat([p.reshape(-1) for p in params])
    n_params = p_flat.numel()

    # Define Hessian-vector product using autograd
    def hvp(v: torch.Tensor) -> torch.Tensor:
        """Hessian-vector product."""
        model.zero_grad(set_to_none=True)

        # Compute gradient of loss
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])

        # Gradient of (grad · v)
        grad_v = (grad_flat * v).sum()
        hvp_grads = torch.autograd.grad(grad_v, params, retain_graph=False)
        return torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                          for g, p in zip(hvp_grads, params)])

    if use_lanczos and n_params > 100:
        # Lanczos algorithm for large models
        from scipy.sparse.linalg import LinearOperator, eigsh

        def matvec(v):
            v_tensor = torch.from_numpy(v.astype(np.float32)).to(p_flat.device)
            with torch.no_grad():
                result = hvp(v_tensor)
            return result.cpu().numpy().astype(np.float32)

        operator = LinearOperator((n_params, n_params), matvec=matvec, dtype=np.float32)
        eigvals, eigvecs = eigsh(operator, k=min(top_k, n_params - 1), which="LA")
        eigvecs_torch = [torch.from_numpy(eigvecs[:, i]).to(p_flat.device) for i in range(eigvecs.shape[1])]
        return eigvals, eigvecs_torch
    else:
        # Full Hessian for small models (expensive!)
        hessian = torch.zeros(n_params, n_params, device=p_flat.device)
        for i in range(n_params):
            v = torch.zeros(n_params, device=p_flat.device)
            v[i] = 1.0
            hessian[:, i] = hvp(v)

        eigvals, eigvecs = torch.linalg.eigh(hessian)
        eigvals = eigvals[-top_k:].cpu().numpy()
        eigvecs_torch = [eigvecs[:, -i - 1] for i in range(top_k)]
        return eigvals, eigvecs_torch


def compute_energy_landscape(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    task_name: str,
    radius: float = 1.0,
    grid: int = 21,
    seed: int = 0,
    direction_method: DirectionMethod = DirectionMethod.GRADIENT_RANDOM,
    hessian_evecs: list[torch.Tensor] | None = None,
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
        direction_method: Method for selecting slice directions.
        hessian_evecs: Pre-computed Hessian eigenvectors (for TOP_EIGEN method).

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
    d1, d2 = _orthonormal_directions(
        params, grads, method=direction_method, seed=seed, hessian_evecs=hessian_evecs
    )
    d1_norm = torch.cat([g.reshape(-1) for g in grads]).norm().item()

    p_flat = list(params)
    alphas = np.linspace(-radius, radius, grid)
    betas = np.linspace(-radius, radius, grid)
    energy = np.zeros((grid, grid))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            deltas = _flat_to_params(d1, d2, params, alpha, beta)
            energy[i, j] = _energy_at(model, xb, yb, p_flat, deltas, use_model_energy)

    # Store flattened directions for later use
    d1_np = d1.cpu().numpy()
    d2_np = d2.cpu().numpy()

    return EnergyLandscape(
        model_name=model_name,
        task_name=task_name,
        alphas=alphas,
        betas=betas,
        energy=energy,
        d1_norm=d1_norm,
        param_count=sum(p.numel() for p in params),
        direction_method=direction_method,
        d1=d1_np,
        d2=d2_np,
    )


def compute_multiple_slices(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
    task_name: str,
    methods: list[DirectionMethod] | None = None,
    radius: float = 1.0,
    grid: int = 21,
) -> list[EnergyLandscape]:
    """Compute energy landscapes for multiple direction methods."""
    if methods is None:
        methods = [
            DirectionMethod.GRADIENT_RANDOM,
            DirectionMethod.GRADIENT_PCA,
            DirectionMethod.TOP_EIGEN,
        ]

    # Pre-compute Hessian eigenvectors for TOP_EIGEN
    hessian_evecs = None
    if DirectionMethod.TOP_EIGEN in methods or DirectionMethod.GRADIENT_PCA in methods:
        try:
            _, hessian_evecs = compute_hessian_spectrum(model, x, y, top_k=5)
        except Exception:
            hessian_evecs = None

    landscapes = []
    for i, method in enumerate(methods):
        seed = 42 + i * 100  # Different seed per method
        landscape = compute_energy_landscape(
            model, x, y, model_name, task_name,
            radius=radius, grid=grid, seed=seed,
            direction_method=method, hessian_evecs=hessian_evecs,
        )
        landscapes.append(landscape)

    return landscapes


def plot_energy_landscape(
    landscape: EnergyLandscape,
    output_dir: str | pathlib.Path = "results/figures",
    cmap: str = "viridis",
    show_gradient: bool = True,
    show_contours: bool = True,
    show_minima: bool = True,
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
    fname = f"energy_landscape_{landscape.model_name}_{landscape.task_name}_{landscape.direction_method.value}.png"
    fig_path = out / fname

    fig, ax = plt.subplots(figsize=(8, 7))
    X, Y = np.meshgrid(landscape.alphas, landscape.betas)

    # Energy surface
    contour_filled = ax.contourf(X, Y, landscape.energy.T, levels=30, cmap=cmap, alpha=0.8)
    plt.colorbar(contour_filled, ax=ax, label="Energy", shrink=0.8)

    if show_contours:
        cf = ax.contour(X, Y, landscape.energy.T, levels=12, colors="k", linewidths=0.5, alpha=0.6)
        ax.clabel(cf, inline=True, fontsize=6)

    if show_gradient:
        # Gradient flow arrows: negative gradient of the energy surface.
        gx, gy = np.gradient(landscape.energy)
        skip = slice(None, None, max(1, len(landscape.alphas) // 10))

        ax.quiver(
            X[skip, skip],
            Y[skip, skip],
            -gx[skip, skip],
            -gy[skip, skip],
            color="white",
            alpha=0.8,
            scale=25,
            width=0.003,
            headwidth=3,
            headlength=4,
        )

    if show_minima:
        # Find local minima
        minima = find_minima(landscape.energy)
        if minima:
            min_alphas = [landscape.alphas[m[0]] for m in minima]
            min_betas = [landscape.betas[m[1]] for m in minima]
            ax.scatter(min_alphas, min_betas, marker="*", s=200, c="red",
                       edgecolors="white", linewidths=1, label="Local Minima", zorder=5)

    # Origin (trained weights)
    ax.scatter([0], [0], marker="*", s=180, c="cyan", edgecolors="black",
               linewidths=1, label="Trained weights w*", zorder=5)

    ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

    ax.set_xlabel(r"perturbation along $d_1$ (units of $|\nabla E|$)")
    ax.set_ylabel(r"perturbation along $d_2$")
    method_str = landscape.direction_method.value.replace("_", " ")
    ax.set_title(
        f"Energy Landscape — {landscape.model_name} ({landscape.task_name})\n"
        f"Method: {method_str} | {landscape.param_count:,} params | "
        f"$|\nabla E|$ = {landscape.d1_norm:.2e}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_energy_landscape_3d(
    landscape: EnergyLandscape,
    output_dir: str | pathlib.Path = "results/figures",
    cmap: str = "viridis",
    elevation: float = 30,
    azimuth: float = 45,
) -> pathlib.Path:
    """Render the energy landscape as a 3D surface plot using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly required for 3D plots. Install with: pip install plotly")

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"energy_landscape_3d_{landscape.model_name}_{landscape.task_name}_{landscape.direction_method.value}.html"
    fig_path = out / fname

    X, Y = np.meshgrid(landscape.alphas, landscape.betas)
    Z = landscape.energy.T

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale=cmap,
            showscale=True,
            colorbar=dict(title="Energy", thickness=20),
            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.4),
        )
    ])

    # Add trained weights point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[landscape.energy[len(landscape.alphas) // 2, len(landscape.betas) // 2]],
        mode="markers",
        marker=dict(size=8, color="cyan", symbol="diamond"),
        name="Trained weights w*",
    ))

    # Find and add minima
    minima = find_minima(landscape.energy)
    if minima:
        min_alphas = [landscape.alphas[m[0]] for m in minima]
        min_betas = [landscape.betas[m[1]] for m in minima]
        min_energies = [landscape.energy[m[0], m[1]] for m in minima]
        fig.add_trace(go.Scatter3d(
            x=min_alphas, y=min_betas, z=min_energies,
            mode="markers",
            marker=dict(size=6, color="red", symbol="x"),
            name="Local Minima",
        ))

    fig.update_layout(
        title=f"Energy Landscape 3D — {landscape.model_name} ({landscape.task_name})",
        scene=dict(
            xaxis_title=r"$\alpha$ (along $d_1$)",
            yaxis_title=r"$\beta$ (along $d_2$)",
            zaxis_title="Energy",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=900,
        height=700,
        template="plotly_white",
    )

    fig.write_html(fig_path)
    return fig_path


def find_minima(energy: np.ndarray, threshold: float = 1e-4) -> list[tuple[int, int]]:
    """Find local minima in the energy grid.

    Returns list of (i, j) indices where energy[i,j] is a local minimum.
    """
    minima = []
    rows, cols = energy.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = energy[i, j]
            # Check 8 neighbors
            neighbors = [
                energy[i - 1, j - 1], energy[i - 1, j], energy[i - 1, j + 1],
                energy[i, j - 1], energy[i, j + 1],
                energy[i + 1, j - 1], energy[i + 1, j], energy[i + 1, j + 1],
            ]
            if all(center <= n + threshold for n in neighbors):
                minima.append((i, j))

    return minima


def analyze_landscape_curvature(
    landscape: EnergyLandscape,
) -> dict[str, float]:
    """Analyze curvature properties of the energy landscape.

    Returns dict with:
        - condition_number: Ratio of max to min eigenvalue (approx)
        - flatness: Fraction of directions with low curvature
        - anisotropy: Ratio of curvature along d1 vs d2
        - min_energy: Minimum energy value
        - energy_at_origin: Energy at trained weights (0,0)
        - barrier_height: Max energy on boundary - min energy
    """
    energy = landscape.energy
    alphas = landscape.alphas
    betas = landscape.betas

    # Curvature along alpha (d1 direction) - second derivative at center
    center_i = len(alphas) // 2
    center_j = len(betas) // 2

    # Second derivative along alpha (d1)
    if center_i > 0 and center_i < len(alphas) - 1:
        da = alphas[1] - alphas[0]
        curv_alpha = (energy[center_i + 1, center_j] - 2 * energy[center_i, center_j]
                      + energy[center_i - 1, center_j]) / (da**2)
    else:
        curv_alpha = 0.0

    # Second derivative along beta (d2)
    if center_j > 0 and center_j < len(betas) - 1:
        db = betas[1] - betas[0]
        curv_beta = (energy[center_i, center_j + 1] - 2 * energy[center_i, center_j]
                     + energy[center_i, center_j - 1]) / (db**2)
    else:
        curv_beta = 0.0

    # Mixed derivative
    if center_i > 0 and center_i < len(alphas) - 1 and center_j > 0 and center_j < len(betas) - 1:
        da = alphas[1] - alphas[0]
        db = betas[1] - betas[0]
        curv_mixed = (energy[center_i + 1, center_j + 1] - energy[center_i + 1, center_j - 1]
                      - energy[center_i - 1, center_j + 1] + energy[center_i - 1, center_j - 1]
                      ) / (4 * da * db)
    else:
        curv_mixed = 0.0

    # Approximate Hessian at center
    hessian_2d = np.array([[curv_alpha, curv_mixed], [curv_mixed, curv_beta]])
    eigvals = np.linalg.eigvalsh(hessian_2d)
    eigvals = np.maximum(eigvals, 1e-12)  # Avoid zero/negative

    condition_number = float(eigvals.max() / eigvals.min())
    anisotropy = float(abs(curv_alpha) / (abs(curv_beta) + 1e-12))

    min_energy = float(energy.min())
    energy_at_origin = float(energy[center_i, center_j])

    # Barrier height: max on boundary - min
    boundary_values = np.concatenate([
        energy[0, :], energy[-1, :],
        energy[:, 0], energy[:, -1]
    ])
    barrier_height = float(boundary_values.max() - min_energy)

    # Flatness: fraction of eigenvalues below threshold
    flatness = float(np.sum(eigvals < 0.1 * eigvals.max()) / len(eigvals))

    return {
        "condition_number": condition_number,
        "flatness": flatness,
        "anisotropy": anisotropy,
        "curvature_d1": float(curv_alpha),
        "curvature_d2": float(curv_beta),
        "curvature_mixed": float(curv_mixed),
        "min_energy": min_energy,
        "energy_at_origin": energy_at_origin,
        "barrier_height": barrier_height,
        "eigenvalue_ratio": float(eigvals[1] / eigvals[0]) if len(eigvals) > 1 else 1.0,
    }
