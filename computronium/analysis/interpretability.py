"""Interpretability Toolkit — Receptive Fields, Weight Spectra, Info Flow.

Provides tools for analyzing learned representations:
- Receptive field visualization
- Weight matrix spectra (singular values, condition numbers)
- Information flow analysis (mutual information, causal mediation)
- Concept alignment metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class WeightSpectra:
    """Spectral analysis of weight matrices."""

    layer_name: str
    singular_values: np.ndarray
    condition_number: float
    effective_rank: float
    spectral_norm: float
    frobenius_norm: float
    stable_rank: float  # Fro^2 / Spectral^2

    def to_dict(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "singular_values": self.singular_values.tolist(),
            "condition_number": self.condition_number,
            "effective_rank": self.effective_rank,
            "spectral_norm": self.spectral_norm,
            "frobenius_norm": self.frobenius_norm,
            "stable_rank": self.stable_rank,
        }


@dataclass(frozen=True, slots=True)
class ReceptiveField:
    """Receptive field analysis for a neuron/unit."""

    layer_name: str
    unit_index: int
    rf_map: np.ndarray  # 2D receptive field
    rf_size: tuple[int, int]  # (height, width)
    center: tuple[float, float]
    spread: float  # Spatial spread (std)

    def to_dict(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "unit_index": self.unit_index,
            "rf_map": self.rf_map.tolist(),
            "rf_size": self.rf_size,
            "center": self.center,
            "spread": self.spread,
        }


@dataclass(frozen=True, slots=True)
class InformationFlow:
    """Information flow analysis between layers."""

    source_layer: str
    target_layer: str
    mutual_information: float
    transfer_entropy: float | None = None
    causal_effect: float | None = None

    def to_dict(self) -> dict:
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "mutual_information": self.mutual_information,
            "transfer_entropy": self.transfer_entropy,
            "causal_effect": self.causal_effect,
        }


@dataclass(frozen=True, slots=True)
class ConceptAlignment:
    """Concept alignment metrics for learned representations."""

    layer_name: str
    concept_scores: dict[str, float]  # concept_name -> alignment score
    top_concepts: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "concept_scores": self.concept_scores,
            "top_concepts": self.top_concepts,
        }


# =============================================================================
# Weight Spectra Analysis
# =============================================================================


def analyze_weight_spectra(
    model: nn.Module,
    layers: list[str] | None = None,
    max_singular_values: int = 100,
) -> list[WeightSpectra]:
    """Analyze weight matrix spectra for specified layers.

    Args:
        model: PyTorch model
        layers: List of layer names to analyze (None = all linear/conv)
        max_singular_values: Maximum singular values to compute

    Returns:
        List of WeightSpectra objects
    """
    results = []

    if layers is None:
        # Auto-detect linear and conv layers
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                layers.append(name)

    for name in layers:
        module = dict(model.named_modules()).get(name)
        if module is None:
            logger.warning("Layer %s not found", name)
            continue

        weight = None
        if isinstance(module, nn.Linear):
            weight = module.weight.data
        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            # Reshape conv weights: (out_channels, in_channels, *kernel)
            w = module.weight.data
            weight = w.reshape(w.shape[0], -1)

        if weight is None:
            continue

        # Compute SVD
        try:
            _, S_vals, _ = torch.linalg.svd(weight.float(), full_matrices=False)
            S_vals = S_vals.cpu().numpy()
        except RuntimeError:
            # Fallback for large matrices
            S_vals = torch.linalg.svdvals(weight.float()).cpu().numpy()

        S_vals = S_vals[:max_singular_values]

        spectral_norm = float(S_vals[0]) if len(S_vals) > 0 else 0.0
        frobenius_norm = float(torch.norm(weight, p="fro").item())
        condition_number = (
            float(S_vals[0] / S_vals[-1])
            if len(S_vals) > 1 and S_vals[-1] > 0
            else float("inf")
        )
        effective_rank = (
            float(np.sum(S_vals) ** 2 / np.sum(S_vals**2))
            if np.sum(S_vals**2) > 0
            else 0.0
        )
        stable_rank = (
            (frobenius_norm**2) / (spectral_norm**2) if spectral_norm > 0 else 0.0
        )

        results.append(
            WeightSpectra(
                layer_name=name,
                singular_values=S_vals,
                condition_number=condition_number,
                effective_rank=effective_rank,
                spectral_norm=spectral_norm,
                frobenius_norm=frobenius_norm,
                stable_rank=stable_rank,
            )
        )

    return results


def plot_weight_spectra(
    spectra: list[WeightSpectra],
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot weight spectra (singular value decay)."""
    import plotly.graph_objects as go

    fig = go.Figure()

    for spec in spectra:
        # Normalize singular values
        norm_s = (
            spec.singular_values / spec.singular_values[0]
            if spec.singular_values[0] > 0
            else spec.singular_values
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(norm_s))),
                y=norm_s,
                mode="lines+markers",
                name=spec.layer_name,
                hovertemplate=(
                    f"{spec.layer_name}<br>"
                    "Rank=%{x}<br>"
                    "Normalized SV=%{y:.4f}<br>"
                    f"Cond={spec.condition_number:.2f}<br>"
                    f"EffRank={spec.effective_rank:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Weight Matrix Singular Value Spectra",
        xaxis_title="Singular Value Index",
        yaxis_title="Normalized Singular Value",
        yaxis_type="log",
        template="plotly_white",
    )

    if output_path:
        fig.write_html(output_path)
        logger.info("Saved weight spectra plot to %s", output_path)

    return fig


# =============================================================================
# Receptive Field Analysis
# =============================================================================


def compute_receptive_field(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
    model: nn.Module,
    layer_name: str,
    unit_index: int,
    input_shape: tuple[int, ...],
    device: str | torch.device = "cpu",
    method: Literal["gradient", "activation"] = "gradient",
    n_samples: int = 100,
) -> ReceptiveField:
    """Compute receptive field for a specific unit.

    Args:
        model: PyTorch model
        layer_name: Target layer name
        unit_index: Index of unit/neuron to analyze
        input_shape: Input shape (C, H, W) or (seq_len,)
        device: Device for computation
        method: "gradient" (input gradient) or "activation" (max activating patches)
        n_samples: Number of samples for activation method

    Returns:
        ReceptiveField object
    """
    model.eval()
    model.to(device)

    # Hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations[layer_name] = output.detach()

    target_module = dict(model.named_modules()).get(layer_name)
    if target_module is None:
        raise ValueError(f"Layer {layer_name} not found")

    handle = target_module.register_forward_hook(hook_fn)

    try:
        if method == "gradient":
            # Gradient-based receptive field
            input_tensor = torch.randn(
                1, *input_shape, device=device, requires_grad=True
            )
            _ = model(input_tensor)
            act = activations[layer_name]

            # Get gradient w.r.t. input for target unit
            if act.dim() == 4:  # Conv: (B, C, H, W)
                target_act = act[0, unit_index].sum()
            elif act.dim() == 3:  # Linear/Sequence: (B, L, D)
                target_act = act[0, :, unit_index].sum()
            else:  # (B, D)
                target_act = act[0, unit_index]

            target_act.backward()
            grad = input_tensor.grad[0].abs().cpu().numpy()

            # For conv, grad is (C, H, W) - take max over channels
            if grad.ndim == 3:  # ruff: ignore[if-else-block-instead-of-if-exp]
                rf_map = grad.max(axis=0)
            else:
                rf_map = grad

        else:  # activation method
            rf_map = np.zeros(input_shape[1:] if len(input_shape) > 1 else input_shape)
            for _ in range(n_samples):
                input_tensor = torch.randn(1, *input_shape, device=device)
                _ = model(input_tensor)
                act = activations[layer_name]

                if act.dim() == 4:
                    unit_act = act[0, unit_index].cpu().numpy()  # (H, W)
                elif act.dim() == 3:
                    unit_act = act[0, :, unit_index].cpu().numpy()  # (L,)
                else:
                    unit_act = act[0, unit_index].cpu().numpy()

                # Accumulate weighted by activation
                if unit_act.ndim == 2:
                    rf_map += unit_act * input_tensor[0].abs().cpu().numpy().max(axis=0)
                elif unit_act.ndim == 1:
                    rf_map += unit_act[:, None] * input_tensor[0].abs().cpu().numpy()

            rf_map /= n_samples

    finally:
        handle.remove()

    # Compute statistics
    if rf_map.ndim >= 2:
        h, w = rf_map.shape[-2], rf_map.shape[-1]
    else:
        h, w = 1, rf_map.shape[-1]
    rf_size = (h, w)

    # Center of mass
    if rf_map.ndim == 2:
        y_coords = np.arange(h, dtype=float)[:, None]
        x_coords = np.arange(w, dtype=float)[None, :]
        y_coords = np.broadcast_to(y_coords, (h, w))
        x_coords = np.broadcast_to(x_coords, (h, w))
        total = rf_map.sum()
        if total > 0:
            center_y = float((y_coords * rf_map).sum() / total)
            center_x = float((x_coords * rf_map).sum() / total)
            center: tuple[float, float] = (center_y, center_x)
            # Spread
            spread = float(
                np.sqrt(
                    ((y_coords - center_y) ** 2 * rf_map).sum() / total
                    + ((x_coords - center_x) ** 2 * rf_map).sum() / total
                )
            )
        else:
            center = (float(h) / 2, float(w) / 2)
            spread = 0.0
    else:
        center = (0.0, 0.0)
        spread = 0.0

    return ReceptiveField(
        layer_name=layer_name,
        unit_index=unit_index,
        rf_map=rf_map,
        rf_size=rf_size,
        center=center,
        spread=spread,
    )


def plot_receptive_fields(
    rfs: list[ReceptiveField],
    n_cols: int = 4,
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot multiple receptive fields as subplots."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(rfs)
    n_rows = (n + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{rf.layer_name}[{rf.unit_index}]" for rf in rfs],
    )

    for i, rf in enumerate(rfs):
        row = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_trace(
            go.Heatmap(
                z=rf.rf_map,
                colorscale="RdBu",
                zmid=0,
                showscale=(i == 0),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Receptive Fields",
        template="plotly_white",
        height=300 * n_rows,
    )

    if output_path:
        fig.write_html(output_path)
        logger.info("Saved receptive fields plot to %s", output_path)

    return fig


# =============================================================================
# Information Flow Analysis
# =============================================================================


def compute_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 20,
) -> float:
    """Compute mutual information between two variables using histogram."""
    # Flatten if needed
    x = x.flatten()
    y = y.flatten()

    # 2D histogram
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    hist_x, _ = np.histogram(x, bins=bins, density=True)
    hist_y, _ = np.histogram(y, bins=bins, density=True)

    # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            p_xy = hist_2d[i, j]
            p_x = hist_x[i]
            p_y = hist_y[j]
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))

    return mi


def analyze_information_flow(  # ruff: ignore[complex-structure, too-many-locals]
    model: nn.Module,
    dataloader: DataLoader,
    layers: list[str],
    device: str | torch.device = "cpu",
    max_samples: int = 1000,
) -> list[InformationFlow]:
    """Analyze information flow between consecutive layers.

    Args:
        model: PyTorch model
        dataloader: Data loader for inputs
        layers: Ordered list of layer names
        device: Device for computation
        max_samples: Maximum samples to process

    Returns:
        List of InformationFlow objects
    """
    model.eval()
    model.to(device)

    # Hook to capture activations
    activations = {name: [] for name in layers}

    def make_hook(name):
        def hook(module, input, output):
            if len(activations[name]) * output.shape[0] < max_samples:
                activations[name].append(output.detach().cpu())

        return hook

    handles = []
    for name in layers:
        module = dict(model.named_modules()).get(name)
        if module:
            handles.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):  # ruff: ignore[if-else-block-instead-of-if-exp]
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            _ = model(x)
            sample_count += x.shape[0]
            if sample_count >= max_samples:
                break

    for h in handles:
        h.remove()

    # Concatenate activations
    layer_acts = {}
    for name in layers:
        if activations[name]:
            layer_acts[name] = torch.cat(activations[name], dim=0).numpy()
        else:
            layer_acts[name] = np.array([])

    # Compute MI between consecutive layers
    flows = []
    for i in range(len(layers) - 1):
        src = layers[i]
        tgt = layers[i + 1]

        if len(layer_acts[src]) == 0 or len(layer_acts[tgt]) == 0:
            continue

        # Flatten spatial dimensions
        src_flat = layer_acts[src].reshape(layer_acts[src].shape[0], -1)
        tgt_flat = layer_acts[tgt].reshape(layer_acts[tgt].shape[0], -1)

        # Sample random dimensions for MI computation (too many dims is expensive)
        n_dims = min(50, src_flat.shape[1], tgt_flat.shape[1])
        src_idx = np.random.choice(src_flat.shape[1], n_dims, replace=False)
        tgt_idx = np.random.choice(tgt_flat.shape[1], n_dims, replace=False)

        mi = compute_mutual_information(
            src_flat[:, src_idx].mean(axis=1),
            tgt_flat[:, tgt_idx].mean(axis=1),
        )

        flows.append(
            InformationFlow(
                source_layer=src,
                target_layer=tgt,
                mutual_information=mi,
            )
        )

    return flows


# =============================================================================
# Concept Alignment
# =============================================================================


def compute_concept_alignment(
    model: nn.Module,
    dataloader: DataLoader,
    concept_datasets: dict[str, DataLoader],
    layer_name: str,
    device: str | torch.device = "cpu",
    n_samples: int = 500,
) -> ConceptAlignment:
    """Compute alignment between layer activations and human-defined concepts.

    Args:
        model: PyTorch model
        dataloader: Main task dataloader
        concept_datasets: Dict of concept_name -> dataloader with concept labels
        layer_name: Layer to analyze
        device: Device
        n_samples: Samples per concept

    Returns:
        ConceptAlignment object
    """
    model.eval()
    model.to(device)

    # Get main task activations
    activations = []

    def hook(module, input, output):
        activations.append(output.detach().cpu())

    target_module = dict(model.named_modules()).get(layer_name)
    if target_module is None:
        raise ValueError(f"Layer {layer_name} not found")

    handle = target_module.register_forward_hook(hook)

    # Main task activations
    main_acts = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            _ = model(x)
            main_acts.append(activations[-1])
            activations.clear()
            if len(main_acts) * x.shape[0] >= n_samples:
                break

    handle.remove()

    if not main_acts:
        return ConceptAlignment(
            layer_name=layer_name, concept_scores={}, top_concepts=[]
        )

    main_acts = torch.cat(main_acts, dim=0).numpy()
    main_acts = main_acts.reshape(main_acts.shape[0], -1)

    # For each concept, compute alignment
    concept_scores = {}
    for concept_name, concept_loader in concept_datasets.items():
        concept_acts = []
        activations = []
        handle = target_module.register_forward_hook(
            lambda m, i, o: activations.append(o.detach().cpu())  # ruff: ignore[unused-lambda-argument]
        )

        with torch.no_grad():
            for batch in concept_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                _ = model(x)
                concept_acts.append(activations[-1])
                activations.clear()
                if len(concept_acts) * x.shape[0] >= n_samples:
                    break

        handle.remove()

        if not concept_acts:
            continue

        concept_acts = torch.cat(concept_acts, dim=0).numpy()
        concept_acts = concept_acts.reshape(concept_acts.shape[0], -1)

        # Compute alignment: correlation between mean activations
        main_mean = main_acts.mean(axis=0)
        concept_mean = concept_acts.mean(axis=0)

        # Cosine similarity
        alignment = np.dot(main_mean, concept_mean) / (
            np.linalg.norm(main_mean) * np.linalg.norm(concept_mean) + 1e-8
        )
        concept_scores[concept_name] = float(alignment)

    # Top concepts
    top_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    return ConceptAlignment(
        layer_name=layer_name,
        concept_scores=concept_scores,
        top_concepts=top_concepts,
    )


# =============================================================================
# Causal Mediation Analysis
# =============================================================================


def causal_mediation_analysis(  # ruff: ignore[too-many-locals]
    model: nn.Module,
    dataloader: DataLoader,
    mediator_layer: str,
    treatment_fn: Callable[[torch.Tensor], torch.Tensor],
    outcome_fn: Callable[[torch.Tensor], torch.Tensor],
    device: str | torch.device = "cpu",
    n_samples: int = 500,
) -> dict:
    """Perform causal mediation analysis.

    Decomposes total effect of treatment on outcome into:
    - Direct effect (not through mediator)
    - Indirect effect (through mediator)

    Args:
        model: PyTorch model
        dataloader: Data loader
        mediator_layer: Layer to treat as mediator
        treatment_fn: Function applying treatment to input
        outcome_fn: Function computing outcome from model output
        device: Device
        n_samples: Number of samples

    Returns:
        Dict with total_effect, direct_effect, indirect_effect, proportion_mediated
    """
    model.eval()
    model.to(device)

    # Capture mediator activations
    mediator_acts = []

    def hook(module, input, output):
        mediator_acts.append(output.detach().cpu())

    target_module = dict(model.named_modules()).get(mediator_layer)
    if target_module is None:
        raise ValueError(f"Layer {mediator_layer} not found")

    handle = target_module.register_forward_hook(hook)

    outcomes_control = []
    outcomes_treated = []
    mediators_control = []
    mediators_treated = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            # Control
            _ = model(x)
            y_control = outcome_fn(model(x))
            outcomes_control.append(y_control.cpu())
            mediators_control.append(mediator_acts[-1])
            mediator_acts.clear()

            # Treated
            x_treated = treatment_fn(x)
            _ = model(x_treated)
            y_treated = outcome_fn(model(x_treated))
            outcomes_treated.append(y_treated.cpu())
            mediators_treated.append(mediator_acts[-1])
            mediator_acts.clear()

            if len(outcomes_control) * x.shape[0] >= n_samples:
                break

    handle.remove()

    if not outcomes_control:
        return {}

    outcomes_control = torch.cat(outcomes_control).numpy()
    outcomes_treated = torch.cat(outcomes_treated).numpy()
    mediators_control = torch.cat(mediators_control).numpy()
    mediators_treated = torch.cat(mediators_treated).numpy()

    # Flatten
    mediators_control = mediators_control.reshape(mediators_control.shape[0], -1)
    mediators_treated = mediators_treated.reshape(mediators_treated.shape[0], -1)

    # Total effect
    total_effect = outcomes_treated.mean() - outcomes_control.mean()

    # Direct effect: outcome difference when mediator is held at control value
    # Approximation: train linear probe from mediator to outcome on control,
    # then predict counterfactual outcome for treated mediator
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(mediators_control, outcomes_control)

    # Predict outcome for treated mediator
    y_pred_treated_mediator = reg.predict(mediators_treated)
    direct_effect = outcomes_treated.mean() - y_pred_treated_mediator.mean()
    indirect_effect = total_effect - direct_effect

    proportion_mediated = (
        indirect_effect / total_effect if abs(total_effect) > 1e-8 else 0.0
    )

    return {
        "total_effect": float(total_effect),
        "direct_effect": float(direct_effect),
        "indirect_effect": float(indirect_effect),
        "proportion_mediated": float(proportion_mediated),
    }


# =============================================================================
# High-Level Pipeline
# =============================================================================


@dataclass(frozen=True, slots=True)
class InterpretabilityConfig:
    """Configuration for interpretability analysis."""

    layers: list[str] | None = None
    compute_spectra: bool = True
    compute_receptive_fields: bool = False
    rf_layers: list[str] | None = None
    rf_units_per_layer: int = 4
    compute_info_flow: bool = False
    compute_concept_alignment: bool = False
    concept_datasets: dict | None = None
    compute_causal_mediation: bool = False
    treatment_fn: Callable | None = None
    outcome_fn: Callable | None = None
    mediator_layer: str | None = None
    max_samples: int = 1000
    device: str = "cpu"


def run_interpretability_analysis(  # ruff: ignore[complex-structure, too-many-branches]
    model: nn.Module,
    dataloader: DataLoader,
    config: InterpretabilityConfig,
    output_dir: str | Path,
) -> dict:
    """Run full interpretability analysis.

    Args:
        model: PyTorch model
        dataloader: Data loader
        config: Analysis configuration
        output_dir: Output directory

    Returns:
        Dictionary with all analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Weight spectra
    if config.compute_spectra:
        logger.info("Computing weight spectra...")
        spectra = analyze_weight_spectra(model, config.layers)
        results["weight_spectra"] = [s.to_dict() for s in spectra]
        plot_weight_spectra(spectra, output_dir / "weight_spectra.html")

    # Receptive fields
    if config.compute_receptive_fields and config.rf_layers:
        logger.info("Computing receptive fields...")
        # Get input shape from first batch
        x = (
            next(iter(dataloader))[0]
            if hasattr(dataloader, "__iter__")
            else next(iter(dataloader))
        )
        input_shape = x.shape[1:]

        rfs = []
        for layer in config.rf_layers:
            # Sample random units
            module = dict(model.named_modules()).get(layer)
            if module is None:
                continue
            if hasattr(module, "out_channels"):
                n_units = int(module.out_channels)
            elif hasattr(module, "out_features"):
                n_units = int(module.out_features)
            else:
                continue

            if n_units <= 0:
                continue
            unit_indices = np.random.choice(
                n_units, min(config.rf_units_per_layer, n_units), replace=False
            )
            for idx in unit_indices:
                try:
                    rf = compute_receptive_field(
                        model, layer, int(idx), input_shape, config.device
                    )
                    rfs.append(rf)
                except Exception as e:
                    logger.warning("Failed to compute RF for %s[%d]: %s", layer, idx, e)

        results["receptive_fields"] = [rf.to_dict() for rf in rfs]
        if rfs:
            plot_receptive_fields(rfs, output_path=output_dir / "receptive_fields.html")

    # Information flow
    if config.compute_info_flow and config.layers:
        logger.info("Analyzing information flow...")
        flows = analyze_information_flow(
            model, dataloader, config.layers, config.device, config.max_samples
        )
        results["information_flow"] = [f.to_dict() for f in flows]

    # Concept alignment
    if config.compute_concept_alignment and config.concept_datasets and config.layers:
        logger.info("Computing concept alignment...")
        for layer in config.layers:
            alignment = compute_concept_alignment(
                model,
                dataloader,
                config.concept_datasets,
                layer,
                config.device,
                config.max_samples,
            )
            results.setdefault("concept_alignment", {})[layer] = alignment.to_dict()

    # Causal mediation
    if (
        config.compute_causal_mediation
        and config.treatment_fn
        and config.outcome_fn
        and config.mediator_layer
    ):
        logger.info("Running causal mediation analysis...")
        mediation = causal_mediation_analysis(
            model,
            dataloader,
            config.mediator_layer,
            config.treatment_fn,
            config.outcome_fn,
            config.device,
            config.max_samples,
        )
        results["causal_mediation"] = mediation

    # Save all results
    import json

    with (output_dir / "interpretability_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Interpretability analysis complete. Results saved to %s", output_dir)
    return results


__all__ = [
    "ConceptAlignment",
    "InformationFlow",
    "InterpretabilityConfig",
    "ReceptiveField",
    "WeightSpectra",
    "analyze_information_flow",
    "analyze_weight_spectra",
    "causal_mediation_analysis",
    "compute_concept_alignment",
    "compute_mutual_information",
    "compute_receptive_field",
    "plot_receptive_fields",
    "plot_weight_spectra",
    "run_interpretability_analysis",
]
