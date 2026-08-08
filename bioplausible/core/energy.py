import time
from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "EnergyProfile",
    "EnergyTracker",
    "count_flops",
    "profile_run",
]


@dataclass(frozen=True, slots=True)
class EnergyProfile:
    forward_flops: int  # via torch.profiler or hook counting
    backward_flops: int  # 0 for EP/FF/PEPITA/Hebbian
    param_count: int
    activation_sparsity: float  # fraction of near-zero activations
    weight_sparsity: float  # fraction of near-zero weights
    wall_time_ms: float  # elapsed per batch
    peak_memory_mb: float  # torch.cuda.max_memory_allocated
    energy_proxy: float  # (fwd + bwd flops) * (1 - activation_sparsity) / param_count
    requires_backward: bool  # from ModelSpec


def _estimate_activation_sparsity(
    model: nn.Module,
    sample_input: torch.Tensor,
    threshold: float = 1e-5,
) -> float:
    """Run a forward pass with hooks to estimate activation sparsity."""
    activations: list[torch.Tensor] = []

    def _hook(_module, _input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.detach().flatten())
        elif isinstance(output, (tuple, list)):
            for t in output:
                if isinstance(t, torch.Tensor):
                    activations.append(t.detach().flatten())

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU)):
            hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in hooks:
            h.remove()

    if not activations:
        return 0.0

    all_acts = torch.cat(activations)
    zero_frac = (all_acts.abs() < threshold).float().mean().item()
    return zero_frac


def count_flops(model, input_shape):
    batch_size = input_shape[0] if input_shape else 1
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 2 * params * batch_size


def profile_run(model, input_shape, requires_backward=True) -> EnergyProfile:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fwd_flops = count_flops(model, input_shape)
    bwd_flops = 2 * fwd_flops if requires_backward else 0

    zero_weights = sum((p.abs() < 1e-5).sum().item() for p in model.parameters())
    weight_sparsity = zero_weights / max(params, 1)

    device = next(model.parameters()).device
    sample_input = torch.zeros(*input_shape, device=device)
    activation_sparsity = _estimate_activation_sparsity(model, sample_input)

    energy_proxy = (fwd_flops + bwd_flops) * (1 - activation_sparsity) / max(params, 1)

    return EnergyProfile(
        forward_flops=fwd_flops,
        backward_flops=bwd_flops,
        param_count=params,
        activation_sparsity=activation_sparsity,
        weight_sparsity=weight_sparsity,
        wall_time_ms=0.0,
        peak_memory_mb=0.0,
        energy_proxy=energy_proxy,
        requires_backward=requires_backward,
    )


class EnergyTracker:
    """Per-step energy/power measurement with throttled heavy metrics.

    The activation-sparsity forward and the GPU weight-sparsity reduction are
    expensive relative to one train step. Inside a probe (``global_step`` is
    not ``None``) they are computed **once**, on the first measured step, and
    cached on the model for reuse on every later step. Standalone use
    (``global_step=None``) always measures, preserving the original eager
    behaviour. The probe driver passes the step counter so the whole run is
    monitored without paying the heavy cost per batch.
    """

    def __init__(
        self,
        model: nn.Module,
        requires_backward: bool = True,
        global_step: int | None = None,
    ) -> None:
        self.model = model
        self.requires_backward = requires_backward
        self.global_step = global_step
        self.start_time = 0.0
        self.wall_time_ms = 0.0
        self.profile = None

    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wall_time_ms = (time.time() - self.start_time) * 1000

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        if exc_type is not None:
            return False

        params = sum(p.numel() for p in self.model.parameters())

        # Heavy metrics are throttled to the first step of a probe and cached on
        # the model; standalone trackers (global_step=None) always measure.
        heavy_cached = hasattr(self.model, "_biopl_activation_sparsity")
        compute_heavy = self.global_step is None or not heavy_cached

        if compute_heavy:
            zero_weights = sum(
                (p.abs() < 1e-5).sum().item() for p in self.model.parameters()
            )
            weight_sparsity = zero_weights / max(params, 1)

            device = next(self.model.parameters()).device
            # Estimate activation sparsity — try to infer input dim from first
            # Linear layer.
            inp_dim = 64
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    inp_dim = m.in_features
                    break
            dummy_input = torch.zeros(1, inp_dim, device=device)
            activation_sparsity = _estimate_activation_sparsity(self.model, dummy_input)

            if self.global_step is not None:
                setattr(self.model, "_biopl_activation_sparsity", activation_sparsity)
                setattr(self.model, "_biopl_weight_sparsity", weight_sparsity)
        else:
            activation_sparsity = float(
                getattr(self.model, "_biopl_activation_sparsity")
            )
            weight_sparsity = float(getattr(self.model, "_biopl_weight_sparsity"))

        batch_size = 64
        fwd_flops = 2 * params * batch_size
        bwd_flops = 2 * fwd_flops if self.requires_backward else 0

        energy_proxy = (
            (fwd_flops + bwd_flops) * (1 - activation_sparsity) / max(params, 1)
        )

        self.profile = EnergyProfile(
            forward_flops=fwd_flops,
            backward_flops=bwd_flops,
            param_count=params,
            activation_sparsity=activation_sparsity,
            weight_sparsity=weight_sparsity,
            wall_time_ms=self.wall_time_ms,
            peak_memory_mb=peak_mem,
            energy_proxy=energy_proxy,
            requires_backward=self.requires_backward,
        )
        return False
