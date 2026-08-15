"""Shared Contrastive Primitives for Kernel Backends.

Common Triton/CuPy operations used across all bio-plausible algorithm kernels:
- Batched outer products
- Contrastive Hebbian deltas
- Spectral normalization
- LIF dynamics
- Phase encoding
- Conductance matmul
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def batched_outer_product(src: Tensor, dst: Tensor) -> Tensor:
    """Batched outer product: (B, D_in) x (B, D_out) -> (D_out, D_in).

    Computes sum over batch: dst.T @ src / B
    """
    return (dst.T @ src) / src.shape[0]


def contrastive_delta(free: Tensor, nudged: Tensor, beta: float) -> Tensor:
    """Contrastive Hebbian update: (nudged - free) / beta.

    Args:
        free: Activations from free phase
        nudged: Activations from nudged/clamped phase
        beta: Nudge strength (non-zero)

    Returns:
        Weight delta for contrastive update
    """
    return (nudged - free) / beta


def spectral_norm_power_iteration(
    W: Tensor,
    u: Tensor | None = None,
    num_iters: int = 1,
) -> tuple[Tensor, Tensor, float]:
    """Power iteration spectral normalization.

    Args:
        W: Weight matrix [out_dim, in_dim]
        u: Previous u vector for warm start [out_dim]
        num_iters: Number of power iterations

    Returns:
        (W_normalized, u_new, sigma)
    """
    out_dim = W.shape[0]
    if u is None:
        u = torch.randn(out_dim, device=W.device, dtype=W.dtype)
        u = u / u.norm()

    for _ in range(num_iters):
        v = W.T @ u
        v = v / v.norm()
        u = W @ v
        u = u / u.norm()

    sigma = (u @ W @ v).item()
    W_normalized = W / sigma
    return W_normalized, u, sigma


def lif_step(
    v: Tensor,
    i_syn: Tensor,
    tau_mem: float,
    tau_syn: float,
    threshold: float,
    dt: float = 1.0,
    refractory: float = 0.0,
    spike_history: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """LIF neuron dynamics step.

    dv/dt = -v/tau_mem + I_syn
    dI/dt = -I/tau_syn + spikes
    Spike when v > threshold

    Args:
        v: Membrane potential [B, N]
        i_syn: Synaptic current [B, N]
        tau_mem: Membrane time constant
        tau_syn: Synaptic time constant
        threshold: Spike threshold
        dt: Time step
        refractory: Refractory period (not implemented in this simple version)
        spike_history: Optional spike history for refractory

    Returns:
        (v_new, i_syn_new, spikes)
    """
    # Membrane potential update
    v_new = v + dt * (-v / tau_mem + i_syn)

    # Synaptic current update (decay)
    i_syn_new = i_syn * (1 - dt / tau_syn)

    # Spike generation
    spikes = (v_new > threshold).float()

    # Reset membrane potential after spike
    v_new = torch.where(spikes.bool(), torch.zeros_like(v_new), v_new)

    # Add spikes to synaptic current
    i_syn_new = i_syn_new + spikes

    return v_new, i_syn_new, spikes


def phase_encode(
    x: Tensor,
    wavelength: float = 1550e-9,
    phase_range: float = 2 * 3.14159,
) -> Tensor:
    """Optical phase encoding.

    Maps input to phase domain for interferometric computation.

    Args:
        x: Input tensor [B, D] in range [0, 1] or [-1, 1]
        wavelength: Optical wavelength in meters
        phase_range: Phase modulation range in radians

    Returns:
        Phase-encoded tensor [B, D] in radians
    """
    # Normalize to [0, 1] then scale to phase range
    x_norm = (x + 1) / 2 if x.min() < 0 else x
    return x_norm * phase_range


def conductance_matmul(
    G: Tensor,
    V: Tensor,
    adc_bits: int = 8,
    dac_bits: int = 6,
    ir_drop_factor: float = 0.1,
) -> Tensor:
    """Analog crossbar matmul via Ohm's law: I = G @ V.

    Args:
        G: Conductance matrix [out, in] (Siemens)
        V: Input voltages [B, in]
        adc_bits: ADC resolution
        dac_bits: DAC resolution
        ir_drop_factor: IR drop factor (0-1)

    Returns:
        Output currents [B, out]
    """
    # Quantize conductances to ADC resolution
    G_max = G.max()
    G_quantized = torch.round(G / G_max * (2**adc_bits - 1)) / (2**adc_bits - 1) * G_max

    # Quantize voltages to DAC resolution
    V_max = V.abs().max()
    V_quantized = torch.round(V / V_max * (2**dac_bits - 1)) / (2**dac_bits - 1) * V_max

    # Matmul
    current = V_quantized @ G_quantized.T

    # IR drop simulation
    if ir_drop_factor > 0:
        current = current * (1 - ir_drop_factor)

    return current


def contrastive_hebbian_update(
    src_free: Tensor,
    dst_free: Tensor,
    src_nudged: Tensor,
    dst_nudged: Tensor,
    lr: float,
    beta: float,
) -> Tensor:
    """Full contrastive Hebbian weight update.

    Delta W = lr * (dst_nudged.T @ src_nudged - dst_free.T @ src_free) / (beta * B)

    Args:
        src_free: Pre-synaptic free activations [B, D_in]
        dst_free: Post-synaptic free activations [B, D_out]
        src_nudged: Pre-synaptic nudged activations [B, D_in]
        dst_nudged: Post-synaptic nudged activations [B, D_out]
        lr: Learning rate
        beta: Nudge strength

    Returns:
        Weight delta [D_out, D_in]
    """
    free_update = batched_outer_product(src_free, dst_free)
    nudged_update = batched_outer_product(src_nudged, dst_nudged)
    return lr * contrastive_delta(free_update, nudged_update, beta)


def forward_forward_goodness(
    pos_acts: Tensor,
    neg_acts: Tensor,
    threshold: float = 1.0,
) -> Tensor:
    """Forward-Forward goodness contrast.

    Goodness = ||pos||^2 - ||neg||^2 - threshold

    Args:
        pos_acts: Positive pass activations [B, D]
        neg_acts: Negative pass activations [B, D]
        threshold: Goodness threshold

    Returns:
        Goodness scalar per sample [B]
    """
    pos_norm = pos_acts.pow(2).sum(dim=1)
    neg_norm = neg_acts.pow(2).sum(dim=1)
    return pos_norm - neg_norm - threshold


def pepita_error_modulation(
    error: Tensor,
    feedback_matrix: Tensor,
    scale: float = 1.0,
) -> Tensor:
    """PEPITA error-modulated update.

    Delta W = scale * error @ feedback_matrix.T

    Args:
        error: Error signal [B, D_out]
        feedback_matrix: Fixed feedback weights [D_in, D_out]
        scale: Modulation scale

    Returns:
        Weight delta [D_out, D_in]
    """
    return scale * (error.T @ feedback_matrix)


def target_propagation_target(
    target: Tensor,
    inverse_weight: Tensor,
    target_lr: float = 0.1,
) -> Tensor:
    """Target propagation: compute layer target from output target.

    h_target = target - target_lr * inverse_weight @ (output - target)

    Args:
        target: Output target [B, D_out]
        inverse_weight: Inverse network weight [D_out, D_in]
        target_lr: Target learning rate

    Returns:
        Layer target [B, D_in]
    """
    return target - target_lr * (target @ inverse_weight.T)


def predictive_coding_inference_step(
    mu: list[Tensor],
    x: Tensor,
    W: list[Tensor],
    b: list[Tensor],
    eta_infer: float,
    activation: Literal["relu", "tanh", "linear"] = "tanh",
) -> list[Tensor]:
    """Predictive Coding inference step (PCN).

    Each state chases its prediction from the layer below: the input state is
    clamped to ``x`` and never updated; every other state is pulled toward its
    parent's prediction, ``mu_l = mu_l - eta * (mu_l - f(mu_{l-1} W_{l-1} + b))``.

    Args:
        mu: List of state estimates per layer (``mu[0]`` is the clamped input).
        x: Input (must match ``mu[0]``).
        W: List of forward weight matrices ``[D_l, D_{l-1}]``.
        b: List of biases per layer.
        eta_infer: Inference learning rate.
        activation: Activation function.

    Returns:
        Updated ``mu`` list (``mu[0]`` unchanged).
    """
    L = len(mu)

    def act(z: Tensor) -> Tensor:
        if activation == "relu":
            return torch.relu(z)
        if activation == "tanh":
            return torch.tanh(z)
        return z

    def act_deriv(z: Tensor) -> Tensor:
        if activation == "relu":
            return (z > 0).float()
        if activation == "tanh":
            return 1 - torch.tanh(z) ** 2
        return torch.ones_like(z)

    mu_new = [m.clone() for m in mu]
    mu_new[0] = x.clone()

    for layer_idx in range(1, L):
        pred = act(
            mu[layer_idx - 1] @ W[layer_idx - 1].T + b[layer_idx - 1]
        )
        error = mu[layer_idx] - pred
        mu_new[layer_idx] = mu[layer_idx] - eta_infer * error * act_deriv(mu[layer_idx])

    return mu_new


def stdp_update(
    pre_spikes: Tensor,
    post_spikes: Tensor,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    A_plus: float = 0.01,
    A_minus: float = 0.01,
) -> Tensor:
    """STDP weight update from spike timing.

    Args:
        pre_spikes: Pre-synaptic spikes [B, N_pre, T]
        post_spikes: Post-synaptic spikes [B, N_post, T]
        tau_plus: LTP time constant
        tau_minus: LTD time constant
        A_plus: LTP amplitude
        A_minus: LTD amplitude

    Returns:
        Weight delta [N_post, N_pre]
    """
    # Correlation-based approximation over (batch, time):
    # LTP: post at t+1 with pre at t -> [N_post, N_pre]
    ltp = torch.einsum("bit,bjt->ij", post_spikes[:, :, 1:], pre_spikes[:, :, :-1])
    # LTD: post at t with pre at t+1 -> [N_post, N_pre]
    ltd = torch.einsum("bit,bjt->ij", post_spikes[:, :, :-1], pre_spikes[:, :, 1:])

    return A_plus * ltp - A_minus * ltd


__all__ = [
    "batched_outer_product",
    "contrastive_delta",
    "spectral_norm_power_iteration",
    "lif_step",
    "phase_encode",
    "conductance_matmul",
    "contrastive_hebbian_update",
    "forward_forward_goodness",
    "pepita_error_modulation",
    "target_propagation_target",
    "predictive_coding_inference_step",
    "stdp_update",
]
