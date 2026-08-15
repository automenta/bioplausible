"""Spiking STDP Kernel Backend.

LIF dynamics + 3-factor STDP kernels for neuromorphic acceleration.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.acceleration.contrastive_primitives import (
    contrastive_delta,
    lif_step,
    stdp_update,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)


class SNNKernelBackend:
    """Spiking STDP kernel backend.

    Implements:
    - LIF neuron dynamics
    - 3-factor STDP (pre * post * modulator)
    - Event-driven contrastive updates
    """

    name = AlgorithmFamily.SNN
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = True  # Time-stepped simulation
    memory_complexity = "O(T)"  # T time steps
    locality_level = LocalityLevel.LOCAL

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._layers: list[torch.nn.Linear] = []
        self._num_steps: int = 100
        self._tau_mem: float = 20.0
        self._tau_syn: float = 5.0
        self._threshold: float = 1.0
        self._refractory: float = 2.0
        self._dt: float = 1.0
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._spike_grad: str = "surrogate"

    def initialize(self, config: KernelConfig) -> None:
        self._config = config
        self._device = torch.device(
            "cuda"
            if config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
            else "cpu"
        )
        self._dtype = config.dtype

        extra = config.extra
        self._num_steps = extra.get("num_steps", 100)
        self._tau_mem = extra.get("tau_mem", 20.0)
        self._tau_syn = extra.get("tau_syn", 5.0)
        self._threshold = extra.get("spike_threshold", 1.0)
        self._refractory = extra.get("refractory_period", 2.0)
        self._dt = extra.get("dt", 1.0)
        self._spike_grad = extra.get("spike_grad", "surrogate")

    def set_model_ref(self, layers: list[torch.nn.Linear]) -> None:
        self._layers = layers

    def simulate(
        self,
        x: Tensor,
        y: Tensor | None = None,
        neuromodulator: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], dict[str, float]]:
        """Run SNN simulation for num_steps.

        Args:
            x: Input [B, D_in] (spike rates or continuous)
            y: Target labels [B]
            neuromodulator: 3rd factor signal [B, D_out] (for supervised STDP)

        Returns:
            (spike_trains, voltage_traces, telemetry)
            spike_trains: List of [B, N, T] per layer
            voltage_traces: List of [B, N, T] per layer
        """
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        batch_size = x.shape[0]
        num_layers = len(self._layers)

        # Initialize state variables per layer
        v_list = []  # Membrane potential [B, N]
        i_syn_list = []  # Synaptic current [B, N]
        spike_trains = []  # List of [B, N, T]
        voltage_traces = []

        # Layer 0: input encoding (Poisson or rate)
        # For simplicity, treat input as spike rate
        v = torch.zeros(
            batch_size,
            self._layers[0].in_features,
            device=self._device,
            dtype=self._dtype,
        )
        i_syn = torch.zeros_like(v)
        v_list.append(v)
        i_syn_list.append(i_syn)

        # Hidden/output layers
        for layer in self._layers:
            v = torch.zeros(
                batch_size, layer.out_features, device=self._device, dtype=self._dtype
            )
            i_syn = torch.zeros_like(v)
            v_list.append(v)
            i_syn_list.append(i_syn)

        # Storage
        for _ in range(num_layers + 1):
            spike_trains.append(
                torch.zeros(
                    batch_size,
                    v_list[-1].shape[1],
                    self._num_steps,
                    device=self._device,
                    dtype=self._dtype,
                )
            )
            voltage_traces.append(
                torch.zeros(
                    batch_size,
                    v_list[-1].shape[1],
                    self._num_steps,
                    device=self._device,
                    dtype=self._dtype,
                )
            )

        # Refractory state
        refractory_count = [torch.zeros_like(v) for v in v_list]

        # Simulation loop
        for t in range(self._num_steps):
            # Input layer: generate spikes from input rate
            input_spikes = (
                torch.bernoulli(x * self._dt) if x.max() <= 1.0 else (x > 0).float()
            )
            spike_trains[0][:, :, t] = input_spikes

            # Current injection to first layer
            i_syn_list[1] += input_spikes @ self._layers[0].weight.data.T * self._dt
            if self._layers[0].bias is not None:
                i_syn_list[1] += self._layers[0].bias.data * self._dt

            # Process each layer
            for layer_idx in range(1, num_layers + 1):
                v = v_list[layer_idx]
                i_syn = i_syn_list[layer_idx]

                # Refractory handling
                v = torch.where(
                    refractory_count[layer_idx] > 0, torch.zeros_like(v), v
                )

                # LIF step
                v_new, i_syn_new, spikes = lif_step(
                    v, i_syn, self._tau_mem, self._tau_syn, self._threshold, self._dt
                )

                # Update refractory
                refractory_count[layer_idx] = torch.where(
                    spikes > 0,
                    torch.full_like(
                        refractory_count[layer_idx], self._refractory / self._dt
                    ),
                    torch.clamp(refractory_count[layer_idx] - 1, min=0),
                )

                # Store
                spike_trains[layer_idx][:, :, t] = spikes
                voltage_traces[layer_idx][:, :, t] = v_new

                v_list[layer_idx] = v_new
                i_syn_list[layer_idx] = i_syn_new

                # Propagate to next layer
                if layer_idx < num_layers:
                    i_syn_list[layer_idx + 1] += (
                        spikes @ self._layers[layer_idx].weight.data.T * self._dt
                    )
                    if self._layers[layer_idx].bias is not None:
                        i_syn_list[layer_idx + 1] += (
                            self._layers[layer_idx].bias.data * self._dt
                        )

        telemetry = {
            "num_steps": self._num_steps,
            "mean_firing_rate": sum(s.mean().item() for s in spike_trains)
            / len(spike_trains),
            "total_spikes": sum(s.sum().item() for s in spike_trains),
        }

        return spike_trains, voltage_traces, telemetry

    def stdp_update(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        modulator: Tensor | None = None,
        layer_idx: int = 0,
    ) -> dict[str, Tensor]:
        """Compute STDP weight update for one layer.

        Args:
            pre_spikes: Pre-synaptic spikes [B, N_pre, T]
            post_spikes: Post-synaptic spikes [B, N_post, T]
            modulator: Optional 3rd factor [B, N_post] (broadcast over time)
            layer_idx: Layer index

        Returns:
            Weight delta dict
        """
        # STDP correlation
        delta = stdp_update(
            pre_spikes,
            post_spikes,
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.01,
            A_minus=0.01,
        )

        # Apply 3-factor modulation if provided
        if modulator is not None:
            # Modulator per post-synaptic neuron, broadcast to pre
            mod_expanded = modulator.mean(dim=0).unsqueeze(1)  # [N_post, 1]
            delta = delta * mod_expanded

        return {f"layers.{layer_idx}.weight": delta}

    def backward_contrastive(
        self,
        free_spikes: list[Tensor],
        nudged_spikes: list[Tensor],
        beta: float,
    ) -> dict[str, Tensor]:
        """Contrastive STDP: free vs nudged phase spike trains.

        Args:
            free_spikes: List of spike trains from free phase
            nudged_spikes: List of spike trains from nudged phase (with target)
            beta: Nudge strength

        Returns:
            Contrastive weight deltas
        """
        weight_deltas: dict[str, Tensor] = {}

        for i in range(len(self._layers)):
            pre_free = free_spikes[i]
            post_free = free_spikes[i + 1]
            pre_nudged = nudged_spikes[i]
            post_nudged = nudged_spikes[i + 1]

            # STDP on free phase
            free_delta = stdp_update(pre_free, post_free)

            # STDP on nudged phase
            nudged_delta = stdp_update(pre_nudged, post_nudged)

            # Contrastive delta
            delta = contrastive_delta(free_delta, nudged_delta, beta)
            weight_deltas[f"layers.{i}.weight"] = delta

        return weight_deltas

    def update_weights(self, gradients: dict[str, Tensor], lr: float) -> None:
        with torch.no_grad():
            for name, grad in gradients.items():
                if "weight" in name:
                    layer_idx = int(name.split(".")[1])
                    self._layers[layer_idx].weight.add_(lr * grad)

    def get_memory_stats(self) -> dict[str, float]:
        total_params = sum(
            p.numel() for layer in self._layers for p in layer.parameters()
        )
        # Spike train memory: B * N * T * 1 byte (bool) per layer
        spike_mb = 0.0
        if self._config is not None:
            extra = self._config.extra
            batch_size = extra.get("batch_size", 64)
            hidden_dim = extra.get("hidden_dim", 256)
            spike_mb = (
                batch_size
                * hidden_dim
                * self._num_steps
                * (len(self._layers) + 1)
                / 1e6
            )

        return {
            "params_mb": total_params * 4 / 1e6,
            "spike_trains_mb": spike_mb,
            "activations_mb": 0.0,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        return None


# Register backend
KernelRegistry.register(AlgorithmFamily.SNN, HardwareTarget.CPU, SNNKernelBackend)
KernelRegistry.register(AlgorithmFamily.SNN, HardwareTarget.CUDA, SNNKernelBackend)
KernelRegistry.register(AlgorithmFamily.SNN, HardwareTarget.TRITON, SNNKernelBackend)
KernelRegistry.register(
    AlgorithmFamily.SNN, HardwareTarget.NEUROMORPHIC, SNNKernelBackend
)


__all__ = ["SNNKernelBackend"]
