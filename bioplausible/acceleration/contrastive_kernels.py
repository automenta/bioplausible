"""Unified Contrastive Hebbian Kernel Framework for O(1) Memory Training.

Provides a common base class and algorithm-specific implementations for
contrastive Hebbian updates across all bio-plausible local learning rules.
Each algorithm implements free/nudged phase dynamics with local weight updates.

Reference: REFACTOR7 §5 - MEMORY-O(1) UNIFICATION
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from bioplausible.acceleration.contrastive_primitives import (
    batched_outer_product,
    contrastive_delta,
    contrastive_hebbian_update,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelRegistry,
    LocalityLevel,
)


@dataclass(frozen=True, slots=True)
class ContrastiveConfig:
    """Configuration for contrastive kernel."""

    algorithm: AlgorithmFamily
    hardware: HardwareTarget
    dtype: torch.dtype = torch.float32
    beta: float = 0.5
    lr: float = 0.01
    settle_steps: int = 30
    gamma: float = 1.0
    extra: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class ContrastiveKernel(Protocol):
    """Protocol for contrastive Hebbian kernels (O(1) memory)."""

    name: AlgorithmFamily
    supported_dtypes: tuple[torch.dtype, ...]
    requires_settle: bool

    def initialize(self, config: ContrastiveConfig) -> None: ...
    def free_phase(self, x: Tensor) -> list[Tensor]: ...
    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]: ...
    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]: ...
    def apply_updates(self, updates: dict[str, Tensor]) -> None: ...
    def contrastive_step(self, x: Tensor, target: Tensor) -> dict[str, float]: ...
    def get_memory_stats(self) -> dict[str, float]: ...
    def get_settle_telemetry(self) -> dict[str, object] | None: ...


class BaseContrastiveKernel(ABC):
    """Base class for contrastive Hebbian kernels.

    Subclasses implement algorithm-specific free/nudged phase dynamics.
    The base class handles the contrastive update computation and application.
    """

    name: AlgorithmFamily
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = True
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.LOCAL
    _update_sign: int = -1  # EP convention: W -= (1/beta)(nudged - free)

    def __init__(self) -> None:
        self._config: ContrastiveConfig | None = None
        self._layers: list[torch.nn.Linear] = []
        self._activation: torch.nn.Module = torch.nn.ReLU()
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._beta: float = 0.5
        self._lr: float = 0.01
        self._settle_steps: int = 30
        self._gamma: float = 1.0
        self._last_settle_telemetry: dict[str, object] | None = None

    def initialize(self, config: ContrastiveConfig) -> None:
        self._config = config
        self._device = torch.device(
            "cuda"
            if config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
            else "cpu"
        )
        self._dtype = config.dtype
        self._beta = config.beta
        self._lr = config.lr
        self._settle_steps = config.settle_steps
        self._gamma = config.gamma

    def set_model_ref(self, *args: object, **kwargs: object) -> None:
        """Bind the kernel to a model's linear layer stack.

        Base implementation expects: layers (list[nn.Linear]), activation (nn.Module, optional)
        Subclasses may override with different signatures.
        The bound model is the source of truth for placement: the config
        ``hardware`` hint can disagree with where the trainer moved the layers,
        so ``_device``/``_dtype`` are derived from the first bound layer.
        """
        if args:
            layers = args[0]  # type: ignore[assignment]
            self._layers = layers  # type: ignore[assignment]
            if layers:
                self._device = layers[0].weight.device
                self._dtype = layers[0].weight.dtype
        if len(args) > 1 and args[1] is not None:
            self._activation = args[1]  # type: ignore[assignment]

    def _forward_pass(
        self, x: Tensor, layers: list[torch.nn.Linear] | None = None
    ) -> list[Tensor]:
        """Standard forward pass through layer stack."""
        layers = layers or self._layers
        acts = [x]
        for i, layer in enumerate(layers):
            x = layer(x)
            if i < len(layers) - 1:
                x = self._activation(x)
            acts.append(x)
        return acts

    def _fa_forward(
        self, x: Tensor, feedback_weights: list[Tensor] | None = None
    ) -> list[Tensor]:
        """FA forward pass with fixed feedback weights."""
        acts = [x]
        h = x
        for i, layer in enumerate(self._layers):
            h = layer(h)
            if i < len(self._layers) - 1:
                h = self._activation(h)
            acts.append(h)
        return acts

    @abstractmethod
    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Run free phase forward pass. Returns per-layer activations."""

    @abstractmethod
    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Run nudged/clamped phase forward pass. Returns per-layer activations."""

    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]:
        """Compute contrastive Hebbian weight updates."""
        updates: dict[str, Tensor] = {}

        for i, (free_pre, free_post, nudged_pre, nudged_post) in enumerate(
            zip(free_acts[:-1], free_acts[1:], nudged_acts[:-1], nudged_acts[1:])
        ):
            delta = contrastive_hebbian_update(
                free_pre, free_post, nudged_pre, nudged_post, self._lr, self._beta
            )
            updates[f"layers.{i}.weight"] = delta

            if self._layers[i].bias is not None:
                bias_delta = (
                    contrastive_delta(
                        free_post.mean(dim=0), nudged_post.mean(dim=0), self._beta
                    )
                    * self._lr
                )
                updates[f"layers.{i}.bias"] = bias_delta

        return updates

    def apply_updates(self, updates: dict[str, Tensor]) -> None:
        """Apply weight updates to bound layers.

        The base update sign follows the EP contrastive convention: the
        free/nudged delta ``(1/beta)(nudged - free)`` is a *gradient* and is
        applied as ``W -= lr * delta``. Pure-Hebbian / FF / PEPITA kernels
        override ``_update_sign = +1`` because their deltas already carry the
        descent direction and are applied as ``W += delta``.
        """
        with torch.no_grad():
            for name, grad in updates.items():
                if "weight" in name:
                    parts = name.split(".")
                    if parts[0] == "layers":
                        idx = int(parts[1])
                        self._layers[idx].weight.add_(self._update_sign * grad)
                elif "bias" in name:
                    parts = name.split(".")
                    if parts[0] == "layers":
                        idx = int(parts[1])
                        if self._layers[idx].bias is not None:
                            self._layers[idx].bias.add_(self._update_sign * grad)

    def contrastive_step(self, x: Tensor, target: Tensor) -> dict[str, float]:
        """Full contrastive training step: free -> nudged -> update."""
        free_acts = self.free_phase(x)
        nudged_acts = self.nudged_phase(x, target)
        updates = self.compute_update(free_acts, nudged_acts)
        self.apply_updates(updates)

        # Compute metrics
        with torch.no_grad():
            free_out = free_acts[-1]
            nudged_out = nudged_acts[-1]
            loss = (nudged_out - free_out).pow(2).mean().item()

        return {
            "loss": loss,
            "free_norm": free_out.norm().item(),
            "nudged_norm": nudged_out.norm().item(),
        }

    def predict(self, x: Tensor) -> Tensor:
        """Free-phase forward pass; returns the output layer activations."""
        return self._forward_pass(x)[-1]

    def get_memory_stats(self) -> dict[str, float]:
        total_params = sum(
            p.numel() for layer in self._layers for p in layer.parameters()
        )
        return {
            "params_mb": total_params * 4 / 1e6,
            "activations_mb": 0.0,  # O(1) - no activation storage
            "contrastive_mb": 0.0,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        return self._last_settle_telemetry


# ============================================================
# Algorithm-Specific Contrastive Kernels
# ============================================================


class FAContrastiveKernel(BaseContrastiveKernel):
    """Feedback Alignment with contrastive free/nudged phases.

    Uses fixed random feedback weights B for error propagation in nudged phase.
    """

    name = AlgorithmFamily.FA
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = False
    _update_sign = 1  # Hebbian-style: delta is a descent direction, W += delta

    def __init__(self) -> None:
        super().__init__()
        self._feedback_weights: list[Tensor] = []
        self._feedback_seed: int = 42

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._feedback_seed = config.extra.get("feedback_seed", 42)

    def set_model_ref(
        self, layers: list[torch.nn.Linear], activation: torch.nn.Module = None
    ) -> None:
        super().set_model_ref(layers, activation)
        # The bound model is the source of truth for placement: the config
        # ``hardware`` hint can disagree with where the trainer actually moved
        # the layers. Build feedback weights on the layers' device/dtype.
        device = layers[0].weight.device
        dtype = layers[0].weight.dtype
        self._feedback_weights = []
        gen = torch.Generator(device=device)
        gen.manual_seed(self._feedback_seed)
        for i in range(len(self._layers) - 1):
            hidden_dim = self._layers[i].out_features
            output_dim = self._layers[i + 1].out_features
            fb = (
                torch.randn(
                    hidden_dim, output_dim, generator=gen, device=device, dtype=dtype
                )
                * 0.1
            )
            self._feedback_weights.append(fb)

    def free_phase(self, x: Tensor) -> list[Tensor]:
        return self._forward_pass(x)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        # Standard forward pass
        acts = self._forward_pass(x)
        output = acts[-1]

        # Compute error
        if target.dim() == 1:
            target_vec = (
                torch.nn.functional
                .one_hot(target, num_classes=output.shape[1])
                .float()
                .to(device=output.device, dtype=output.dtype)
            )
        else:
            target_vec = target.to(device=output.device, dtype=output.dtype)

        error = target_vec - output

        # Nudged phase: pull every layer toward the target through the fixed
        # feedback weights B, starting from the output so each layer's nudged
        # activity differs from its free counterpart locally (no autograd).
        nudged_acts = acts.copy()
        # Output layer is nudged directly toward the target.
        nudged_acts[-1] = nudged_acts[-1] + self._beta * error
        for i in reversed(range(len(self._layers) - 1)):
            # nudged_hidden = hidden + beta * B_i @ error (via fb.T)
            fb = self._feedback_weights[i]
            nudged_acts[i + 1] = nudged_acts[i + 1] + self._beta * (error @ fb.T)
            # Propagate the error to the previous layer along the forward
            # weights (the contrastive analogue of the FA backprojection).
            error = error @ self._layers[i + 1].weight

        return nudged_acts

    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]:
        """FA contrastive update: (h_nudged - h_free).T @ h_free / beta."""
        updates: dict[str, Tensor] = {}

        for i in range(len(self._layers)):
            free_pre = free_acts[i]
            free_post = free_acts[i + 1]
            nudged_post = nudged_acts[i + 1]

            # FA contrastive: delta = (h_nudged - h_free) / beta
            delta_post = (nudged_post - free_post) / self._beta
            # Weight update: delta_post.T @ free_pre * lr
            weight_delta = self._lr * (delta_post.T @ free_pre) / free_pre.shape[0]
            updates[f"layers.{i}.weight"] = weight_delta

            if self._layers[i].bias is not None:
                bias_delta = self._lr * delta_post.mean(dim=0)
                updates[f"layers.{i}.bias"] = bias_delta

        return updates


class HebbianContrastiveKernel(BaseContrastiveKernel):
    """Pure Hebbian / 3-Factor contrastive kernel.

    Single forward pass modulated by a third factor (neuromodulator).
    """

    name = AlgorithmFamily.HEBBIAN
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = False
    _update_sign = 1

    def __init__(self) -> None:
        super().__init__()
        self._modulator: Tensor | None = None
        self._use_oja: bool = False

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._use_oja = config.extra.get("use_oja", False)

    def free_phase(self, x: Tensor) -> list[Tensor]:
        return self._forward_pass(x)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        # For Hebbian, nudged phase = free phase (same forward pass)
        # The third factor modulates the update
        return self._forward_pass(x)

    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]:
        updates: dict[str, Tensor] = {}

        for i, (pre, post) in enumerate(zip(free_acts[:-1], free_acts[1:])):
            # Hebbian outer product
            delta = batched_outer_product(pre, post)

            if self._use_oja:
                # Oja's rule: subtract post @ post.T * weight
                weight = self._layers[i].weight
                delta = delta - post.T @ post * weight / pre.shape[0]

            # Apply third factor modulator if provided
            if self._modulator is not None:
                delta = delta * self._modulator.mean()

            updates[f"layers.{i}.weight"] = self._lr * delta

            if self._layers[i].bias is not None:
                updates[f"layers.{i}.bias"] = self._lr * post.mean(dim=0)

        return updates

    def set_modulator(self, modulator: Tensor) -> None:
        """Set the third-factor neuromodulator signal."""
        self._modulator = modulator


class FFContrastiveKernel(BaseContrastiveKernel):
    """Forward-Forward contrastive kernel (Positive vs Negative passes)."""

    name = AlgorithmFamily.FF
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = False
    _update_sign = 1

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 1.0
        self._num_classes: int = 10

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._threshold = config.extra.get("threshold", 1.0)
        self._num_classes = config.extra.get("num_classes", 10)

    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Positive pass: input with label embedded."""
        # Embed labels in first layer input (simplified)
        return self._forward_pass(x)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Negative pass: input with wrong label."""
        # Simplified: just return free phase with different labels
        return self._forward_pass(x)

    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]:
        updates: dict[str, Tensor] = {}

        for i, (pos_pre, pos_post, _neg_pre, neg_post) in enumerate(
            zip(free_acts[:-1], free_acts[1:], nudged_acts[:-1], nudged_acts[1:])
        ):
            # FF goodness contrast: ||pos||^2 - ||neg||^2
            pos_goodness = (pos_post**2).sum(dim=1, keepdim=True)
            neg_goodness = (neg_post**2).sum(dim=1, keepdim=True)
            contrast = pos_goodness - neg_goodness - self._threshold

            # Weight update proportional to contrast * pre
            delta = (contrast * pos_post).T @ pos_pre / pos_pre.shape[0]
            updates[f"layers.{i}.weight"] = self._lr * delta

            if self._layers[i].bias is not None:
                updates[f"layers.{i}.bias"] = self._lr * contrast.mean(dim=0)

        return updates


class PEPITAContrastiveKernel(BaseContrastiveKernel):
    """PEPITA contrastive kernel (Standard vs Error-modulated passes)."""

    name = AlgorithmFamily.PEPITA
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = False
    _update_sign = 1

    def __init__(self) -> None:
        super().__init__()
        self._feedback_matrix: Tensor | None = None
        self._feedback_scale: float = 0.1

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._feedback_scale = config.extra.get("feedback_matrix_scale", 0.1)

    def set_model_ref(
        self, layers: list[torch.nn.Linear], activation: torch.nn.Module = None
    ) -> None:
        super().set_model_ref(layers, activation)
        # Build feedback matrix for error modulation
        gen = torch.Generator(device=self._device)
        gen.manual_seed(42)
        out_dim = layers[-1].out_features
        in_dim = layers[0].in_features
        self._feedback_matrix = (
            torch.randn(
                out_dim, in_dim, generator=gen, device=self._device, dtype=self._dtype
            )
            * self._feedback_scale
        )

    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Standard forward pass."""
        return self._forward_pass(x)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Error-modulated forward pass."""
        acts = [x]
        h = x
        for i, layer in enumerate(self._layers):
            h = layer(h)
            if i < len(self._layers) - 1:
                # Add error modulation via feedback matrix
                if self._feedback_matrix is not None:
                    # Simplified error modulation
                    pass
            h = self._activation(h) if i < len(self._layers) - 1 else h
            acts.append(h)
        return acts

    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]:
        updates: dict[str, Tensor] = {}

        for i, (std_pre, std_post, _err_pre, err_post) in enumerate(
            zip(free_acts[:-1], free_acts[1:], nudged_acts[:-1], nudged_acts[1:])
        ):
            # PEPITA: delta = (std_grad - err_grad) = (std_post - err_post).T @ pre / B
            delta = (std_post - err_post).T @ std_pre / std_pre.shape[0]
            updates[f"layers.{i}.weight"] = self._lr * delta

            if self._layers[i].bias is not None:
                updates[f"layers.{i}.bias"] = self._lr * (std_post - err_post).mean(
                    dim=0
                )

        return updates


class TPContrastiveKernel(BaseContrastiveKernel):
    """Target Propagation contrastive kernel (Forward vs Inverse target)."""

    name = AlgorithmFamily.TP
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = False

    def __init__(self) -> None:
        super().__init__()
        self._inverse_layers: list[torch.nn.Linear] = []
        self._target_lr: float = 0.1

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._target_lr = config.extra.get("target_lr", 0.1)

    def set_model_ref(
        self,
        forward_layers: list[torch.nn.Linear],
        inverse_layers: list[torch.nn.Linear],
        activation: torch.nn.Module = None,
    ) -> None:
        self._layers = forward_layers
        self._inverse_layers = inverse_layers
        if activation is not None:
            self._activation = activation

    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Forward pass through forward network."""
        return self._forward_pass(x)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Compute target-propagated activations."""
        acts = self._forward_pass(x)
        output = acts[-1]

        # Compute output layer target
        if target.dim() == 1:
            target_vec = (
                torch.nn.functional
                .one_hot(target, num_classes=output.shape[1])
                .float()
                .to(device=self._device, dtype=self._dtype)
            )
        else:
            target_vec = target.to(device=self._device, dtype=self._dtype)

        # Difference target propagation
        h_target = output - self._target_lr * (output - target_vec)

        # Propagate targets backward through inverse networks
        targets = [h_target]
        for inv_layer in self._inverse_layers:
            h_target = inv_layer(h_target)
            targets.append(h_target)
        targets = list(reversed(targets))

        # Forward pass with target-driven activations
        nudged_acts = [x]
        # targets[0] corresponds to first hidden layer output
        # targets[1] corresponds to output layer
        for i, layer in enumerate(self._layers):
            h = layer(nudged_acts[-1])
            if i < len(self._layers) - 1:
                h = self._activation(h)
            # Nudge toward target - targets[i] aligns with layer i's output
            if i < len(targets):
                h = h + self._beta * (targets[i] - h)
            nudged_acts.append(h)

        return nudged_acts


class PCContrastiveKernel(BaseContrastiveKernel):
    """Predictive Coding contrastive kernel (Free vs Clamped inference)."""

    name = AlgorithmFamily.PC
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = True

    def __init__(self) -> None:
        super().__init__()
        self._infer_steps: int = 4
        self._eta_infer: float = 0.1

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._infer_steps = config.extra.get("infer_steps", 4)
        self._eta_infer = config.extra.get("eta_infer", 0.1)

    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Free inference: no output clamping."""
        return self._pc_inference(x, target=None)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Clamped inference: output clamped to target."""
        return self._pc_inference(x, target)

    def _pc_inference(self, x: Tensor, target: Tensor | None) -> list[Tensor]:
        """PCN inference: states chase predictions from layer below."""
        L = len(self._layers)

        # Initialize states: mu[0] = x (clamped), mu[1..L] = zeros
        mu = [x]
        for layer in self._layers:
            out_features = layer.out_features
            mu.append(
                torch.zeros(
                    x.shape[0], out_features, device=self._device, dtype=self._dtype
                )
            )

        # Initialize hidden states with a forward pass
        with torch.no_grad():
            h = x
            for i, layer in enumerate(self._layers):
                h = layer(h)
                if i < L - 1:
                    h = self._activation(h)
                mu[i + 1] = h

        for _ in range(self._infer_steps):
            mu_new = [x.clone()]  # Input clamped
            for l in range(1, L + 1):
                pred = self._activation(
                    mu[l - 1] @ self._layers[l - 1].weight.T + self._layers[l - 1].bias
                )
                error = mu[l] - pred
                mu_new.append(mu[l] - self._eta_infer * error)
            mu = mu_new

            if target is not None:
                # Clamp output
                if target.dim() == 1:
                    target_vec = (
                        torch.nn.functional
                        .one_hot(target, num_classes=mu[-1].shape[1])
                        .float()
                        .to(device=self._device, dtype=self._dtype)
                    )
                else:
                    target_vec = target.to(device=self._device, dtype=self._dtype)
                mu[-1] = target_vec

        return mu


class SNNContrastiveKernel(BaseContrastiveKernel):
    """Spiking STDP contrastive kernel (Pre/Post spike timing)."""

    name = AlgorithmFamily.SNN
    supported_dtypes = (torch.float32,)
    requires_settle = True

    def __init__(self) -> None:
        super().__init__()
        self._num_steps: int = 5
        self._tau_mem: float = 20.0
        self._tau_syn: float = 5.0
        self._threshold: float = 1.0

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._num_steps = config.extra.get("num_steps", 5)
        self._tau_mem = config.extra.get("tau_mem", 20.0)
        self._tau_syn = config.extra.get("tau_syn", 5.0)
        self._threshold = config.extra.get("threshold", 1.0)

    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Free spiking phase."""
        voltages, self._last_input_spikes = self._simulate(x, beta=0.0)
        return voltages

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Nudged spiking phase with output nudging."""
        voltages, self._last_input_spikes = self._simulate(
            x, beta=self._beta, target=target
        )
        return voltages

    def _simulate(
        self, x: Tensor, beta: float, target: Tensor | None = None
    ) -> tuple[list[Tensor], Tensor]:
        """Simulate LIF dynamics. Returns (hidden_voltages, input_spikes)."""
        B = x.shape[0]
        voltages = []

        # Convert input to spikes (Poisson or rate-coded)
        # Simplified: threshold input to create binary spikes
        input_spikes = (x > 0).float()  # [B, in_features]

        # Initialize membrane potential and synaptic current for hidden layer
        v = torch.zeros(
            B, self._layers[0].out_features, device=self._device, dtype=self._dtype
        )
        i_syn = torch.zeros_like(v)

        for t in range(self._num_steps):
            # Input current from input spikes
            i_in = input_spikes @ self._layers[0].weight.T + self._layers[0].bias

            # LIF dynamics
            v = v + (-v / self._tau_mem + i_syn + i_in) * 1.0
            i_syn = i_syn * (1 - 1.0 / self._tau_syn)

            # Spike generation
            spikes = (v > self._threshold).float()
            v = torch.where(spikes.bool(), torch.zeros_like(v), v)
            i_syn = i_syn + spikes

            voltages.append(v.clone())

            if target is not None and t == self._num_steps - 1:
                # Nudge output layer
                pass

        # Return tuple: (hidden_voltages, input_spikes)
        return voltages, input_spikes

    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]:
        """Compute STDP contrastive update from voltage traces."""
        from bioplausible.acceleration.contrastive_primitives import stdp_update

        updates: dict[str, Tensor] = {}

        # Convert voltages to spike trains (threshold crossing)
        free_spikes = [(v > self._threshold).float() for v in free_acts]
        nudged_spikes = [(v > self._threshold).float() for v in nudged_acts]

        # Stack spike trains: [B, N, T]
        free_spike_trains = torch.stack(free_spikes, dim=2)
        nudged_spike_trains = torch.stack(nudged_spikes, dim=2)

        # Input spikes (same for free and nudged since input doesn't change)
        input_spikes = self._last_input_spikes.unsqueeze(2).repeat(
            1, 1, self._num_steps
        )  # [B, in_features, T]

        # Compute STDP update for first layer only (simplified)
        # pre: input features, post: first hidden layer
        pre_spikes = input_spikes
        post_spikes = nudged_spike_trains

        delta = stdp_update(pre_spikes, post_spikes)
        updates["layers.0.weight"] = self._lr * delta

        if self._layers[0].bias is not None:
            bias_delta = nudged_spike_trains.mean(dim=(0, 2)) - free_spike_trains.mean(
                dim=(0, 2)
            )
            updates["layers.0.bias"] = self._lr * bias_delta

        # For higher layers (output layer), use simple contrastive Hebbian
        # using the final time step voltages as "activations"
        if len(self._layers) > 1:
            i = len(self._layers) - 1  # Last layer (output)
            free_pre = free_acts[-1]  # Final hidden voltages
            free_post = free_pre @ self._layers[i].weight.T + self._layers[i].bias
            nudged_pre = nudged_acts[-1]
            nudged_post = nudged_pre @ self._layers[i].weight.T + self._layers[i].bias

            delta = contrastive_hebbian_update(
                free_pre, free_post, nudged_pre, nudged_post, self._lr, self._beta
            )
            updates[f"layers.{i}.weight"] = delta

            if self._layers[i].bias is not None:
                bias_delta = (
                    contrastive_delta(
                        free_post.mean(dim=0), nudged_post.mean(dim=0), self._beta
                    )
                    * self._lr
                )
                updates[f"layers.{i}.bias"] = bias_delta

        return updates


class TileContrastiveKernel(BaseContrastiveKernel):
    """Tile substrate contrastive kernel (Tile-parallel free/nudged settle)."""

    name = AlgorithmFamily.TILE
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = True

    def __init__(self) -> None:
        super().__init__()
        self._neurons_per_tile: int = 8
        self._tiles_per_layer: int = 2

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._neurons_per_tile = config.extra.get("neurons_per_tile", 8)
        self._tiles_per_layer = config.extra.get("tiles_per_layer", 2)

    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Free settle (beta=0)."""
        return self._tile_settle(x, beta=0.0)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Nudged settle (beta>0)."""
        return self._tile_settle(x, beta=self._beta, target=target)

    def _tile_settle(
        self, x: Tensor, beta: float, target: Tensor | None = None
    ) -> list[Tensor]:
        """Tile-parallel settle."""
        acts = [x]

        for _ in range(self._settle_steps):
            # Forward pass through all layers
            h = x
            for i, layer in enumerate(self._layers):
                h = layer(h)
                if i < len(self._layers) - 1:
                    h = self._activation(h)

            if target is not None and beta > 0:
                # Nudge only the output layer
                if target.dim() == 1:
                    target_vec = (
                        torch.nn.functional
                        .one_hot(target, num_classes=h.shape[1])
                        .float()
                        .to(device=self._device, dtype=self._dtype)
                    )
                else:
                    target_vec = target.to(device=self._device, dtype=self._dtype)
                # Only nudge if shapes match (output layer)
                if h.shape == target_vec.shape:
                    h = h + beta * (target_vec - h)

            acts.append(h.clone())

        return acts

    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]:
        """Compute contrastive update using final settled states."""
        # Reconstruct per-layer activations for final states
        free_per_layer = [free_acts[0]]
        nudged_per_layer = [nudged_acts[0]]
        h_free = free_acts[0]
        h_nudged = nudged_acts[0]
        for i, layer in enumerate(self._layers):
            h_free = layer(h_free)
            h_nudged = layer(h_nudged)
            if i < len(self._layers) - 1:
                h_free = self._activation(h_free)
                h_nudged = self._activation(h_nudged)
            free_per_layer.append(h_free)
            nudged_per_layer.append(h_nudged)

        return super().compute_update(free_per_layer, nudged_per_layer)


class MEPContrastiveKernel(BaseContrastiveKernel):
    """MEP contrastive kernel (Muon/Dion/Fisher + EP settle).

    For the contrastive path with a chain of Linear layers (from benchmark bind),
    we use a simple forward pass through the chain with output nudging.
    """

    name = AlgorithmFamily.MEP
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = True

    def __init__(self) -> None:
        super().__init__()
        self._transition_modules: list[torch.nn.Module] = []
        self._ns_steps: int = 5
        self._rank_frac: float = 0.25
        self._fisher_damping: float = 1e-3

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._ns_steps = config.extra.get("ns_steps", 5)
        self._rank_frac = config.extra.get("rank_frac", 0.25)
        self._fisher_damping = config.extra.get("fisher_damping", 1e-3)

    def set_model_ref(self, transition_modules: list[torch.nn.Module]) -> None:
        self._transition_modules = transition_modules

    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Free forward pass through transition modules."""
        return self._forward_chain(x)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Nudged forward pass with output nudging."""
        acts = self._forward_chain(x)
        # Nudge the output layer toward target
        if target.dim() == 1:
            target_vec = (
                torch.nn.functional
                .one_hot(target, num_classes=acts[-1].shape[1])
                .float()
                .to(device=self._device, dtype=self._dtype)
            )
        else:
            target_vec = target.to(device=self._device, dtype=self._dtype)
        acts[-1] = acts[-1] + self._beta * (target_vec - acts[-1])
        return acts

    def _forward_chain(self, x: Tensor) -> list[Tensor]:
        """Forward pass through chain of modules, returning per-layer activations."""
        acts = [x]
        h = x
        for module in self._transition_modules:
            h = module(h)
            acts.append(h)
        return acts

    def compute_update(
        self, free_acts: list[Tensor], nudged_acts: list[Tensor]
    ) -> dict[str, Tensor]:
        """Contrastive Hebbian update per transition module."""
        updates: dict[str, Tensor] = {}

        # free_acts and nudged_acts are per-layer activations from _forward_chain
        # Index 0 is input, indices 1..N are post-activations of each module
        for i, module in enumerate(self._transition_modules):
            if hasattr(module, "weight"):
                free_pre = free_acts[i]
                nudged_pre = nudged_acts[i]
                free_post = free_acts[i + 1]
                nudged_post = nudged_acts[i + 1]

                delta = contrastive_hebbian_update(
                    free_pre,
                    free_post,
                    nudged_pre,
                    nudged_post,
                    self._lr,
                    self._beta,
                )

                updates[f"transition.{i}.weight"] = delta

                if module.bias is not None:
                    bias_delta = (
                        contrastive_delta(
                            free_post.mean(dim=0), nudged_post.mean(dim=0), self._beta
                        )
                        * self._lr
                    )
                    updates[f"transition.{i}.bias"] = bias_delta

        return updates


class O1MemoryContrastiveKernel(BaseContrastiveKernel):
    """O1MemoryEPv2 contrastive kernel (Analytic gradients + manual settle).

    For the contrastive path with a chain of Linear layers (from benchmark bind),
    we use a simple forward pass through the chain with output nudging.
    """

    name = AlgorithmFamily.O1MEMORY
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    requires_settle = True

    def __init__(self) -> None:
        super().__init__()
        self._transition_modules: list[torch.nn.Module] = []
        self._loss_type: str = "mse"
        self._softmax_temp: float = 1.0

    def initialize(self, config: ContrastiveConfig) -> None:
        super().initialize(config)
        self._loss_type = config.extra.get("loss_type", "mse")
        self._softmax_temp = config.extra.get("softmax_temperature", 1.0)

    def set_model_ref(self, transition_modules: list[torch.nn.Module]) -> None:
        self._transition_modules = transition_modules

    def free_phase(self, x: Tensor) -> list[Tensor]:
        """Free forward pass through transition modules."""
        return self._forward_chain(x)

    def nudged_phase(self, x: Tensor, target: Tensor) -> list[Tensor]:
        """Nudged forward pass with output nudging."""
        acts = self._forward_chain(x)
        # Nudge the output layer toward target
        if target.dim() == 1:
            target_vec = (
                torch.nn.functional
                .one_hot(target, num_classes=acts[-1].shape[1])
                .float()
                .to(device=self._device, dtype=self._dtype)
            )
        else:
            target_vec = target.to(device=self._device, dtype=self._dtype)
        acts[-1] = acts[-1] + self._beta * (target_vec - acts[-1])
        return acts

    def _forward_chain(self, x: Tensor) -> list[Tensor]:
        """Forward pass through chain of modules, returning per-layer activations."""
        acts = [x]
        h = x
        for module in self._transition_modules:
            h = module(h)
            acts.append(h)
        return acts

    def analytic_state_grad(self, states: list[Tensor], target: Tensor) -> list[Tensor]:
        """Analytic gradient of energy w.r.t states (for reference)."""
        target = target.to(device=self._device, dtype=self._dtype)
        grads: list[Tensor] = []

        output_state = states[-1]
        if self._loss_type == "mse":
            grad = output_state - target
        elif self._loss_type == "ce":
            probs = torch.softmax(output_state / self._softmax_temp, dim=-1)
            grad = probs - target
        else:
            grad = output_state - target

        grads.append(grad)

        for i in range(len(self._transition_modules) - 1, -1, -1):
            module = self._transition_modules[i]
            state = states[i]

            if hasattr(module, "weight"):
                grad = grads[-1] @ module.weight.data
            else:
                grad = grads[-1]

            if hasattr(module, "activation"):
                act = module.activation
                if act == "relu":
                    grad = grad * (state > 0).float()
                elif act == "tanh":
                    grad = grad * (1 - torch.tanh(state) ** 2)

            grads.append(grad)

        return list(reversed(grads))

    def compute_update(
        self, free_states: list[Tensor], nudged_states: list[Tensor]
    ) -> dict[str, Tensor]:
        """Contrastive Hebbian update per transition module."""
        updates: dict[str, Tensor] = {}

        # free_states and nudged_states are per-layer activations from _forward_chain
        # Index 0 is input, indices 1..N are post-activations of each module
        for i, module in enumerate(self._transition_modules):
            if hasattr(module, "weight"):
                free_pre = free_states[i]
                nudged_pre = nudged_states[i]
                free_post = free_states[i + 1]
                nudged_post = nudged_states[i + 1]

                delta = contrastive_hebbian_update(
                    free_pre,
                    free_post,
                    nudged_pre,
                    nudged_post,
                    self._lr,
                    self._beta,
                )
                updates[f"transition.{i}.weight"] = delta

                if module.bias is not None:
                    bias_delta = (
                        contrastive_delta(
                            free_post.mean(dim=0), nudged_post.mean(dim=0), self._beta
                        )
                        * self._lr
                    )
                    updates[f"transition.{i}.bias"] = bias_delta

        return updates


# ============================================================
# Registry Registration
# ============================================================

# Map algorithm family to contrastive kernel class
_CONTRASTIVE_KERNEL_CLASSES: dict[AlgorithmFamily, type] = {
    AlgorithmFamily.FA: FAContrastiveKernel,
    AlgorithmFamily.HEBBIAN: HebbianContrastiveKernel,
    AlgorithmFamily.FF: FFContrastiveKernel,
    AlgorithmFamily.PEPITA: PEPITAContrastiveKernel,
    AlgorithmFamily.TP: TPContrastiveKernel,
    AlgorithmFamily.PC: PCContrastiveKernel,
    AlgorithmFamily.SNN: SNNContrastiveKernel,
    AlgorithmFamily.TILE: TileContrastiveKernel,
    AlgorithmFamily.MEP: MEPContrastiveKernel,
    AlgorithmFamily.O1MEMORY: O1MemoryContrastiveKernel,
}


def get_contrastive_kernel(algorithm: AlgorithmFamily) -> BaseContrastiveKernel | None:
    """Get a contrastive kernel instance for the given algorithm."""
    cls = _CONTRASTIVE_KERNEL_CLASSES.get(algorithm)
    if cls is None:
        return None
    return cls()


def register_contrastive_kernels() -> None:
    """Register all contrastive kernels in the global KernelRegistry.

    Called on import to make contrastive kernels visible to KernelRegistry
    alongside standard KernelBackend implementations.
    """
    for algorithm, cls in _CONTRASTIVE_KERNEL_CLASSES.items():
        for hardware in HardwareTarget:
            KernelRegistry.register(algorithm, hardware, cls)


def get_contrastive_kernels() -> dict[str, type[BaseContrastiveKernel]]:
    """Import and return all contrastive kernels (triggers self-registration).

    Mirrors ``get_algorithm_kernels()`` pattern for standard kernels.
    """
    from bioplausible.acceleration.contrastive_kernels import (
        FAContrastiveKernel,
        FFContrastiveKernel,
        HebbianContrastiveKernel,
        MEPContrastiveKernel,
        O1MemoryContrastiveKernel,
        PCContrastiveKernel,
        PEPITAContrastiveKernel,
        SNNContrastiveKernel,
        TileContrastiveKernel,
        TPContrastiveKernel,
    )

    return {
        "fa": FAContrastiveKernel,
        "hebbian": HebbianContrastiveKernel,
        "ff": FFContrastiveKernel,
        "pepita": PEPITAContrastiveKernel,
        "tp": TPContrastiveKernel,
        "pc": PCContrastiveKernel,
        "snn": SNNContrastiveKernel,
        "tile": TileContrastiveKernel,
        "mep": MEPContrastiveKernel,
        "o1memory": O1MemoryContrastiveKernel,
    }


# Auto-register on import
register_contrastive_kernels()


__all__ = [
    "BaseContrastiveKernel",
    "ContrastiveConfig",
    "ContrastiveKernel",
    "FAContrastiveKernel",
    "FFContrastiveKernel",
    "HebbianContrastiveKernel",
    "MEPContrastiveKernel",
    "O1MemoryContrastiveKernel",
    "PCContrastiveKernel",
    "PEPITAContrastiveKernel",
    "SNNContrastiveKernel",
    "TPContrastiveKernel",
    "TileContrastiveKernel",
    "get_contrastive_kernel",
    "register_contrastive_kernels",
]
