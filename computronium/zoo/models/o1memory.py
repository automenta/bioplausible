"""O1Memory Model with SettleProtocol integration.

Wraps the O1MemoryEPv2 analytic settling into a BioModel that
implements SettleProtocol for unified convergence instrumentation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from computronium.config.unified import ModelConfig, resolve_hidden_dims
from computronium.core.local_learning.settling import (
    SettleConfig,
    SettleProtocol,
    SettleTelemetry,
    settle_universal,
)
from computronium.core.model import BioModel
from computronium.core.model_status import status_tag
from computronium.core.registry import LocalityLevel, register_model
from computronium.zoo.mep.optimizers.o1_memory_v2 import (
    O1MemoryEPv2,
    _capture_states_no_grad,
    analytic_state_gradients,
    settle_manual_o1,
)


@register_model(
    "o1memory",
    family="mep",
    locality_level=LocalityLevel.EQUILIBRIUM,
    tags=["mep", "o1memory", "equilibrium", "ep", status_tag("experimental")],
)
class O1MemoryModel(BioModel, SettleProtocol):
    """O(1) Memory EP model with SettleProtocol support.

    Uses analytic state gradients for true O(1) activation memory
    during settling. Implements SettleProtocol (Family B: activations list)
    for unified telemetry.
    """

    algorithm_name = "O1Memory Equilibrium Propagation"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        beta: float = 0.5,
        settle_steps: int = 30,
        settle_lr: float = 0.15,
        loss_type: str = "cross_entropy",
        softmax_temperature: float = 1.0,
        momentum: float = 0.5,
        optimizer_lr: float = 0.01,
        optimizer_momentum: float = 0.9,
        weight_decay: float = 0.0005,
        **kwargs,
    ) -> None:
        # Extract O1Memory-specific config before super().__init__
        if config is not None:
            extra = config.extra or {}
            beta = extra.get("beta", beta)
            settle_steps = extra.get("settle_steps", settle_steps)
            settle_lr = extra.get("settle_lr", settle_lr)
            loss_type = extra.get("loss_type", loss_type)
            softmax_temperature = extra.get("softmax_temperature", softmax_temperature)
            momentum = extra.get("momentum", momentum)
            optimizer_lr = extra.get("optimizer_lr", optimizer_lr)
            optimizer_momentum = extra.get("optimizer_momentum", optimizer_momentum)
            weight_decay = extra.get("weight_decay", weight_decay)

        # Build config if not provided
        if config is None:
            config = ModelConfig(
                name=self.algorithm_name,
                input_dim=kwargs.get("input_dim", 784),
                output_dim=kwargs.get("output_dim", 10),
                hidden_dims=[kwargs.get("hidden_dim", 256)],
                learning_rate=optimizer_lr,
                beta=beta,
                max_steps=settle_steps,
                extra={
                    "settle_steps": settle_steps,
                    "settle_lr": settle_lr,
                    "loss_type": loss_type,
                    "softmax_temperature": softmax_temperature,
                    "momentum": momentum,
                    "optimizer_lr": optimizer_lr,
                    "optimizer_momentum": optimizer_momentum,
                    "weight_decay": weight_decay,
                },
            )

        super().__init__(config, **kwargs)

        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
        self.momentum = momentum
        self.optimizer_lr = optimizer_lr
        self.optimizer_momentum = optimizer_momentum
        self.weight_decay = weight_decay

        # Build layers
        self._build_layers()

        # O1Memory optimizer (handles settling and parameter updates)
        self.o1_optimizer = O1MemoryEPv2(
            params=self.parameters(),
            model=self,
            lr=self.optimizer_lr,
            momentum=self.optimizer_momentum,
            weight_decay=self.weight_decay,
            settle_steps=self.settle_steps,
            settle_lr=self.settle_lr,
            beta=self.beta,
            loss_type=self.loss_type,
            backend="pytorch",
        )

        # SettleProtocol attributes
        self.convergence_threshold = (
            1e-4  # O1Memory doesn't have built-in convergence check
        )
        self.convergence_start = 3
        self.max_steps = self.settle_steps

        # Transient state for settle_universal
        self._settle_beta: float = 0.0
        self._settle_target: torch.Tensor | None = None
        self._last_activations: list[torch.Tensor] | None = None
        self._last_settle_converged: bool = False
        self._last_settle_steps: int = 0
        self._last_settle_final_delta: float = 0.0
        self._last_settle_telemetry: SettleTelemetry | None = None

    def _build_layers(self) -> None:
        """Build the forward layers."""
        hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        n_layers = len(dims) - 1
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.Linear(dims[i], dims[i + 1])
            role = "output" if i == n_layers - 1 else "hidden"
            layer = self.apply_spectral_norm(layer, layer_role=role)
            self.layers.append(layer)

    def transition_modules(self) -> list[nn.Module]:
        """Return transition modules for state capture."""
        return list(self.layers)

    def _prepare_target(
        self, target: torch.Tensor, num_classes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert target to appropriate format."""
        if self.loss_type == "cross_entropy":
            if target.dim() > 1 and target.shape[1] > 1:
                return target.argmax(dim=1).long()
            return target.squeeze().long()
        else:
            if target.dim() == 1:
                return F.one_hot(target, num_classes=num_classes).to(dtype=dtype)
            return target.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Forward / Settle
    # ------------------------------------------------------------------

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward pass without settling (for state capture)."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        *,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[list[torch.Tensor]]]
        | tuple[torch.Tensor, dict[str, object]]
    ):
        """Settle the network and return output logits."""
        if return_dynamics:
            out, steps_taken, converged, telemetry = self._run_settle_universal(
                x,
                beta=beta,
                target=target,
                steps=steps,
                return_trajectory=return_trajectory,
                return_dynamics=return_dynamics,
            )
            if telemetry:
                dynamics = {
                    "deltas": telemetry.deltas,
                    "final_delta": telemetry.final_delta,
                    "steps_taken": telemetry.steps_taken,
                    "converged": telemetry.converged,
                    "settle_time_s": telemetry.settle_time_ms / 1000.0,
                }
            else:
                dynamics = {}
            return out, dynamics

        # Default path: use O1MemoryEPv2 settle_manual_o1
        transition_modules = self.transition_modules()
        target_vec = None
        if target is not None:
            target_vec = self._prepare_target(target, self.output_dim, x.dtype)

        states = settle_manual_o1(
            self,
            x,
            target_vec,
            beta,
            transition_modules,
            steps=steps if steps is not None else self.settle_steps,
            lr=self.settle_lr,
            momentum=self.momentum,
            loss_type=self.loss_type,
            softmax_temperature=self.softmax_temperature,
            forward_fn=self._forward_impl,
        )

        # Output is the last state
        out = (
            states[-1]
            if states
            else torch.zeros(
                x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
            )
        )
        self._last_activations = states

        if return_trajectory:
            return out, [states]
        return out

    # ------------------------------------------------------------------
    # SettleProtocol Implementation (Family B: activations list)
    # ------------------------------------------------------------------

    def _initialize_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return initial activations list for settle_universal.

        Captures initial states from transition modules without autograd.
        """
        return _capture_states_no_grad(
            self, x, list(self.transition_modules()), forward_fn=self._forward_impl
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input (beta/target stored instead)."""
        return x

    def _step(
        self,
        state: list[torch.Tensor],
        x_transformed: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Single settle step using analytic gradients.

        Runs one iteration of O(1) memory settling.
        """
        transition_modules = self.transition_modules()
        target_vec = None
        if self._settle_target is not None:
            target_vec = self._prepare_target(
                self._settle_target, self.output_dim, x_transformed.dtype
            )

        # Compute analytic gradients (no autograd!)
        grads = analytic_state_gradients(
            self,
            x_transformed,
            state,
            transition_modules,
            target_vec,
            self._settle_beta,
            loss_type=self.loss_type,
            softmax_temperature=self.softmax_temperature,
        )

        # Update states with momentum (no_grad)
        with torch.no_grad():
            for i, (s, g) in enumerate(zip(state, grads)):
                # Need momentum buffers - create if not exist
                if not hasattr(self, "_o1_momentum_buffers"):
                    self._o1_momentum_buffers = [torch.zeros_like(s) for s in state]
                buf = self._o1_momentum_buffers[i]
                buf.mul_(self.momentum).add_(g)
                s.sub_(buf, alpha=self.settle_lr)

        return state

    def _check_converged(
        self,
        state_new: list[torch.Tensor],
        state_old: list[torch.Tensor],
        step: int,
    ) -> bool:
        """Custom convergence check for O1Memory.

        Since O1Memory doesn't have built-in convergence, we use
        a simple relative delta check across all layers.
        """
        if step <= self.convergence_start:
            return False

        convergence_norm = 2
        max_rel_delta = 0.0
        for s_new, s_old in zip(state_new, state_old):
            abs_delta = torch.dist(s_new, s_old, p=convergence_norm).item()
            norm = s_old.norm(p=convergence_norm).item() + 1e-8
            rel_delta = abs_delta / norm
            max_rel_delta = max(max_rel_delta, rel_delta)

        return max_rel_delta < self.convergence_threshold

    def _on_step_end(
        self,
        step: int,
        state: list[torch.Tensor],
        delta: float,
    ) -> None:
        """Telemetry hook: called after each step."""
        # Telemetry collected by settle_universal

    def _on_converged(self, step: int, final_delta: float) -> None:
        """Telemetry hook: called when convergence is detected."""
        self._last_settle_converged = True
        self._last_settle_steps = step
        self._last_settle_final_delta = final_delta

    def _on_max_steps(self, step: int, final_delta: float) -> None:
        """Telemetry hook: called when max steps reached without convergence."""
        self._last_settle_converged = False
        self._last_settle_steps = step
        self._last_settle_final_delta = final_delta

    def _run_settle_universal(
        self,
        x: torch.Tensor,
        *,
        beta: float = 0.0,
        target: torch.Tensor | None = None,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> tuple[torch.Tensor, int, bool, SettleTelemetry | None]:
        """Run settle using the universal primitive with full telemetry."""
        self._settle_beta = beta
        self._settle_target = target

        # Initialize momentum buffers
        transition_modules = self.transition_modules()
        with torch.no_grad():
            init_states = _capture_states_no_grad(self, x, list(transition_modules))
        self._o1_momentum_buffers = [torch.zeros_like(s) for s in init_states]

        config = SettleConfig(
            max_steps=steps if steps is not None else self.max_steps,
            convergence_threshold=self.convergence_threshold,
            convergence_start=self.convergence_start,
        )

        state, steps_taken, converged, telemetry = settle_universal(
            self,
            x,
            config=config,
            algorithm="o1memory",
            family="B",
            hardware=self.config.device if hasattr(self.config, "device") else "cpu",
            backend="pytorch",
            return_trajectory=return_trajectory,
        )

        self._last_activations = state
        self._last_settle_telemetry = telemetry

        out = (
            state[-1]
            if state
            else torch.zeros(
                x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
            )
        )

        return out, steps_taken, converged, telemetry

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """O1Memory training step using the O1MemoryEPv2 optimizer."""
        self.train()
        self.o1_optimizer.step(x=x, target=y)

        # Metrics from free phase
        with torch.no_grad():
            transition_modules = self.transition_modules()
            states_free = settle_manual_o1(
                self,
                x,
                None,
                0.0,
                transition_modules,
                steps=self.settle_steps,
                lr=self.settle_lr,
                momentum=self.momentum,
                loss_type=self.loss_type,
                softmax_temperature=self.softmax_temperature,
                forward_fn=self._forward_impl,
            )
            out = (
                states_free[-1]
                if states_free
                else torch.zeros(
                    x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
                )
            )
            loss = F.cross_entropy(out, y).item() if out.numel() > 0 else 0.0
            from computronium.core.losses import compute_accuracy

            acc = compute_accuracy(out, y) if out.numel() > 0 else 0.0

        return {"loss": loss, "accuracy": acc}

    def get_settle_telemetry(self) -> SettleTelemetry | None:
        """Return the last settle telemetry for external consumers."""
        return self._last_settle_telemetry
