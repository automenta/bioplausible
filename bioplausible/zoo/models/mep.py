"""MEP Model with SettleProtocol integration.

Wraps the MEP Settler and EPGradient strategies into a BioModel that
implements SettleProtocol for unified convergence instrumentation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.config.unified import ModelConfig, resolve_hidden_dims
from bioplausible.core.local_learning.settling import (
    SettleConfig,
    SettleProtocol,
    SettleTelemetry,
    settle_universal,
)
from bioplausible.core.model import BioModel
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import LocalityLevel, register_model
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer
from bioplausible.zoo.mep.optimizers.energy import EnergyFunction
from bioplausible.zoo.mep.optimizers.settling import Settler
from bioplausible.zoo.mep.optimizers.strategies.gradient import EPGradient


@register_model(
    "mep",
    family="mep",
    locality_level=LocalityLevel.EQUILIBRIUM,
    tags=["mep", "equilibrium", "ep", status_tag("experimental")],
)
class MEPEqPropModel(BioModel, SettleProtocol):
    """MEP Equilibrium Propagation model with SettleProtocol support.

    Uses the MEP Settler for energy-based settling and EPGradient
    for contrastive gradient computation. Implements SettleProtocol
    (Family B: activations list) for unified telemetry.
    """

    algorithm_name = "MEP Equilibrium Propagation"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        beta: float = 0.5,
        settle_steps: int = 30,
        settle_lr: float = 0.15,
        loss_type: str = "cross_entropy",
        softmax_temperature: float = 1.0,
        tol: float = 1e-4,
        patience: int = 5,
        adaptive: bool = True,
        gradient_method: str = "contrastive",
        **kwargs,
    ) -> None:
        # Extract MEP-specific config before super().__init__
        if config is not None:
            extra = config.extra or {}
            beta = extra.get("beta", beta)
            settle_steps = extra.get("settle_steps", settle_steps)
            settle_lr = extra.get("settle_lr", settle_lr)
            loss_type = extra.get("loss_type", loss_type)
            softmax_temperature = extra.get("softmax_temperature", softmax_temperature)
            tol = extra.get("tol", tol)
            patience = extra.get("patience", patience)
            adaptive = extra.get("adaptive", adaptive)
            gradient_method = extra.get("gradient_method", gradient_method)

        # Build config if not provided
        if config is None:
            config = ModelConfig(
                name=self.algorithm_name,
                input_dim=kwargs.get("input_dim", 784),
                output_dim=kwargs.get("output_dim", 10),
                hidden_dims=[kwargs.get("hidden_dim", 256)],
                learning_rate=kwargs.get("learning_rate", 0.01),
                beta=beta,
                max_steps=settle_steps,
                extra={
                    "settle_steps": settle_steps,
                    "settle_lr": settle_lr,
                    "loss_type": loss_type,
                    "softmax_temperature": softmax_temperature,
                    "tol": tol,
                    "patience": patience,
                    "adaptive": adaptive,
                    "gradient_method": gradient_method,
                },
            )

        super().__init__(config, **kwargs)

        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
        self.tol = tol
        self.patience = patience
        self.adaptive = adaptive
        self.gradient_method = gradient_method

        # Build layers
        self._build_layers()

        # MEP components
        self.settler = Settler(
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
            softmax_temperature=self.softmax_temperature,
            tol=self.tol,
            patience=self.patience,
            adaptive=self.adaptive,
        )
        self.ep_gradient = EPGradient(
            beta=self.beta,
            settle_steps=self.settle_steps,
            settle_lr=self.settle_lr,
            loss_type=self.loss_type,
            softmax_temperature=self.softmax_temperature,
            tol=self.tol,
            patience=self.patience,
            adaptive=self.adaptive,
        )
        self.energy_fn = EnergyFunction(
            loss_type=self.loss_type,
            softmax_temperature=self.softmax_temperature,
        )

        # Optimizer for contrastive step
        self.optimizer: torch.optim.Optimizer | None = None

        # SettleProtocol attributes
        self.convergence_threshold = self.tol
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
        """Build the forward and transition layers."""
        hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        n_layers = len(dims) - 1
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.Linear(dims[i], dims[i + 1])
            role = "output" if i == n_layers - 1 else "hidden"
            layer = self.apply_spectral_norm(layer, layer_role=role)
            self.layers.append(layer)

        # Transition modules for Settler (used to capture states)
        self.transition_modules = nn.ModuleList(self.layers)

    def transition_modules(self) -> list[nn.Module]:
        """Return transition modules for Settler state capture."""
        return list(self.layers)

    def _get_structure(self) -> list[dict[str, object]]:
        """Get model structure for energy function."""
        return [{"type": "layer", "module": m} for m in self.transition_modules()]

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

        # Default path: use settler directly with internal forward pass
        # to avoid recursion (settler calls model(x) which would call forward again)
        target_vec = None
        if target is not None:
            target_vec = self._prepare_target(target, self.output_dim, x.dtype)

        # Capture states using internal forward pass (bypass public forward)
        transition_modules = self.transition_modules()
        states = self.settler._capture_states_from_transitions(
            self, x, list(transition_modules), forward_fn=self._forward_impl
        )

        if not states:
            out = torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
            self._last_activations = []
            if return_trajectory:
                return out, [[]]
            return out

        # Run settling
        states = [s.requires_grad_(True) for s in states]

        def wrapped_energy_fn(s: list[torch.Tensor]) -> torch.Tensor:
            return self.energy_fn(
                self,
                x,
                s,
                [{"type": "layer", "module": m} for m in transition_modules],
                target_vec,
                beta,
            )

        from bioplausible.core.local_learning.settling import energy_gradient_descent
        states = energy_gradient_descent(
            states,
            wrapped_energy_fn,
            steps if steps is not None else self.settle_steps,
            lr=self.settle_lr,
            momentum=Settler.MOMENTUM,
            adaptive=self.adaptive,
            tol=self.tol,
            patience=self.patience,
            step_size_growth=self.settler.step_size_growth,
            step_size_decay=self.settler.step_size_decay,
        )

        out = states[-1] if states else torch.zeros(
            x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
        )
        self._last_activations = states

        if return_trajectory:
            return out, [states]
        return out

    # ------------------------------------------------------------------
    # SettleProtocol Implementation (Family B: activations list)
    # ------------------------------------------------------------------

    def _get_states_for_settle(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get initial states for settling from transition modules."""
        # Use the same logic as settler but with our internal forward
        transition_modules = list(self.transition_modules())
        return self.settler._capture_states_from_transitions(
            self, x, transition_modules, forward_fn=self._forward_impl
        )

    def _initialize_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return initial activations list for settle_universal."""
        return self._get_states_for_settle(x)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input (not directly used; beta/target stored instead)."""
        return x

    def _step(
        self,
        state: list[torch.Tensor],
        x_transformed: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Single settle step for settle_universal.

        Runs one iteration of energy gradient descent using stored beta/target.
        """
        structure = self._get_structure()
        target_vec = None
        if self._settle_target is not None:
            target_vec = self._prepare_target(
                self._settle_target, self.output_dim, x_transformed.dtype
            )

        # Run one step of energy gradient descent
        states = [s.requires_grad_(True) for s in state]

        def wrapped_energy_fn(s: list[torch.Tensor]) -> torch.Tensor:
            return self.energy_fn(
                self,
                x_transformed,
                s,
                structure,
                target_vec,
                self._settle_beta,
            )

        # Single step of energy_gradient_descent
        from bioplausible.core.local_learning.settling import energy_gradient_descent

        return energy_gradient_descent(
            states,
            wrapped_energy_fn,
            1,  # single step
            lr=self.settle_lr,
            momentum=Settler.MOMENTUM,
            adaptive=self.adaptive,
            tol=self.tol,
            patience=self.patience,
            step_size_growth=self.settler.step_size_growth,
            step_size_decay=self.settler.step_size_decay,
        )

    def _check_converged(
        self,
        state_new: list[torch.Tensor],
        state_old: list[torch.Tensor],
        step: int,
    ) -> bool:
        """Custom convergence check matching MEP Settler behavior."""
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

        config = SettleConfig(
            max_steps=steps if steps is not None else self.max_steps,
            convergence_threshold=self.convergence_threshold,
            convergence_start=self.convergence_start,
        )

        state, steps_taken, converged, telemetry = settle_universal(
            self,
            x,
            config=config,
            algorithm="mep",
            family="B",
            hardware=self.config.device if hasattr(self.config, "device") else "cpu",
            backend="pytorch",
            return_trajectory=return_trajectory,
        )

        self._last_activations = state
        self._last_settle_telemetry = telemetry

        # Return output (last state)
        out = state[-1] if state else torch.zeros(
            x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
        )

        return out, steps_taken, converged, telemetry

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> None:
        if self.optimizer is None:
            self.optimizer = create_optimizer(
                self, OptimizerConfig(name="adam", lr=self.config.learning_rate, weight_decay=0.0)
            )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """MEP training step: free phase -> nudged phase -> contrastive update."""
        if self.gradient_method != "contrastive":
            return None  # type: ignore[return-value]

        self._ensure_optimizer()
        self.train()

        structure = self._get_structure()

        # Free phase
        states_free = self.settler.settle(
            self, x, None, 0.0, self.energy_fn, forward_fn=self._forward_impl
        )

        # Nudged phase
        states_nudged = self.settler.settle(
            self, x, y, self.beta, self.energy_fn, forward_fn=self._forward_impl
        )

        # Apply contrast via EPGradient
        self.ep_gradient.compute_gradients(
            self, x, y, energy_fn=self.energy_fn, structure_fn=lambda m: structure
        )

        # Step optimizer
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Metrics
        with torch.no_grad():
            out = states_nudged[-1] if states_nudged else torch.zeros(
                x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
            )
            loss = F.cross_entropy(out, y).item() if out.numel() > 0 else 0.0
            from bioplausible.core.losses import compute_accuracy
            acc = compute_accuracy(out, y) if out.numel() > 0 else 0.0

        return {"loss": loss, "accuracy": acc}

    def get_settle_telemetry(self) -> SettleTelemetry | None:
        """Return the last settle telemetry for external consumers."""
        return self._last_settle_telemetry
