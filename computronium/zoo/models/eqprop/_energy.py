"""Consolidated deep Equilibrium Propagation engine (Scellier & Bengio).

The ``EquilibriumMLP`` class implements the deep energy-contrastive EqProp MLP.
All variants (plain, momentum, sparse, feedback) are controlled by the
``variant`` field and ``config.extra`` knobs — no subclasses needed.

Native 5-D compositions should use ``Registry.to_system("eqprop_mlp")`` or
``SystemConfig.from_experiment()`` instead of subclassing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from computronium.config.unified import ModelConfig
from computronium.core.local_learning.settling import (
    SettleConfig,
    SettleTelemetry,
    settle_activations_list,
    settle_universal,
)
from computronium.core.model import BioModel
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer

from ._contrastive import _contrastive_step

__all__ = [
    "EquilibriumMLP",
    "LazyStats",
]

#: Discriminator for the per-step dynamics variant.
Variant = Literal["plain", "momentum", "sparse", "feedback"]


@dataclass(frozen=True, slots=True)
class LazyStats:
    """Statistics for lazy execution (computed lazily during settle)."""

    total_neurons: int = 0
    active_neurons: int = 0
    skipped_neurons: int = 0

    @property
    def skip_ratio(self) -> float:
        if self.total_neurons == 0:
            return 0.0
        return self.skipped_neurons / self.total_neurons

    @property
    def flop_savings(self) -> float:
        return self.skip_ratio * 100

    @staticmethod
    def reset() -> LazyStats:
        return LazyStats()


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1) if x.dim() > 2 else x


def _unwrap_weight(layer: nn.Module) -> nn.Module | nn.Parameter:
    """Return the storage object holding a layer's weight.

    Spectral-norm parametrization exposes the trainable matrix as
    ``layer.parametrizations.weight.original`` (an ``nn.Parameter``); plain
    layers are the ``nn.Module`` itself with a ``.weight`` attribute.
    """
    if hasattr(layer, "parametrizations") and hasattr(layer.parametrizations, "weight"):
        return layer.parametrizations.weight.original
    return layer


def _init_weight_storage(storage: nn.Module | nn.Parameter, gain: float | None) -> None:
    """Xavier (with ``gain``) or zero-initialize a layer's weight storage.

    ``gain=None`` selects zero initialization. ``storage`` is either a bare
    ``nn.Parameter`` or an ``nn.Module`` with a ``.weight`` attribute.
    """
    weight = (
        storage
        if isinstance(storage, nn.Parameter)
        else getattr(storage, "weight", None)
    )
    if weight is None:
        return
    if gain is None:
        nn.init.zeros_(weight)
    else:
        nn.init.xavier_uniform_(weight, gain=gain)


def _zero_bias(layer: nn.Module, storage: nn.Module | nn.Parameter) -> None:
    """Zero the bias on ``storage`` or, for spectral-norm layers, on ``layer``."""
    bias = getattr(storage, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)
    elif isinstance(layer, nn.Linear) and layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _wrec_init_gain(
    w_rec_init: str, w_rec_gain: float, has_spectral_norm: bool
) -> float | None:
    """Resolve the W_rec initialization gain, or ``None`` for zero init.

    ``"xavier"`` uses the configured ``w_rec_gain``. A requested ``"zero"``
    under spectral norm falls back to a small xavier so the power iteration
    never divides by zero.
    """
    if w_rec_init == "xavier":
        return w_rec_gain
    if has_spectral_norm:
        return 0.1
    return None


class EquilibriumMLP(BioModel):
    """Deep energy-contrastive EqProp MLP (Scellier & Bengio).

    Architecture:
        Layers ``W[0]: input→h_0``, ``W[1..L-1]: h_{i-1}→h_i``,
        ``W_out: h_{L-1}→out``. Depth is ``len(config.hidden_dims)`` and is
        honoured by ``_build_layers`` — so a sampled ``num_layers=N`` actually
        produces an N-hidden MLP and varies the parameter count, unlike the
        prior engine which silently truncated every probe to one hidden layer
        (phantom ``num_layers``).

    Dynamics:
        Each hidden layer relaxes jointly to equilibrium from its neighbours
        via ``settle_activations_list``. The output layer is a linear
        projection clamped by ``beta·(target − out)`` in the nudged phase.

    Training:
        ``train_step`` runs the consolidated energy-contrastive rule
        (``_contrastive_step``): free settle → nudged settle →
        ``ΔW = (post_nudge·pre_nudge.T − post_free·pre_free.T) / β`` per layer.
        No external optimizer is touched by the bio rule itself; the
        ``self.optimizer`` (a plain SGD) is created lazily and steps the
        contrastive gradients.

    Variants:
        Subclasses pick ``variant = "plain" | "momentum" | "sparse" |
        "feedback"``. ``momentum`` adds a per-layer velocity term; ``sparse``
        masks each hidden state to its top-k activations; ``feedback`` adds a
        backward path ``W_fb: out→h_i`` injected into the nudged phase. None of
        these alters the layer structure.
    """

    #: Per-step dynamics / topology variant, set by subclasses. The base class
    #: implements all four — subclasses just select.
    variant: Variant = "plain"

    def __init__(self, config: ModelConfig | None = None, **kwargs) -> None:
        cfg = config
        # Read eqprop-specific config knobs BEFORE super().__init__ because
        # BioModel.__init__ calls _build_layers() which calls _init_weights().
        if cfg is not None:
            self.w_rec_init = cfg.extra.get("w_rec_init", "zero")
            self.w_rec_gain = float(cfg.extra.get("w_rec_gain", 0.1))
            self.update_scale = float(cfg.extra.get("update_scale", 1.0))
            self.update_scale_by_depth = float(
                cfg.extra.get("update_scale_by_depth", 1.0)
            )
            self.contrastive_diagnostics = bool(
                cfg.extra.get("contrastive_diagnostics", False)
            )
            self.feedback_gain = float(cfg.extra.get("feedback_gain", 1.0))
            self.feedback_init_gain = float(cfg.extra.get("feedback_init_gain", 0.5))
        else:
            self.w_rec_init = kwargs.get("w_rec_init", "zero")
            self.w_rec_gain = float(kwargs.get("w_rec_gain", 0.1))
            self.update_scale = float(kwargs.get("update_scale", 1.0))
            self.update_scale_by_depth = float(kwargs.get("update_scale_by_depth", 1.0))
            self.contrastive_diagnostics = bool(kwargs.get("contrastive_diagnostics"))
            self.feedback_gain = float(kwargs.get("feedback_gain", 1.0))
            self.feedback_init_gain = float(kwargs.get("feedback_init_gain", 0.5))

        super().__init__(config=config, **kwargs)
        cfg = self.config
        # ``gradient_method`` selects the trainer routing: ``"contrastive"`` →
        # energy-contrastive ``train_step`` (the bio rule); ``"equilibrium"``
        # → O(1) implicit differentiation backward; ``"bptt"`` → unrolled
        # backprop through the settle (baseline). Read from ``extra`` (search
        # space passes it there) or fallback kwarg.
        gm = cfg.extra.get("gradient_method")
        self.gradient_method = (
            gm if isinstance(gm, str) else kwargs.get("gradient_method", "equilibrium")
        )
        self.lr = float(cfg.learning_rate)
        self.beta = float(cfg.beta)
        self.max_steps = int(cfg.max_steps)
        # EqProp dynamics use a bounded activation so the joint settle is
        # contractive under spectral-normed weights. The legacy ``LoopedMLP``
        # used ``tanh``; ``ModelConfig.activation`` defaults to ``silu``. Force
        # ``tanh`` here unless the caller explicitly overrode ``activation``
        # via ``config.extra``. This keeps the consolidated layered engine
        # architecturally compatible with the prior single-hidden recurrent
        # ``LoopedMLP`` at ``num_layers=1``.
        explicit_activation = cfg.extra.get("activation")
        if isinstance(explicit_activation, str):
            object.__setattr__(cfg, "activation", explicit_activation)
        elif cfg.activation == "silu":
            object.__setattr__(cfg, "activation", "tanh")
        self.activation_fn = self._get_activation(cfg.activation)
        # ``ModelConfig`` defaults slippery params first; allow ``**kwargs`` to
        # override non-field knobs that the search space samples per-rule
        # (the search space for the eqprop family defines these alongside the
        # ``ModelConfig`` knobs and the build path lands them in ``extra``).
        self.nudge_steps = int(
            kwargs.get("nudge_steps", cfg.extra.get("nudge_steps"))
            or max(3, self.max_steps // 3)
        )
        self.sparse_ratio = float(
            kwargs.get("sparse_ratio", cfg.extra.get("sparse_ratio", 0.5))
        )
        self.momentum = float(kwargs.get("momentum", cfg.extra.get("momentum", 0.5)))
        self.convergence_threshold = float(
            cfg.extra.get("convergence_threshold", cfg.convergence_threshold)
        )
        self.convergence_start = int(
            cfg.extra.get("convergence_start", cfg.convergence_start)
        )
        # Lazily-created optimizer for the contrastive step. The bio rule does
        # not strictly require an optimizer (updates are manual in the
        # contrastive step), but ``_contrastive_step`` expects one to apply the
        # accumulated gradients. A plain SGD keeps the rule transparent.
        self.optimizer: torch.optim.Optimizer | None = None
        # Transient velocity buffer (variant == "momentum").
        self._velocity: list[torch.Tensor] | None = None
        # _build_layers() already called by BioModel.__init__

    # ------------------------------------------------------------------
    # Architecture — single source of truth for the deep eqprop MLP.
    # ------------------------------------------------------------------

    def _hidden_dims(self) -> list[int]:
        """The full hidden-width list, validated.

        ``ModelConfig`` is built via ``compute_hidden_dims`` so it already
        encodes ``num_layers``; here we just surface it and ensure at least one
        hidden layer exists.
        """
        dims = list(self.config.hidden_dims)
        if not dims:
            # Defensive: a config with an empty ``hidden_dims`` (legacy direct
            # init) maps to a single hidden layer of ``self.hidden_dim`` so the
            # model still constructs. The construction supervisor separately
            # reports this as a knob-threading defect, but the constructor
            # raises instead of silently building a degenerate model.
            dim = self.hidden_dim if self.hidden_dim > 0 else 64
            dims = [dim]
        return dims

    def _build_layers(self) -> None:
        hidden = self._hidden_dims()
        dims = [self.input_dim, *hidden, self.output_dim]
        # Forward stack: input→h_0→...→h_{L-1}→out (one Linear per adjacent pair).
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        # Self-recurrent stack: one (h_i → h_i) Linear per hidden layer. Keeps
        # the prior single-hidden ``LoopedMLP`` dynamics as the ``num_layers=1``
        # case (with ``tanh`` + spectral norm the two are bit-equivalent).
        self.W_rec = nn.ModuleList([nn.Linear(h, h) for h in hidden])
        if self.config.use_spectral_norm:
            self.layers = nn.ModuleList([
                spectral_norm(layer) if i < len(self.layers) - 1 else layer
                for i, layer in enumerate(self.layers)
            ])
            # Apply spectral norm to W_rec only for the implicit equilibrium
            # path (gradient_method="equilibrium"): that path relies on it for
            # contractivity. The contrastive path initializes W_rec to zero
            # by default (w_rec_init="zero"), and spectral norm's power
            # iteration would divide by zero on a zero matrix — so skip it.
            if getattr(self, "gradient_method", "contrastive") == "equilibrium":
                self.W_rec = nn.ModuleList([spectral_norm(l) for l in self.W_rec])
        self._init_weights()
        # Feedback pathway (variant == "feedback") — one backward Linear per
        # hidden layer whose input is the output state.
        if self.variant == "feedback":
            self.feedback_layers = nn.ModuleList([
                nn.Linear(self.output_dim, h) for h in hidden
            ])
            for layer in self.feedback_layers:
                _init_weight_storage(
                    _unwrap_weight(layer), gain=self.feedback_init_gain
                )
                _zero_bias(layer, _unwrap_weight(layer))

    def _init_weights(self, modules: nn.Module | None = None) -> None:
        """Xavier-init forward (and feedback) weights, biases to zero."""
        layers = list(modules) if isinstance(modules, nn.ModuleList) else self.layers
        for layer in layers:
            _init_weight_storage(_unwrap_weight(layer), gain=0.5)
            _zero_bias(layer, _unwrap_weight(layer))
        # Self-recurrent weights: configurable init (Plan 8).
        if isinstance(modules, nn.ModuleList):
            return
        has_spectral_norm = any(
            hasattr(layer, "parametrizations")
            and hasattr(layer.parametrizations, "weight")
            for layer in self.W_rec
        )
        for layer in self.W_rec:
            gain = _wrec_init_gain(self.w_rec_init, self.w_rec_gain, has_spectral_norm)
            _init_weight_storage(_unwrap_weight(layer), gain=gain)
            _zero_bias(layer, _unwrap_weight(layer))

    # ------------------------------------------------------------------
    # Forward / settle (deep eqprop state is an *activations list*).
    # ------------------------------------------------------------------

    def _initial_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Seed per-layer activations from ``x``.

        Layers below the output are first set by a single feedforward pass, so
        the settle starts close to a real fixed point and converges quickly.
        The output layer is clamped to zero (it will be driven by the dynamics
        on the first settle step).
        """
        x = _flatten(x)
        acts: list[torch.Tensor] = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation_fn(h)
            acts.append(h)
        return acts

    def forward_dynamics(
        self,
        activations: list[torch.Tensor],
        beta: float = 0.0,
        target: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """One joint settle step for every layer state.

        ``activations[0]`` is the (fixed) input ``x``. ``activations[1..L]``
        are hidden-layer states ``h_0..h_{L-1}``. ``activations[L+1]`` is the
        output ``out``. The output is the only layer that receives a nudge
        term ``beta·(target − out)`` when ``target`` is provided; hidden layers
        settle from their forward and backward neighbours.

        Variant hooks:
            * ``momentum`` — adds a decaying velocity term per hidden layer.
            * ``sparse`` — top-k masks each hidden state post-step.
            * ``feedback`` — a constant output→hidden backward drive is added
              during the nudged phase.
        """
        x = activations[0]
        num_hidden = len(self.layers) - 1
        new_acts: list[torch.Tensor] = [x]
        # Update velocity bookkeeping only when momentum is active.
        use_momentum = self.variant == "momentum"
        if use_momentum:
            batch_size = activations[0].size(0)
            if (
                self._velocity is None
                or len(self._velocity) != num_hidden
                or self._velocity[0].size(0) != batch_size
            ):
                self._velocity = [
                    torch.zeros_like(activations[i + 1]) for i in range(num_hidden)
                ]
        for i in range(num_hidden):
            layer = self.layers[i]
            pre = layer(activations[i])  # bottom-up from previous layer
            # Self-recurrent term (LoopedMLP dynamics, generalised per layer).
            pre = pre + self.W_rec[i](activations[i + 1])
            # Top-down drive: every hidden layer reads a backward pass from the
            # state *above* it — ``activations[i + 2]``. For the last hidden
            # layer that above-state is the output ``activations[-1]`` (driven
            # by ``W_out``), so the nudged output error propagates down to the
            # last hidden unit and every hidden weight receives a contrastive
            # (free vs nudged) gradient. Without this, the last hidden layer is
            # a feedforward terminal and free ≡ nudged ⇒ zero learning signal.
            # Weight used as-is (a ``Linear(out, in)`` maps its input space to
            # its output space via ``y = x @ weight.T``, so the adjoint that
            # back-projects ``h_above`` into the current layer's space is
            # ``h_above @ weight`` — see shape derivation in the docstring).
            next_layer = self.layers[i + 1]
            w_bwd = next_layer.weight
            top_down = activations[i + 2] @ w_bwd
            total = pre + top_down
            if use_momentum:
                total = self.momentum * self._velocity[i] + total
                self._velocity[i] = total.detach().clone()
            h_new = self.activation_fn(total)
            if self.variant == "sparse":
                k = max(1, int(h_new.size(1) * self.sparse_ratio))
                vals, _ = torch.topk(torch.abs(h_new), k, dim=1)
                thr = vals[:, -1].unsqueeze(1)
                h_new = h_new * (torch.abs(h_new) >= thr).to(h_new.dtype)
            if self.variant == "feedback" and beta > 0:
                # Output→hidden feedback drive, active only during nudge.
                # ``feedback_gain`` scales the pathway (Plan 8 B3); the drive
                # is added to the hidden pre-activation, not multiplied into
                # the energy nudge term ``beta`` itself.
                fb = self.feedback_layers[i](activations[-1])
                h_new = h_new + beta * self.feedback_gain * fb
            new_acts.append(h_new)
        # Output layer: linear projection of the last hidden state.
        out = self.layers[-1](new_acts[-1])
        if beta > 0 and target is not None:
            out = out + beta * (target - out)
        new_acts.append(out)
        return new_acts

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
        """Settle the eqprop state and return the output layer.

        Two settle paths:

        * ``gradient_method="equilibrium"`` on a single-hidden model (the
          canonical "small recurrent" EqProp of Scellier & Bengio) uses the
          O(1)-memory implicit backward — memory is flat in ``max_steps`` via
          the implicit-function theorem (kept for the memory-advantage
          contract; deep multi-hidden models settle explicitly instead).
        * otherwise (deep, ``contrastive``, ``bptt``, inference) the model
          settles via :func:`settle_activations_list`; under ``no_grad``
          memory is O(L) activations, not O(L · steps).
        """
        if (
            self.gradient_method == "equilibrium"
            and len(self._hidden_dims()) == 1
            and beta <= 0
            and target is None
            and not return_trajectory
            and not return_dynamics
            and self.training
        ):
            # O(1)-memory implicit path: single-state fixed point, adjoint
            # solved iteratively by ``EquilibriumFunction`` (no unrolled graph).
            return self._implicit_forward(x, steps=steps)
        return self._explicit_forward(
            x,
            beta=beta,
            target=target,
            steps=steps,
            return_trajectory=return_trajectory,
            return_dynamics=return_dynamics,
        )

    def _implicit_forward(
        self, x: torch.Tensor, steps: int | None = None
    ) -> torch.Tensor:
        """O(1)-in-steps implicit settle for the single-hidden case."""
        from computronium.core.local_learning.settling import EquilibriumFunction

        xf = _flatten(x)
        n_steps = steps if steps is not None else self.max_steps
        params = [p for p in self.parameters() if p.requires_grad]
        h0 = torch.zeros(
            (xf.size(0), self._hidden_dims()[0]), device=xf.device, dtype=xf.dtype
        )
        x_transformed = self._transform_input(xf)
        # The implicit function stores the requested step count on the model
        # for its forward/backward loop (it reads ``model.max_steps``).
        saved_steps = self.max_steps
        self.max_steps = n_steps
        try:
            h_star = EquilibriumFunction.apply(self, x_transformed, h0, *params)
        finally:
            self.max_steps = saved_steps
        out = self._output_projection(h_star)
        self._last_activations = None
        return out

    # Single-hidden shims for the O(1) implicit ``EquilibriumFunction`` path
    # (it calls ``forward_step`` / ``_transform_input`` / ``_output_projection``
    # / ``_initialize_hidden_state``). For single-hidden eqprop these reduce to
    # the canonical "small recurrent" Scellier-Bengio dynamics
    # ``h ← tanh(W_in(x) + W_rec h)``.
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(x_transformed + self.W_rec[0](h))

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0](_flatten(x))

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.layers[-1](h)

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        xf = _flatten(x)
        return torch.zeros(
            (xf.size(0), self._hidden_dims()[0]), device=xf.device, dtype=xf.dtype
        )

    def _explicit_forward(
        self,
        x: torch.Tensor,
        *,
        beta: float,
        target: torch.Tensor | None,
        steps: int | None,
        return_trajectory: bool,
        return_dynamics: bool,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[list[torch.Tensor]]]
        | tuple[torch.Tensor, dict[str, object]]
    ):
        """Settle the (possibly deep) activations list explicitly.

        Uses settle_activations_list by default for backward compatibility.
        When return_dynamics is True, uses settle_universal for richer telemetry.
        """
        if x.dtype not in (torch.float32, torch.float64, torch.float16):
            x = x.float()

        # Use settle_universal when detailed telemetry is requested
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

        # Default path: use settle_activations_list (existing behavior)
        activations = self._initial_activations(x)
        n_steps = steps if steps is not None else self.max_steps
        settled, trajectory, dynamics = settle_activations_list(
            activations_0=activations,
            forward_dynamics=self.forward_dynamics,
            steps=n_steps,
            beta=beta,
            target=target,
            return_trajectory=return_trajectory,
            return_dynamics=return_dynamics,
            convergence_threshold=self.convergence_threshold,
            convergence_start=self.convergence_start,
        )
        self._last_activations = settled
        out = settled[-1]
        if return_dynamics:
            return out, dynamics
        if return_trajectory:
            return out, trajectory
        return out

    # ------------------------------------------------------------------
    # SettleProtocol implementation (Family B: activations list)
    # ------------------------------------------------------------------

    def _initialize_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return initial activations list for settle_universal.

        Family B pattern: state is a list of per-layer activations [x, h1, ..., out].
        """
        x = _flatten(x)
        return self._initial_activations(x)

    def _step(
        self,
        state: list[torch.Tensor],
        x_transformed: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Single settle step for settle_universal.

        Uses stored _settle_beta and _settle_target for the dynamics.
        """
        return self.forward_dynamics(
            state, beta=self._settle_beta, target=self._settle_target
        )

    def _check_converged(
        self,
        state_new: list[torch.Tensor],
        state_old: list[torch.Tensor],
        step: int,
    ) -> bool:
        """Custom convergence check matching settle_activations_list behavior.

        Convergence is based on max relative change across hidden+output layers.
        """
        if step <= self.convergence_start:
            return False

        convergence_norm = 2
        max_rel_delta = 0.0
        for k in range(1, len(state_new)):
            abs_delta = torch.dist(
                state_new[k], state_old[k], p=convergence_norm
            ).item()
            norm = state_old[k].norm(p=convergence_norm).item() + 1e-8
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
        # Telemetry is collected by settle_universal

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
        """Run settle using the universal primitive with full telemetry.

        Stores beta/target for the _step method, then calls settle_universal.
        """
        self._settle_beta = beta
        self._settle_target = target

        # Use model's convergence knobs
        config = SettleConfig(
            max_steps=steps if steps is not None else self.max_steps,
            convergence_threshold=self.convergence_threshold,
            convergence_start=self.convergence_start,
        )

        state, steps_taken, converged, telemetry = settle_universal(
            self,
            x,
            config=config,
            algorithm="eqprop",
            family="B",
            hardware=self.config.device if hasattr(self.config, "device") else "cpu",
            backend="pytorch",
            return_trajectory=return_trajectory,
        )

        # Return in the format expected by existing code
        settled_activations = state
        out = settled_activations[-1]
        self._last_activations = settled_activations

        return out, steps_taken, converged, telemetry

    def transition_modules(self) -> list[nn.Module]:
        """Forward ``Linear`` layers in order — used by propagator/audit."""
        return list(self.layers)

    # ------------------------------------------------------------------
    # Train step — consolidated energy-contrastive.
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> None:
        if self.optimizer is None:
            self.optimizer = create_optimizer(
                self, OptimizerConfig(name="sgd", lr=self.lr, weight_decay=0.0)
            )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Energy-contrastive free/nudged update (Scellier & Bengio).

        Returns a metrics dict with ``loss`` and ``accuracy``. Runs when the
        sweep activates the model with ``gradient_method="contrastive"`` or
        ``"equilibrium"`` (the trainer tries ``model.train_step`` before any
        fallback, so the local rule runs under the fast path too). Models
        without a native rule fall through to the implicit / BPTT trainer
        paths. ``"bptt"`` explicitly opts out.

        For single-hidden models with ``gradient_method="equilibrium"``, return
        ``None`` so the trainer's EnergyModel path (Phase 1) runs the O(1)-memory
        implicit equilibrium backward — the historic fast path.
        """
        if self.gradient_method not in ("equilibrium", "contrastive"):
            return None  # type: ignore[return-value]
        # Single-hidden equilibrium models use the implicit O(1)-memory path
        # via the EnergyModel protocol (trainer Phase 1).
        if self.gradient_method == "equilibrium" and len(self._hidden_dims()) == 1:
            return None  # type: ignore[return-value]
        self._ensure_optimizer()
        # For DirectedEP (variant == "feedback"), pass feedback_layers to
        # _contrastive_step so the output->hidden feedback weights are updated.
        feedback_layers = (
            list(self.feedback_layers) if self.variant == "feedback" else None
        )
        # Compute per-layer update scales (Plan 8: separate β from update scaling)
        num_layers = len(self.layers)
        update_scales = [
            self.update_scale * (self.update_scale_by_depth**i)
            for i in range(num_layers)
        ]
        return _contrastive_step(
            self,
            x,
            y,
            layer_list=list(self.layers),
            beta=self.beta,
            update_scales=update_scales,
            diagnostics=self.contrastive_diagnostics,
            use_conj=False,
            recurrent_layer_list=list(self.W_rec),
            feedback_layer_list=feedback_layers,
        )
