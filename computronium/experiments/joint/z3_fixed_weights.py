"""Z3 Fixed Weights Benchmark (Level 4).

Question: Can frozen θ solve multiple tasks via ψ?

Constraint: θ frozen. Tasks: parity, last-symbol, threshold.
Operator library: Identity, Threshold, Accumulate, LastSymbol, Parity, SparseTopKRoute, SignFlip, Delay

Gating: T_t = Σ_k g_k(ψ_t) T_k, g_k(ψ_t) = softmax(controller(ψ_t, x_t))
Differentiable: straight-through Gumbel during training, hard selection at eval

Parameter invariance MUST be exact: ||θ_after - θ_before|| == 0

Meta-training repair (2026-08-26 autopsy): all tasks share identical input
distributions, so task identity is observable only through within-episode
adaptation feedback. The recipe therefore (a) evolves ψ from selection
consequences via ``step_plasticity`` (gates + loss), (b) anneals gating
temperature with an entropy bonus, and (c) warms θ up under forced correct
operator selections before controller-only straight-through training over
per-task episodes.

Order-robustness redesign (2026-08-26, v4 registration): the v3 proportion
run proved every seed-level failure is parity-only and order-governed —
coverage was decided by whatever routing basin earlier adaptation phases
left behind, not by the frozen trunk. The registered changes:

1. Per-task optimizer rebuild: ``_adapt_all_tasks`` (and the fine-tune
   baseline) start a FRESH Adam at every task boundary. Carried
   second-moment estimates from the previous task's loss landscape were the
   starvation driver — with them, later phases cannot re-acquire their
   solver operator however much they sample (measured: the deterministic
   v3 failure order last_symbol→threshold→parity went from parity@0.48 to
   parity@1.0 with rebuild alone). Same PR-1 hygiene principle the
   meta-train→adaptation boundary already used, extended to every boundary;
   identical for both arms.
2. Adaptation entropy floor ``adapt_entropy_beta``: a small gate-entropy
   bonus during every ψ-adaptation step keeps sampling diversity alive.
   Identical for both arms.

A coded-parity variant (T_4 emitting fixed class codes instead of the label,
requiring a trained decoder) was triaged first and REVOKED pre-data: it
fixed parity-last but made every post-parity phase starve worse — the deep
exclusive-op4 basin its mixtures create erodes later tasks' acquisition.
See DECISIONS.md for the three-way triage record.

Every adaptation step also records a gate history (mean gate distribution,
hard-selection histogram, entropy) so bandit dynamics during each phase are
recoverable from artifacts without rerunning.

R8 suite-level engagement gate (2026-09-01): every ``evaluate_z3`` run embeds
its own validation — a frozen-ψ control arm (identical meta-state, task
stream, seed, and budget; only ψ stepping disabled) plus a mechanistic ψ→gate
wiring probe — and emits a ``psi_gate`` verdict over: exact θ invariance
(bitwise, was tolerance-only), ψ non-constancy, ψ task-conditioning, ψ→gate
wiring, frozen-ψ control metric sensitivity, and per-task probe accuracy
above ``TASK_CHANCE`` + margin. Instrument items must hold at any scale; the
above-chance item is the capability item and is expected to fail below
meta-training scales that acquire the decoder (quick mode's 10-epoch budget
does not — registered scale does). Per-task ψ trajectories (norm + step
delta) and final ψ vectors are recorded so the gate is auditable from
artifacts alone.

R9.1 retention pivot (2026-09-01): every run also executes the retention arm
— task A → task B → task A with θ frozen and the ψ-system (controller + rule
state) snapshot/restored between stages — upgrading Z3 from capability (ψ can
switch) to utility (switching prevents forgetting). Instrument items: exact θ
invariance across the arm, lossless ψ restore (the restored ψ-system must
reproduce stage-A accuracy exactly on the fixed probe), and ψ-state task
conditioning (ψ_A ≠ ψ_B — fails by construction on a ψ-disabled run, the
probe-the-probe property). Capability items: stage-A acquisition,
restored-ψ accuracy above chance and above the meta-trained fresh-ψ floor —
expected to fail below decoder-acquiring scales, like the engagement gate's
above-chance item. The arm snapshots/restores the RNG streams so baselines
see an identical stream with or without it (imp-56).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.core.utils.device import get_device
from computronium.experiments.joint._claims import (
    CLAIMS_SCOPE_PLUMBING_ONLY,
    CLAIMS_SCOPE_PSI_ENGAGED,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

from computronium.core.plasticity.theta_audit import (
    ThetaAuditReport,
    ThetaInvarianceAudit,
)


class Z3InstrumentError(RuntimeError):
    """Z3 suite instrument invariant violated (wiring probe, θ audit, gate)."""


# ============================================================
# Z3 Task Generators
# ============================================================


def create_parity_task(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Parity task: classify parity of number of positive elements."""
    device = get_device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    # Parity of positive elements across sequence
    pos_count = (x > 0).sum(dim=(1, 2))  # [batch]
    y = (pos_count % 2).long()
    return x, y


def create_last_symbol_task(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Last symbol task: classify by last element sign."""
    device = get_device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    last = x[:, -1, :].mean(dim=-1)  # [batch]
    y = (last > 0).long()
    return x, y


def create_threshold_task(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Threshold task: classify if sum exceeds threshold."""
    device = get_device(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    total = x.sum(dim=(1, 2))  # [batch]
    y = (total > 0).long()  # threshold at 0
    return x, y


# ============================================================
# Z3 Operators (Minimal Rule Library)
# ============================================================


class Z3Operators:
    """Z3 minimal operator library.

    T_0 = Identity
    T_1 = Threshold
    T_2 = Accumulate
    T_3 = LastSymbol
    T_4 = Parity
    T_5 = SparseTopKRoute
    T_6 = SignFlip
    T_7 = Delay
    """

    @staticmethod
    def identity(x: Tensor) -> Tensor:
        """T_0: Identity - pass through unchanged."""
        return x

    @staticmethod
    def threshold(x: Tensor, threshold: float = 0.0) -> Tensor:
        """T_1: Threshold - binary activation."""
        return (x > threshold).float()

    @staticmethod
    def accumulate(x: Tensor) -> Tensor:
        """T_2: Accumulate - cumulative sum over sequence."""
        # x: [batch, seq_len, dim] -> cumsum over seq
        return x.cumsum(dim=1)

    @staticmethod
    def last_symbol(x: Tensor) -> Tensor:
        """T_3: LastSymbol - extract last timestep."""
        # x: [batch, seq_len, dim] -> [batch, dim]
        return x[:, -1, :]

    @staticmethod
    def parity(x: Tensor) -> Tensor:
        """T_4: Parity - compute parity of positive elements."""
        # x: [batch, seq_len, dim] -> parity per batch, expanded to dim
        pos = (x > 0).float()
        parity = (pos.sum(dim=(1, 2), keepdim=True) % 2).float()  # [batch, 1, 1]
        # Expand to match input_dim by repeating
        return parity.expand(-1, 1, x.shape[-1])  # [batch, 1, dim]

    @staticmethod
    def sparse_topk_route(x: Tensor, k: int = 2) -> Tensor:
        """T_5: SparseTopKRoute - route to top-k dimensions."""
        # x: [batch, dim] -> keep only top-k
        if x.dim() == 3:
            x = x.mean(dim=1)  # [batch, dim]
        _topk_vals, topk_indices = torch.topk(x.abs(), k=min(k, x.shape[-1]), dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_indices, 1.0)
        return x * mask

    @staticmethod
    def sign_flip(x: Tensor) -> Tensor:
        """T_6: SignFlip - flip sign of negative values."""
        return x.abs()

    @staticmethod
    def delay(x: Tensor, delay: int = 1) -> Tensor:
        """T_7: Delay - shift sequence by delay steps."""
        if x.dim() == 3:
            # [batch, seq_len, dim] -> shift right
            padded = torch.cat([torch.zeros_like(x[:, :delay]), x[:, :-delay]], dim=1)
            return padded
        return x


# ============================================================
# Z3 Controller
# ============================================================


class Z3Controller(torch.nn.Module):
    """Controller for Z3 operator selection.

    Takes (ψ_t, x_t) -> operator logits g_k
    """

    def __init__(
        self,
        operator_dim: int,
        controller_hidden: int,
        num_operators: int,
    ):
        super().__init__()
        self.num_operators = num_operators
        self.operator_dim = operator_dim

        # Controller network
        self.net = torch.nn.Sequential(
            torch.nn.Linear(controller_hidden + operator_dim, controller_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(controller_hidden, num_operators),
        )

    def forward(self, psi: Tensor, x: Tensor) -> Tensor:
        """Compute operator logits.

        Args:
            psi: Plastic state [batch, controller_hidden]
            x: Input [batch, operator_dim] (or [batch, seq_len, operator_dim])

        Returns:
            operator_logits: [batch, num_operators]
        """
        # Flatten x if needed
        if x.dim() > 2:
            x = x.mean(dim=1)  # Average over sequence
        elif x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure psi matches batch size
        if psi.shape[0] != x.shape[0]:
            psi = psi[:1].expand(x.shape[0], -1)

        combined = torch.cat([psi, x], dim=-1)
        return self.net(combined)


# ============================================================
# Z3 Model
# ============================================================

# Operator index that solves each task (forced-selection warm-up map).
# Verified by linear-probe solvability (2026-08-26): the threshold label is
# sum(values) > 0, which sign-only features cannot separate — its solver is
# the value-preserving Identity operator.
TASK_OPERATOR_MAP = {"parity": 4, "last_symbol": 3, "threshold": 0}


class Z3Model(torch.nn.Module):
    """Z3 Model: Frozen θ (operator embeddings) + Adaptive ψ (controller).

    The operator embeddings are learned during meta-training and FROZEN
    during eval. The controller learns to select operators for each task.

    ψ is the global plastic state ``psi_state`` ([1, hidden], canonical
    batch-free shape, expanded at read). It evolves ONLY through
    :meth:`step_plasticity` — ``forward`` is pure, so probe/eval passes can
    never corrupt adaptation dynamics.
    """

    def __init__(
        self,
        num_operators: int = 8,
        operator_dim: int = 64,
        controller_hidden: int = 128,
        temperature: float = 1.0,
        feedback_scale: float = 0.15,
        feedback_decay: float = 0.9,
    ):
        super().__init__()
        self.num_operators = num_operators
        self.operator_dim = operator_dim
        self.temperature = temperature
        self.feedback_scale = feedback_scale
        self.feedback_decay = feedback_decay

        # Operator embeddings (θ - FROZEN during eval)
        self.operator_embeddings = torch.nn.Parameter(
            torch.randn(num_operators, operator_dim) * 0.02,
            requires_grad=True,  # Trainable during meta-training
        )

        # Controller (part of ψ - ADAPTS during eval)
        self.controller = Z3Controller(operator_dim, controller_hidden, num_operators)
        self.controller_hidden = controller_hidden

        # Fixed random projection of [gate distribution ; loss] into ψ-space.
        # ψ then linearly encodes the cumulative selection histogram weighted
        # by loss outcomes — the sufficient statistic for identifying which
        # task's reward map the episode is running under.
        self.feedback_proj = torch.nn.Linear(num_operators + 1, controller_hidden)

        # Plastic state (ψ): global task-context vector, [1, hidden] canonical.
        self.psi_state = torch.nn.Parameter(
            torch.zeros(1, controller_hidden), requires_grad=False
        )
        self._last_gates: Tensor | None = None
        self._forced_op: int | None = None

        # Operator functions
        self.operators = [
            Z3Operators.identity,
            Z3Operators.threshold,
            Z3Operators.accumulate,
            Z3Operators.last_symbol,
            Z3Operators.parity,
            Z3Operators.sparse_topk_route,
            Z3Operators.sign_flip,
            Z3Operators.delay,
        ]

    @property
    def last_gates(self) -> Tensor | None:
        """Soft gating distribution of the most recent forward pass."""
        return self._last_gates

    def reset_psi(self) -> None:
        """Zero the plastic state at episode/task boundaries."""
        self.psi_state.zero_()
        self._last_gates = None

    def force_operator(self, index: int | None) -> None:
        """Hard-select one operator for every forward (None restores control)."""
        self._forced_op = index

    @torch.no_grad()
    def step_plasticity(self, loss: Tensor) -> None:
        """Evolve ψ one adaptation step: ψ ← tanh(ψ + proj([ḡ ; loss])).

        Feeding selection consequences back makes identity observable within
        an episode — with identical input distributions across tasks, which
        operator lowers the loss is the only task signal in existence.
        No-op until a forward has produced gates.
        """
        if self._last_gates is None:
            return
        gates = self._last_gates.detach().mean(dim=0)
        features = torch.cat([
            gates,
            loss.detach().reshape(1).to(gates.device),
        ]).unsqueeze(0)
        # Decay + scale keep ψ inside tanh's responsive regime: a raw O(1)
        # projection per step rails the state within a few steps and the
        # controller loses the running feedback summary entirely.
        update = self.feedback_scale * self.feedback_proj(features)
        self.psi_state.data = torch.tanh(
            self.feedback_decay * self.psi_state.data + update
        )

    @torch.no_grad()
    def gate_entropy(self, x: Tensor) -> float:
        """Entropy of the mean gate distribution over one input batch."""
        self.forward(x)
        gates = self._last_gates
        if gates is None:
            return 0.0
        probs = gates.mean(dim=0)
        return float(-(probs * (probs + 1e-8).log()).sum())

    def freeze_theta(self) -> None:
        """Freeze operator embeddings (θ) for Z3 evaluation.

        Only freezes θ (operator_embeddings), NOT the controller (which is part of ψ).
        """
        self.operator_embeddings.requires_grad_(False)
        # Controller is part of ψ, should remain trainable for adaptation

    def unfreeze_theta(self) -> None:
        """Unfreeze for meta-training."""
        self.operator_embeddings.requires_grad_(True)
        for param in self.controller.parameters():
            param.requires_grad_(True)

    def verify_theta_frozen(self) -> bool:
        """Verify θ (operator_embeddings) is frozen."""
        return not self.operator_embeddings.requires_grad

    def get_theta_snapshot(self) -> dict[str, Tensor]:
        """Get snapshot of θ (operator embeddings)."""
        return {"operator_embeddings": self.operator_embeddings.data.clone()}

    def compute_theta_change(self, before: dict, after: dict) -> float:
        """Compute ||θ_after - θ_before||."""
        diff = (
            (after["operator_embeddings"] - before["operator_embeddings"]).norm().item()
        )
        return diff

    def forward(self, x: Tensor, is_training: bool = False) -> Tensor:
        """Forward pass with operator selection.

        Pure w.r.t. plastic state: ψ evolves only via :meth:`step_plasticity`.

        Args:
            x: Input [batch, seq_len, operator_dim]
            is_training: If True, straight-through Gumbel selection. If False,
                hard argmax selection (eval semantics).

        Returns:
            Output logits [batch, num_classes]
        """
        batch_size = x.shape[0]

        psi = self.psi_state.to(x.device).expand(batch_size, -1)
        gate_logits = self.controller(psi, x)
        self._last_gates = torch.softmax(gate_logits / self.temperature, dim=-1)

        if self._forced_op is not None:
            # Forced warm-up: constant one-hot selection; θ receives gradients
            # through the operator stack, the controller through nothing.
            operator_weights = torch.zeros_like(gate_logits)
            operator_weights[:, self._forced_op] = 1.0
        elif is_training:
            # Straight-through Gumbel-softmax: forward propagates the HARD
            # selection (matching eval semantics exactly) while gradients
            # flow through the soft distribution. Plain soft mixtures let
            # the controller solve tasks by steering the mixture — a
            # solution that evaporates under eval's argmax (2026-08-26).
            gumbels = -torch.empty_like(gate_logits).exponential_().log()
            scores = (gate_logits + gumbels) / self.temperature
            soft = torch.softmax(scores, dim=-1)
            indices = scores.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
            operator_weights = hard + soft - soft.detach()
        else:
            _, top_idx = torch.topk(gate_logits, k=1, dim=-1)
            operator_weights = torch.zeros_like(gate_logits).scatter_(-1, top_idx, 1.0)

        operator_stack = torch.stack(
            [
                _operator_feature(op(x), batch_size=batch_size, dim=self.operator_dim)
                for op in self.operators
            ],
            dim=1,
        )
        combined = (operator_weights.unsqueeze(-1) * operator_stack).sum(dim=1)
        return combined @ self.operator_embeddings.T


def _operator_feature(op_out: Tensor, *, batch_size: int, dim: int) -> Tensor:
    """Reduce one operator's raw output to a [batch, dim] feature vector."""
    if op_out.dim() == 3:
        op_out = op_out.mean(dim=1)
    elif op_out.dim() == 2 and op_out.shape[1] == 1:
        op_out = op_out.expand(-1, dim)
    if op_out.shape != (batch_size, dim):
        width = op_out.shape[1]
        if width < dim:
            padding = torch.zeros(
                batch_size, dim - width, device=op_out.device, dtype=op_out.dtype
            )
            op_out = torch.cat([op_out, padding], dim=1)
        else:
            op_out = op_out[:, :dim]
    return op_out


# ============================================================
# Z3 Evaluation
# ============================================================


def _is_theta_param(name: str, _p: torch.nn.Parameter) -> bool:
    """Select θ (operator embeddings) — the parameters that must never move."""
    return name == "operator_embeddings"


_CRITERION_ACCURACY = 0.98
_WINDOW_STEPS = 100

# R8.1 gate chance levels: exact for these tasks — balanced binary labels by
# construction — and the pre-registered above-chance margin.
TASK_CHANCE: dict[str, float] = {"parity": 0.5, "last_symbol": 0.5, "threshold": 0.5}
_GATE_ACCURACY_MARGIN = 0.1


def _windowed_criterion_step(
    curve: list[float],
    *,
    window: int = _WINDOW_STEPS,
    threshold: float = _CRITERION_ACCURACY,
) -> int | None:
    """First 1-indexed step whose trailing ``window`` mean meets ``threshold``.

    The registered RESEARCH3 Z3 definition (configs/preregistrations/
    z3_psi_vs_finetune_steps.json). ``None`` when the budget ends first —
    censored at the budget by the analysis harness.
    """
    if len(curve) < window:
        return None
    total = sum(curve[:window])
    for step in range(window, len(curve) + 1):
        if total / window >= threshold:
            return step
        if step < len(curve):
            total += curve[step] - curve[step - window]
    return None


def _apply_task_order(tasks: list, task_order: Sequence[str] | None) -> None:
    """Reorder the (name, task_fn) switching stream in place to ``task_order``.

    Identity when ``task_order`` is None. All consumers downstream of the
    call site (adaptation, baselines, diversity) follow the realized order,
    so both treatment and control arms see the identical stream per seed.
    """
    if task_order is None:
        return
    known = [name for name, _ in tasks]
    if sorted(task_order) != sorted(known):
        raise ValueError(f"task_order must be a permutation of {known}")
    rank = {name: i for i, name in enumerate(task_order)}
    tasks.sort(key=lambda pair: rank[pair[0]])


@dataclass(frozen=True, slots=True)
class TaskShape:
    """Batch geometry shared by every task-protocol helper."""

    batch_size: int
    seq_len: int
    input_dim: int
    device: torch.device

    def sample(self, task_fn) -> tuple[Tensor, Tensor]:
        return task_fn(self.batch_size, self.seq_len, self.input_dim, self.device)


@dataclass(frozen=True, slots=True)
class MetaRecipe:
    """Meta-training recipe implementing the E-2 repair attacks (a)–(c).

    (a) ``feedback`` + ``episode_len``: ψ evolves from selection consequences
    (gates + loss) over consecutive same-task episodes — the only task signal
    that exists when all tasks share identical input distributions.
    (b) ``temp_start``→``temp_end`` linear anneal + gate-entropy bonus,
    optionally curriculumed ``entropy_beta``→``entropy_end`` so routing
    locks instead of merely exploring.
    (c) ``warmup_fraction`` of epochs under forced correct-operator
    selection trains θ first; the controller phase then runs θ-frozen.
    ``replay_steps`` adds per-epoch supervised distillation passes over a
    FIFO buffer of episode trajectories (ψ, input summary → episode-best
    operator), sharpening the policy without Gumbel-noise gradients.
    ``adapt_entropy_beta`` is NOT a meta-training knob: it sets the
    gate-entropy bonus applied during every ψ-adaptation step at eval time
    (order-robustness redesign — keeps entrenched routing reachable to
    unsampled operators; identical for both arms). ``adapt_temp_end``
    linearly anneals the gating temperature from its phase-entry value down
    to this target ACROSS each adaptation phase (None = flat): discovery
    happens hot, locking happens cold — without it, open-ended races at
    constant hot temperature both delay solver discovery past any fixed
    budget and leave the criterion window unfilled after late discoveries.
    """

    episode_len: int = 8
    feedback: bool = True
    entropy_beta: float = 0.05
    temp_start: float = 2.0
    temp_end: float = 0.5
    warmup_fraction: float = 0.4
    warmup_lr: float = 3e-3
    adapt_temp: float | None = None
    adapt_temp_end: float | None = None
    entropy_end: float | None = None
    replay_steps: int = 0
    adapt_entropy_beta: float = 0.0


@dataclass(frozen=True, slots=True)
class _AdaptationRecord:
    """Per-phase adaptation outputs shared by every adaptation arm."""

    losses: list[float]
    accuracy_curve: list[float]
    steps_to_criterion: int | None
    gate_history: dict[str, list[list[float]]]
    psi_history: dict[str, list[float]]


def _eval_task_accuracy(
    model: Z3Model,
    shape: TaskShape,
    task_fn,
    *,
    batches: int = 20,
    soft: bool = False,
) -> float:
    """Accuracy over fresh batches; ``soft=True`` keeps the differentiable mixture."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = shape.sample(task_fn)
            logits = model(x, is_training=soft)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.shape[0]
    return correct / total


def _fixed_probe(shape: TaskShape, task_fn, *, batches: int) -> tuple[Tensor, Tensor]:
    """Fixed held-out probe set scored every adaptation step.

    Generated once per task before adaptation so the registered window
    metric is deterministic and disjoint from the fresh training stream.
    """
    samples = [shape.sample(task_fn) for _ in range(batches)]
    return (
        torch.cat([x for x, _ in samples]),
        torch.cat([y for _, y in samples]),
    )


def _probe_accuracy(
    model: Z3Model, probe: tuple[Tensor, Tensor], *, chunk_size: int = 4096
) -> float:
    """Hard-selection accuracy over the whole probe set, batch-chunked.

    ψ is a global [1, hidden] state and ``forward`` is pure, so any chunk
    size is safe — probe passes can no longer corrupt adaptation dynamics.
    """
    x, y = probe
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            logits = model(x[start : start + chunk_size])
            target = y[start : start + chunk_size]
            correct += (logits.argmax(dim=-1) == target).sum().item()
    return correct / x.shape[0]


def _gate_record(model: Z3Model) -> dict[str, list[float]]:
    """Bandit-state snapshot of the most recent forward pass.

    Mean soft-gate distribution, hard-selection histogram, and gate entropy —
    the per-step record that makes operator-routing dynamics during each
    adaptation phase recoverable from artifacts.
    """
    gates = model.last_gates
    if gates is None:
        return {"mean_gates": [], "hard_op_fraction": [], "entropy": []}
    with torch.no_grad():
        probs = gates.detach().mean(dim=0)
        hard = gates.detach().argmax(dim=-1)
        fraction = (
            torch.bincount(hard, minlength=model.num_operators).float() / hard.shape[0]
        )
        entropy = float(-(probs * (probs + 1e-8).log()).sum())
    return {
        "mean_gates": [float(v) for v in probs],
        "hard_op_fraction": [float(v) for v in fraction],
        "entropy": [entropy],
    }


def _new_gate_history() -> dict[str, list[list[float]]]:
    return {"mean_gates": [], "hard_op_fraction": [], "entropy": []}


def _record_gate_step(history: dict[str, list[list[float]]], model: Z3Model) -> None:
    record = _gate_record(model)
    for key, values in record.items():
        history[key].append(values)


def _adaptation_objective(
    loss: Tensor, model: Z3Model, adapt_entropy_beta: float
) -> Tensor:
    """CE plus the registered gate-entropy exploration floor (β·H term)."""
    if adapt_entropy_beta <= 0:
        return loss
    gates = model.last_gates
    if gates is None:
        return loss
    entropy = -(gates * (gates + 1e-8).log()).sum(-1).mean()
    return loss - adapt_entropy_beta * entropy


def _run_adaptation(  # ruff: ignore[too-many-arguments]
    model: Z3Model,
    optimizer: torch.optim.Optimizer,
    criterion,
    shape: TaskShape,
    task_fn,
    *,
    epochs: int,
    probe: tuple[Tensor, Tensor],
    feedback: bool = True,
    adapt_entropy_beta: float = 0.0,
    adapt_temp_end: float | None = None,
) -> _AdaptationRecord:
    """Adam steps over whatever ``requires_grad`` currently selects.

    One epoch = one fresh-batch gradient step; with ``feedback`` each step
    also evolves ψ from its selection consequences. The objective is CE minus
    ``adapt_entropy_beta``·H(gates) — the registered exploration floor that
    keeps entrenched routing reachable to unsampled operators. When
    ``adapt_temp_end`` is set, the gating temperature anneals linearly from
    its phase-entry value to that target across the phase. Returns an
    :class:`_AdaptationRecord` with per-step losses, the per-step
    hard-selection probe-accuracy curve, the registered 100-step-window
    criterion step (None when censored at the budget), the per-step gate
    history, and the per-step ψ trajectory (norm + step delta).
    """
    losses: list[float] = []
    curve: list[float] = []
    gate_history = _new_gate_history()
    psi_history: dict[str, list[float]] = {"norm": [], "delta_norm": []}
    prev_psi = model.psi_state.detach().clone()
    start_temp = model.temperature
    for _epoch in range(epochs):
        if adapt_temp_end is not None:
            progress = len(losses) / max(1, epochs - 1)
            model.temperature = start_temp + (adapt_temp_end - start_temp) * progress
        model.train()
        x, y = shape.sample(task_fn)
        optimizer.zero_grad()
        logits = model(x, is_training=True)
        loss = criterion(logits, y)
        _adaptation_objective(loss, model, adapt_entropy_beta).backward()
        optimizer.step()
        if feedback:
            model.step_plasticity(loss)
        psi = model.psi_state.detach()
        psi_history["norm"].append(float(psi.norm()))
        psi_history["delta_norm"].append(float((psi - prev_psi).norm()))
        prev_psi = psi.clone()
        losses.append(loss.item())
        curve.append(_probe_accuracy(model, probe))
        _record_gate_step(gate_history, model)
    return _AdaptationRecord(
        losses=losses,
        accuracy_curve=curve,
        steps_to_criterion=_windowed_criterion_step(curve),
        gate_history=gate_history,
        psi_history=psi_history,
    )


_REPLAY_BUFFER_CAP = 1024


def _replay_pass(
    model: Z3Model,
    optimizer: torch.optim.Optimizer,
    buffer: deque[tuple[Tensor, Tensor, int]],
    *,
    steps: int,
    batch_size: int = 64,
) -> None:
    """Distill episode-best operator labels into the controller (attack c).

    Supervised CE over frozen trajectories — no Gumbel noise — sharpens the
    ψ-history → operator mapping that straight-through sampling only ever
    estimates noisily. No-op on an empty buffer.
    """
    if not buffer or steps <= 0:
        return
    ce = torch.nn.CrossEntropyLoss()
    rows = list(buffer)
    for _ in range(steps):
        sample = random.sample(rows, min(batch_size, len(rows)))
        psi = torch.stack([r[0] for r in sample])
        x_feat = torch.stack([r[1] for r in sample])
        labels = torch.tensor([r[2] for r in sample], device=psi.device)
        optimizer.zero_grad()
        ce(model.controller(psi, x_feat), labels).backward()
        optimizer.step()


def _forced_episode(
    model: Z3Model,
    optimizer: torch.optim.Optimizer,
    criterion,
    shape: TaskShape,
    task_fn,
    *,
    recipe: MetaRecipe,
) -> float:
    """One forced-selection warm-up episode (θ receives all gradients)."""
    model.reset_psi()
    episode_loss = 0.0
    for _step in range(recipe.episode_len):
        x, y = shape.sample(task_fn)
        optimizer.zero_grad()
        task_loss = criterion(model(x, is_training=True), y)
        task_loss.backward()
        optimizer.step()
        episode_loss += task_loss.item()
    return episode_loss / recipe.episode_len


def _controller_episode(
    model: Z3Model,
    optimizer: torch.optim.Optimizer,
    criterion,
    shape: TaskShape,
    task_fn,
    *,
    recipe: MetaRecipe,
    beta: float,
) -> tuple[float, list[tuple[Tensor, Tensor, int, float]]]:
    """One straight-through selection episode.

    Returns the mean objective loss plus per-step replay transitions
    (ψ snapshot, batch-mean input feature, majority hard operator, loss).
    """
    model.reset_psi()
    transitions: list[tuple[Tensor, Tensor, int, float]] = []
    episode_loss = 0.0
    for _step in range(recipe.episode_len):
        x, y = shape.sample(task_fn)
        psi_before = model.psi_state.detach()[0]
        optimizer.zero_grad()
        logits = model(x, is_training=True)
        task_loss = criterion(logits, y)
        gates = model.last_gates
        objective = task_loss
        if gates is not None and beta > 0:
            entropy = -(gates * (gates + 1e-8).log()).sum(-1).mean()
            objective = task_loss - beta * entropy
        objective.backward()
        optimizer.step()
        episode_loss += objective.item()
        if gates is not None:
            hard_op = int(gates.argmax(dim=-1).mode().values.item())
            transitions.append((
                psi_before,
                x.mean(dim=(0, 1)).detach(),
                hard_op,
                task_loss.item(),
            ))
            if recipe.feedback:
                model.step_plasticity(task_loss)
    return episode_loss / recipe.episode_len, transitions


def _episode_best_op(
    transitions: list[tuple[Tensor, Tensor, int, float]],
) -> int | None:
    """Operator with the lowest mean observed loss within one episode."""
    per_op: dict[int, list[float]] = {}
    for _psi, _x_feat, op, loss in transitions:
        per_op.setdefault(op, []).append(loss)
    if not per_op:
        return None
    return min(per_op, key=lambda op: sum(per_op[op]) / len(per_op[op]))


def _meta_train(
    model: Z3Model,
    optimizer: torch.optim.Optimizer,
    tasks,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
    recipe: MetaRecipe,
    forced_ops: dict[str, int] | None = None,
) -> dict[str, list[float]]:
    """Episode-structured meta-training.

    Each episode runs ``recipe.episode_len`` consecutive batches of ONE task
    with ψ reset at the boundary — mirroring the eval switching stream,
    where identity must be acquired from within-episode feedback alone. When
    ``forced_ops`` is set, selections are pinned to the per-task solving
    operator (θ warm-up phase; controller receives no gradient). Otherwise
    straight-through Gumbel selection trains whatever ``requires_grad``
    currently selects under the recipe's temperature anneal and entropy
    bonus (linearly annealed ``entropy_beta``→``entropy_end`` when a
    curriculum is set). ψ evolves only when ``recipe.feedback`` is on AND
    selections are not forced. With ``replay_steps`` > 0 each epoch ends by
    distilling episode-best-operator labels from a FIFO trajectory buffer
    into the controller.

    Returns per-epoch mean episode loss per task.
    """
    model.train()
    total_episodes = max(epochs * len(tasks), 1)
    losses_by_task: dict[str, list[float]] = {name: [] for name, _fn in tasks}
    replay_buffer: deque[tuple[Tensor, Tensor, int]] = deque(maxlen=_REPLAY_BUFFER_CAP)
    episode = 0
    for epoch in range(epochs):
        progress = episode / total_episodes
        model.temperature = recipe.temp_end + (recipe.temp_start - recipe.temp_end) * (
            1 - progress
        )
        beta = recipe.entropy_beta
        if recipe.entropy_end is not None:
            beta += (recipe.entropy_end - recipe.entropy_beta) * progress
        for task_name, task_fn in tasks:
            if forced_ops is not None:
                model.force_operator(forced_ops[task_name])
                mean_loss = _forced_episode(
                    model, optimizer, criterion, shape, task_fn, recipe=recipe
                )
            else:
                model.force_operator(None)
                mean_loss, transitions = _controller_episode(
                    model,
                    optimizer,
                    criterion,
                    shape,
                    task_fn,
                    recipe=recipe,
                    beta=beta,
                )
                best_op = _episode_best_op(transitions)
                if best_op is not None:
                    replay_buffer.extend(
                        (psi, x_feat, best_op)
                        for psi, x_feat, _op, _loss in transitions
                    )
            losses_by_task[task_name].append(mean_loss)
            episode += 1
        if forced_ops is None:
            _replay_pass(model, optimizer, replay_buffer, steps=recipe.replay_steps)
        if epoch % 10 == 0:
            mean_total = sum(v[-1] for v in losses_by_task.values()) / len(tasks)
            print(f"    Epoch {epoch}: loss={mean_total:.4f} T={model.temperature:.2f}")
    model.force_operator(None)
    return losses_by_task


def _snapshot(model: Z3Model) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _reinit_psi(model: Z3Model) -> None:
    """Reset the controller to a fresh random init and zero plastic buffers."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
    model.reset_psi()


def _adapt_all_tasks(
    model: Z3Model,
    tasks,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
    probe_batches: int = 16,
    feedback: bool = True,
    adapt_entropy_beta: float = 0.0,
    adapt_temp_end: float | None = None,
) -> tuple[dict[str, dict], float]:
    """ψ-only adaptation protocol over the switching stream (θ stays frozen).

    A FRESH Adam is built at every task boundary: second-moment estimates
    carried from the previous task's loss landscape starve later phases of
    effective steps, which is the mechanism behind the v3 order-governed
    coverage failures (PR-1 hygiene applied at every phase boundary).
    ψ resets at each task boundary too. Returns per-task result rows
    (incl. per-step gate histories) and elapsed wall-clock.

    When ``adapt_temp_end`` is set, the gating temperature anneals linearly
    from its phase-entry value to that target across the phase.
    """
    started = time.perf_counter()
    rows: dict[str, dict] = {}
    for task_name, task_fn in tasks:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        model.reset_psi()
        probe = _fixed_probe(shape, task_fn, batches=probe_batches)
        pre_adapt = _probe_accuracy(model, probe)
        record = _run_adaptation(
            model,
            optimizer,
            criterion,
            shape,
            task_fn,
            epochs=epochs,
            probe=probe,
            feedback=feedback,
            adapt_entropy_beta=adapt_entropy_beta,
            adapt_temp_end=adapt_temp_end,
        )
        rows[task_name] = {
            "accuracy": _eval_task_accuracy(model, shape, task_fn),
            "soft_eval_accuracy": _eval_task_accuracy(model, shape, task_fn, soft=True),
            "pre_adapt_accuracy": pre_adapt,
            "adaptation_losses": record.losses,
            "accuracy_curve": record.accuracy_curve,
            "steps_to_criterion": record.steps_to_criterion,
            "gate_history": record.gate_history,
            "psi_history": record.psi_history,
            "final_psi": [float(v) for v in model.psi_state.detach().squeeze(0)],
        }
    return rows, time.perf_counter() - started


def _finetune_forgetting_baseline(
    model: Z3Model,
    tasks,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
    probe_batches: int = 16,
    feedback: bool = True,
    adapt_entropy_beta: float = 0.0,
    adapt_temp_end: float | None = None,
) -> dict:
    """Baseline (a): sequential θ fine-tuning at the same per-task step budget.

    Produces the stage×task accuracy matrix whose diagonal-vs-last-column
    gap is the forgetting tax Z3 claims to avoid. Stages are scored by the
    same registered window definition as the ψ-only arm, and each stage
    starts a fresh Adam exactly like the treatment arm (identical protocol).
    """
    model.unfreeze_theta()
    matrix: dict[str, dict[str, float]] = {}
    steps_by_stage: dict[str, int | None] = {}
    pre_adapt_by_stage: dict[str, float] = {}
    curves_by_stage: dict[str, list[float]] = {}
    gate_histories_by_stage: dict[str, dict[str, list[list[float]]]] = {}
    started = time.perf_counter()
    for stage_name, stage_fn in tasks:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.reset_psi()
        probe = _fixed_probe(shape, stage_fn, batches=probe_batches)
        pre_adapt = _probe_accuracy(model, probe)
        pre_adapt_by_stage[stage_name] = pre_adapt
        record = _run_adaptation(
            model,
            optimizer,
            criterion,
            shape,
            stage_fn,
            epochs=epochs,
            probe=probe,
            feedback=feedback,
            adapt_entropy_beta=adapt_entropy_beta,
            adapt_temp_end=adapt_temp_end,
        )
        steps_by_stage[stage_name] = record.steps_to_criterion
        curves_by_stage[stage_name] = record.accuracy_curve
        gate_histories_by_stage[stage_name] = record.gate_history
        matrix[stage_name] = {
            name: _eval_task_accuracy(model, shape, fn) for name, fn in tasks
        }
    elapsed = time.perf_counter() - started

    final_row = matrix[tasks[-1][0]]
    forgetting = {
        name: matrix[name][name] - final_row[name] for name, _fn in tasks[:-1]
    }
    return {
        "accuracy_matrix": matrix,
        "forgetting": forgetting,
        "final_accuracy": final_row,
        "steps_to_criterion": steps_by_stage,
        "pre_adapt_accuracy": pre_adapt_by_stage,
        "accuracy_curves": curves_by_stage,
        "gate_histories": gate_histories_by_stage,
        "wall_clock_s": elapsed,
    }


def _run_baselines(  # ruff: ignore[too-many-arguments]
    model: Z3Model,
    meta_state: dict[str, Tensor],
    tasks,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
    probe_batches: int = 16,
    feedback: bool = True,
    adapt_entropy_beta: float = 0.0,
    adapt_temp_end: float | None = None,
) -> dict:
    """E-10 control set; arms restore the meta-trained state first.

    Order matters because the fine-tune arm unfreezes θ and must run last.
    """
    # (c) floor control: meta-trained trunk, no ψ adaptation at all
    model.load_state_dict(meta_state)
    model.freeze_theta()
    floor_tasks = {
        name: {"accuracy": _eval_task_accuracy(model, shape, fn)} for name, fn in tasks
    }

    # (b) random-ψ init: isolates what meta-training bought the controller
    model.load_state_dict(meta_state)
    model.freeze_theta()
    _reinit_psi(model)
    random_psi_rows, _ = _adapt_all_tasks(
        model,
        tasks,
        criterion,
        shape,
        epochs=epochs,
        probe_batches=probe_batches,
        feedback=feedback,
        adapt_entropy_beta=adapt_entropy_beta,
        adapt_temp_end=adapt_temp_end,
    )

    # (a) fine-tune θ, same step budget — the forgetting tax
    model.load_state_dict(meta_state)
    finetune = _finetune_forgetting_baseline(
        model,
        tasks,
        criterion,
        shape,
        epochs=epochs,
        probe_batches=probe_batches,
        feedback=feedback,
        adapt_entropy_beta=adapt_entropy_beta,
        adapt_temp_end=adapt_temp_end,
    )

    return {
        "frozen_floor": {"tasks": floor_tasks},
        "random_psi": {
            "tasks": {
                n: {
                    "accuracy": r["accuracy"],
                    "steps_to_criterion": r["steps_to_criterion"],
                    "pre_adapt_accuracy": r["pre_adapt_accuracy"],
                    "accuracy_curve": r["accuracy_curve"],
                    "gate_history": r["gate_history"],
                }
                for n, r in random_psi_rows.items()
            }
        },
        "finetune_forgetting": finetune,
    }


def _operator_diversity(
    model: Z3Model,
    tasks,
    criterion,
    shape: TaskShape,
    *,
    steps: int = 5,
    feedback: bool = True,
) -> float:
    """Mean gate entropy after a brief per-task ψ adaptation.

    Probes whether the controller routes DIFFERENT tasks to DIFFERENT
    operators once adaptation history has accumulated (collapse detection:
    H < log 2 flags single-operator reliance).
    """
    model.eval()
    entropies: list[float] = []
    with torch.no_grad():
        for _task_name, task_fn in tasks:
            model.reset_psi()
            for _ in range(steps):
                x, y = shape.sample(task_fn)
                loss = criterion(model(x, is_training=True), y)
                if feedback:
                    model.step_plasticity(loss)
            x, _y = shape.sample(task_fn)
            entropies.append(model.gate_entropy(x))
    return sum(entropies) / len(entropies)


def _meta_train_phases(
    model: Z3Model,
    tasks,
    criterion,
    shape: TaskShape,
    *,
    recipe: MetaRecipe,
    meta_train_epochs: int,
) -> None:
    """Two-phase meta-training: forced-operator θ warm-up, then selection."""
    print("  Meta-training phase...")
    warmup_epochs = round(meta_train_epochs * recipe.warmup_fraction)
    if warmup_epochs:
        # Phase 1: θ warm-up under forced correct-operator selections so the
        # decoder is meaningful before selection has any signal to find.
        model.unfreeze_theta()
        warmup_optimizer = torch.optim.Adam(model.parameters(), lr=recipe.warmup_lr)
        _meta_train(
            model,
            warmup_optimizer,
            tasks,
            criterion,
            shape,
            epochs=warmup_epochs,
            recipe=MetaRecipe(
                episode_len=recipe.episode_len,
                temp_start=recipe.temp_start,
                temp_end=recipe.temp_start,
            ),
            forced_ops=TASK_OPERATOR_MAP,
        )
        print(f"    Forced-selection θ warm-up done ({warmup_epochs} epochs)")

    # Phase 2: selection training. Two-phase recipe freezes θ here; without
    # a warm-up phase θ trains jointly with the controller.
    controller_epochs = meta_train_epochs - warmup_epochs
    if controller_epochs > 0:
        if warmup_epochs:
            model.freeze_theta()
        else:
            model.unfreeze_theta()
        # Rebuild Adam between phases: no stale momentum crosses the boundary.
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        _meta_train(
            model,
            optimizer,
            tasks,
            criterion,
            shape,
            epochs=controller_epochs,
            recipe=recipe,
        )


def _snapshot_rng(device: torch.device) -> dict[str, Tensor]:
    """Capture the global RNG streams consumed by the adaptation protocol."""
    states: dict[str, Tensor] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    return states


def _restore_rng(states: dict[str, Tensor], device: torch.device) -> None:
    """Replay the captured streams so a control arm sees identical batches."""
    torch.set_rng_state(states["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(states["cuda"], device)


def _gates_respond_to_psi(model: Z3Model, shape: TaskShape, *, seed: int) -> float:
    """Max |Δgate| between ψ=0 and a seeded nonzero ψ over one identical batch.

    Mechanistic wiring probe for the ψ→selection causal link: ψ is a
    controller input, so any nonzero weight path must move the gate
    distribution. Uses an isolated generator so the global RNG stream — and
    therefore the entire downstream run — is untouched; eval forward passes
    draw no randomness.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape.batch_size, shape.seq_len, shape.input_dim, generator=gen).to(
        shape.device
    )
    model.eval()
    with torch.no_grad():
        model.reset_psi()
        model(x)
        gates_zero = model.last_gates
        model.psi_state.data = torch.randn(
            1, model.controller_hidden, generator=gen
        ).to(shape.device)
        model(x)
        gates_psi = model.last_gates
    model.reset_psi()
    if gates_zero is None or gates_psi is None:
        msg = "forward did not populate gates"
        raise Z3InstrumentError(msg)
    return float((gates_psi - gates_zero).abs().max())


def _adapt_under_audit(
    model: Z3Model,
    tasks: list,
    criterion,
    shape: TaskShape,
    *,
    epochs: int,
    probe_batches: int,
    recipe: MetaRecipe,
    feedback: bool | None = None,
) -> tuple[dict, float, ThetaAuditReport]:
    """ψ-adaptation over the switching stream under the exact θ audit.

    Shared by the treatment and frozen-ψ control arms. Returns per-task
    rows, elapsed wall-clock, and the audit report; a missing report is a
    RuntimeError, never a silent pass. ``feedback=None`` defers to
    ``recipe.feedback``.
    """
    with ThetaInvarianceAudit(model, selector=_is_theta_param) as audit:
        rows, wall = _adapt_all_tasks(
            model,
            tasks,
            criterion,
            shape,
            epochs=epochs,
            probe_batches=probe_batches,
            feedback=recipe.feedback if feedback is None else feedback,
            adapt_entropy_beta=recipe.adapt_entropy_beta,
            adapt_temp_end=recipe.adapt_temp_end,
        )
    report = audit.report
    if report is None:
        msg = "θ audit produced no report"
        raise Z3InstrumentError(msg)
    return rows, wall, report


def _psi_gate(
    task_rows: dict[str, dict],
    control_rows: dict[str, dict],
    gate_response: float,
    *,
    theta_invariant: bool,
    control_theta_change: float,
) -> dict:
    """R8.1 suite-level engagement gate over the emitted per-task records.

    Instrument items must hold at any scale; ``probe_above_chance`` is the
    capability item (per task, against ``TASK_CHANCE`` + margin) and is
    expected to fail below meta-training scales that acquire the decoder.
    The frozen-ψ control arm shares θ, task stream, seed, and budget with the
    engaged arm — only ψ stepping differs (non-confounded by construction).
    """
    psi_moved = any(
        max(row["psi_history"]["delta_norm"], default=0.0) > 0.0
        for row in task_rows.values()
    )
    finals = [torch.tensor(row["final_psi"]) for row in task_rows.values()]
    pairwise = [
        float((a - b).abs().max())
        for i, a in enumerate(finals)
        for b in finals[i + 1 :]
    ]
    above = {
        name: bool(row["accuracy_curve"])
        and row["accuracy_curve"][-1] > TASK_CHANCE[name] + _GATE_ACCURACY_MARGIN
        for name, row in task_rows.items()
    }
    gaps = {
        name: [task_rows[name]["accuracy"], control_rows[name]["accuracy"]]
        for name in task_rows
    }
    items: dict[str, object] = {
        "theta_exact_invariant": bool(theta_invariant),
        "psi_non_constant": psi_moved,
        "psi_task_conditioned": any(delta > 0.0 for delta in pairwise),
        "gates_respond_to_psi": gate_response > 0.0,
        "frozen_psi_control_changes_metrics": any(
            engaged != frozen for engaged, frozen in gaps.values()
        ),
        "probe_above_chance": above,
    }
    flat = {k: v for k, v in items.items() if isinstance(v, bool)}
    flat["probe_above_chance"] = all(above.values())
    failed = [k for k, v in flat.items() if not v]
    return {
        "chance": dict(TASK_CHANCE),
        "margin": _GATE_ACCURACY_MARGIN,
        "items": items,
        "detail": {
            "gate_response_max_delta": gate_response,
            "min_pairwise_final_psi_delta": min(pairwise, default=0.0),
            "engaged_vs_frozen_accuracy": gaps,
            "control_theta_change": control_theta_change,
        },
        "passed": not failed,
        "failed": failed,
    }


def _run_psi_gate_arm(  # ruff: ignore[too-many-arguments]
    model: Z3Model,
    meta_state: dict[str, Tensor],
    tasks: list,
    task_rows: dict[str, dict],
    shape: TaskShape,
    *,
    epochs: int,
    probe_batches: int,
    recipe: MetaRecipe,
    entry_temperature: float,
    rng_snapshot: dict[str, Tensor],
    gate_response: float,
    theta_invariant: bool,
) -> tuple[dict, dict]:
    """Run the embedded frozen-ψ control arm and assemble the R8.1 gate.

    Planted-ψ control: identical meta-state, task stream, seed, budget, and
    protocol temperature — only ψ stepping is disabled. The RNG streams are
    replayed from the treatment arm's entry point so both arms see
    bit-identical batches and probes: any metric difference is a causal ψ
    effect, and on a ψ-disabled run the arms are bit-identical (the gate
    item must then be False — the probe-the-probe property). Returns the
    ``psi_control`` record and the ``psi_gate`` verdict.
    """
    model.load_state_dict(meta_state)
    model.freeze_theta()
    model.temperature = entry_temperature
    _restore_rng(rng_snapshot, shape.device)
    control_rows, _wall, control_report = _adapt_under_audit(
        model,
        tasks,
        torch.nn.CrossEntropyLoss(),
        shape,
        epochs=epochs,
        probe_batches=probe_batches,
        recipe=recipe,
        feedback=False,
    )
    psi_control = {
        "feedback": False,
        "theta_change": control_report.max_abs_change,
        "tasks": {
            name: {
                "accuracy": row["accuracy"],
                "pre_adapt_accuracy": row["pre_adapt_accuracy"],
                "steps_to_criterion": row["steps_to_criterion"],
                "accuracy_curve": row["accuracy_curve"],
            }
            for name, row in control_rows.items()
        },
    }
    gate = _psi_gate(
        task_rows,
        control_rows,
        gate_response,
        theta_invariant=theta_invariant,
        control_theta_change=control_report.max_abs_change,
    )
    if gate["passed"]:
        print("  ψ gate: PASS")
    else:
        print(f"  ψ gate: FAIL — {', '.join(gate['failed'])}")
    return psi_control, gate


_RETENTION_STAGES = 2  # A→B: exactly two adaptation stages before the restore


def _psi_system_state(model: Z3Model) -> dict[str, Tensor]:
    """Snapshot the trainable ψ-system: controller weights + rule state.

    θ (``operator_embeddings``) is excluded — the ψ-system is everything the
    adaptation protocol may move (the freeze selects θ only, and the
    controller is part of ψ: its weights carry the per-task routing learned
    during adaptation). A restore without the controller would leave the
    routing behind and silently break the lossless-restore invariant.
    """
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k != "operator_embeddings"
    }


def _restore_psi_system(model: Z3Model, snapshot: dict[str, Tensor]) -> None:
    """Restore a ψ-system snapshot (controller + ψ) over the current state."""
    model.load_state_dict({**model.state_dict(), **snapshot})


def _run_retention_arm(  # ruff: ignore[too-many-locals]
    model: Z3Model,
    meta_state: dict[str, Tensor],
    tasks: list,
    shape: TaskShape,
    *,
    epochs: int,
    probe_batches: int,
    recipe: MetaRecipe,
    entry_temperature: float,
) -> dict[str, dict]:
    """R9.1 retention pivot: A→B→A with θ frozen and ψ carrying the task.

    Upgrades Z3 from capability (ψ can switch) to utility (switching
    prevents forgetting). Stage A adapts the ψ-system to the first task and
    snapshots it; stage B adapts a fresh ψ-system to the second; the restore
    stage switches ψ back with NO re-adaptation. The fixed probe sets make
    every retention readout score identical batches, so a restored ψ-system
    must reproduce stage-A accuracy exactly — a lossless snapshot/restore
    instrument item that holds at any scale. The meta-trained ψ-system with
    a fresh ψ is the floor, and the B-adapted ψ-system is the forgetting
    contrast; θ must remain bitwise frozen across the whole arm. RNG streams
    are snapshotted/restored around the arm so downstream baselines see the
    same stream with or without it (imp-56 hygiene). Returns
    ``{"retention": ..., "retention_gate": ...}`` for direct
    ``results.update()`` merging.
    """
    if len(tasks) < _RETENTION_STAGES:
        msg = "retention arm needs at least two tasks"
        raise Z3InstrumentError(msg)
    (name_a, fn_a), (name_b, fn_b) = tasks[0], tasks[1]
    criterion = torch.nn.CrossEntropyLoss()
    rng = _snapshot_rng(shape.device)
    try:
        model.load_state_dict(meta_state)
        model.freeze_theta()
        model.temperature = entry_temperature
        meta_psi_system = _psi_system_state(model)
        probe_a = _fixed_probe(shape, fn_a, batches=probe_batches)
        probe_b = _fixed_probe(shape, fn_b, batches=probe_batches)

        with ThetaInvarianceAudit(model, selector=_is_theta_param) as audit:
            # Stage A: acquire the probed task; snapshot the carrying ψ.
            model.reset_psi()
            optimizer_a = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            record_a = _run_adaptation(
                model,
                optimizer_a,
                criterion,
                shape,
                fn_a,
                epochs=epochs,
                probe=probe_a,
                feedback=recipe.feedback,
                adapt_entropy_beta=recipe.adapt_entropy_beta,
                adapt_temp_end=recipe.adapt_temp_end,
            )
            a_post = _probe_accuracy(model, probe_a)
            psi_a = model.psi_state.detach().clone()
            psi_a_system = _psi_system_state(model)

            # Stage B: learn the interfering task on a fresh ψ.
            model.reset_psi()
            optimizer_b = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            record_b = _run_adaptation(
                model,
                optimizer_b,
                criterion,
                shape,
                fn_b,
                epochs=epochs,
                probe=probe_b,
                feedback=recipe.feedback,
                adapt_entropy_beta=recipe.adapt_entropy_beta,
                adapt_temp_end=recipe.adapt_temp_end,
            )
            b_post = _probe_accuracy(model, probe_b)
            a_under_psi_b = _probe_accuracy(model, probe_a)
            psi_b = model.psi_state.detach().clone()

            # Restored stage: switch ψ back — no re-adaptation, no θ movement.
            _restore_psi_system(model, psi_a_system)
            a_retained = _probe_accuracy(model, probe_a)
            # Fresh-ψ floor: the meta-trained ψ-system before any task adaptation.
            _restore_psi_system(model, meta_psi_system)
            model.reset_psi()
            a_floor = _probe_accuracy(model, probe_a)
        report = audit.report
        if report is None:
            msg = "retention θ audit produced no report"
            raise Z3InstrumentError(msg)
    finally:
        _restore_rng(rng, shape.device)

    chance = TASK_CHANCE[name_a]
    margin = _GATE_ACCURACY_MARGIN
    items: dict[str, object] = {
        "theta_exact_invariant": bool(report.invariant),
        "psi_restore_reproduces_stage_a": a_retained == a_post,
        "psi_state_task_conditioned": float((psi_a - psi_b).abs().max()) > 0.0,
        "stage_a_acquired": a_post > chance + margin,
        "restored_psi_above_chance": a_retained > chance + margin,
        "restored_psi_beats_floor": a_retained > a_floor + margin,
    }
    failed = [k for k, v in items.items() if not v]
    retention = {
        "task_a": name_a,
        "task_b": name_b,
        "stage_a": {
            "accuracy": a_post,
            "steps_to_criterion": record_a.steps_to_criterion,
            "final_psi": [float(v) for v in psi_a.squeeze(0)],
        },
        "stage_b": {
            "accuracy": b_post,
            "steps_to_criterion": record_b.steps_to_criterion,
            "final_psi": [float(v) for v in psi_b.squeeze(0)],
            "task_a_accuracy_under_psi_b": a_under_psi_b,
        },
        "restored": {
            "task_a_accuracy": a_retained,
            "fresh_psi_floor": a_floor,
            "theta_change": report.max_abs_change,
        },
        "retention_gain_vs_floor": a_retained - a_floor,
        "forgetting_via_psi_switch": a_post - a_under_psi_b,
    }
    gate = {
        "chance": chance,
        "margin": margin,
        "items": items,
        "passed": not failed,
        "failed": failed,
    }
    if gate["passed"]:
        print("  retention gate: PASS")
    else:
        print(f"  retention gate: FAIL — {', '.join(failed)}")
    return {"retention": retention, "retention_gate": gate}


def evaluate_z3(  # ruff: ignore[too-many-arguments]
    coordinate: str,
    meta_train_epochs: int = 50,
    eval_epochs_per_task: int = 20,
    batch_size: int = 64,
    seq_len: int = 10,
    input_dim: int = 32,
    probe_batches: int = 16,
    device: torch.device | str = "cpu",
    seed: int = 42,
    *,
    recipe: MetaRecipe = MetaRecipe(),
    with_baselines: bool = True,
    task_order: Sequence[str] | None = None,
) -> dict:
    """Evaluate Z3: meta-train then freeze θ and evaluate task switching.

    Meta-training runs ``recipe``: optional forced-operator θ warm-up
    phase, then a straight-through controller phase over per-task episodes.
    ``with_baselines=False`` skips the E-10 control arms (triage rounds).
    ``task_order`` permutes the adaptation/baseline switching stream (all
    arms see the identical order); the realized order is echoed in results
    under ``"task_order"``.
    """
    from torch import nn

    torch.manual_seed(seed)
    random.seed(seed)
    device = get_device(device)

    parts = coordinate.split("/")
    if len(parts) != 6:
        raise ValueError(f"Invalid coordinate: {coordinate}")

    plasticity_type = parts[3]
    if plasticity_type != "rule_state":
        # Z3 requires rule_state plasticity
        raise ValueError(f"Z3 requires rule_state plasticity, got {plasticity_type}")

    # Build Z3 model
    model = Z3Model(
        num_operators=8,
        operator_dim=input_dim,
        controller_hidden=128,
        temperature=recipe.temp_start,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    tasks = [
        ("parity", create_parity_task),
        ("last_symbol", create_last_symbol_task),
        ("threshold", create_threshold_task),
    ]
    _apply_task_order(tasks, task_order)

    results: dict = {
        "coordinate": coordinate,
        "task_order": [name for name, _ in tasks],
        "tasks": {},
        "meta_recipe": {"epochs": meta_train_epochs}
        | {
            f: getattr(recipe, f)
            for f in (
                "episode_len",
                "feedback",
                "entropy_beta",
                "temp_start",
                "temp_end",
                "warmup_fraction",
                "warmup_lr",
                "entropy_end",
                "replay_steps",
                "adapt_temp",
                "adapt_entropy_beta",
            )
        },
    }
    shape = TaskShape(
        batch_size=batch_size, seq_len=seq_len, input_dim=input_dim, device=device
    )

    # ===== META-TRAINING PHASE =====
    _meta_train_phases(
        model,
        tasks,
        criterion,
        shape,
        recipe=recipe,
        meta_train_epochs=meta_train_epochs,
    )

    meta_state = _snapshot(model)

    # ===== EVALUATION PHASE (θ FROZEN, ψ adapts) =====
    print("  Evaluation phase (θ frozen)...")
    model.freeze_theta()
    if recipe.adapt_temp is not None:
        # Warmer gating during adaptation keeps sampling the unsolved
        # operators — flat-at-chance curves are exploration failures, not
        # optimization ones (2026-08-26 pilot rerun autopsy).
        model.temperature = recipe.adapt_temp
    assert model.verify_theta_frozen(), "θ not frozen!"  # ruff: ignore[assert]
    # Protocol state the frozen-ψ control arm must replicate exactly.
    entry_temperature = model.temperature

    # R8.1 wiring probe: ψ→gate causal link on the meta-trained controller.
    results["psi_gate_response"] = _gates_respond_to_psi(model, shape, seed=seed)

    # PR-2 audit: exact-diff θ across the whole switching/adaptation phase.
    # PR-1 hygiene: _adapt_all_tasks rebuilds Adam over the trainable set so
    # no meta-training momentum survives into ψ adaptation.
    rng_snapshot = _snapshot_rng(device)
    results["tasks"], psi_wall, report = _adapt_under_audit(
        model,
        tasks,
        criterion,
        shape,
        epochs=eval_epochs_per_task,
        probe_batches=probe_batches,
        recipe=recipe,
    )
    results["wall_clock_s"] = {"psi_adaptation": psi_wall}

    # R8.1: exact invariance — bitwise equality, not a tolerance band.
    results["theta_change"] = report.max_abs_change
    results["theta_invariant"] = report.invariant
    results["theta_sha256"] = hashlib.sha256(
        bytes(
            model.operator_embeddings
            .detach()
            .cpu()
            .contiguous()
            .view(-1)
            .view(torch.uint8)
            .tolist()
        )
    ).hexdigest()

    print(
        f"  θ change: {report.max_abs_change:.8f} "
        f"(exact-invariant: {results['theta_invariant']})"
    )
    for task_name, row in results["tasks"].items():
        print(
            f"    {task_name}: acc={row['accuracy']:.4f} "
            f"criterion@{row['steps_to_criterion']}"
        )

    # Operator diversity: entropy of post-adaptation gate usage
    entropy = _operator_diversity(
        model, tasks, criterion, shape, feedback=recipe.feedback
    )
    results["operator_diversity"] = entropy
    results["diversity_collapsed"] = bool(entropy < math.log(2))

    # ===== R8.1/R8.2 EMBEDDED FROZEN-ψ CONTROL + ENGAGEMENT GATE =====
    results["psi_control"], results["psi_gate"] = _run_psi_gate_arm(
        model,
        meta_state,
        tasks,
        results["tasks"],
        shape,
        epochs=eval_epochs_per_task,
        probe_batches=probe_batches,
        recipe=recipe,
        entry_temperature=entry_temperature,
        rng_snapshot=rng_snapshot,
        gate_response=results["psi_gate_response"],
        theta_invariant=results["theta_invariant"],
    )
    # Claims-scope audit status (R8.1): ψ-engaged iff the embedded gate passes
    # this run; a failed gate downgrades the run to no-ψ-claim plumbing.
    results["claims_scope"] = (
        CLAIMS_SCOPE_PSI_ENGAGED
        if results["psi_gate"]["passed"]
        else CLAIMS_SCOPE_PLUMBING_ONLY
    )

    # ===== R9.1 RETENTION PIVOT (A→B→A; θ frozen, ψ carries the task) =====
    results.update(
        _run_retention_arm(
            model,
            meta_state,
            tasks,
            shape,
            epochs=eval_epochs_per_task,
            probe_batches=probe_batches,
            recipe=recipe,
            entry_temperature=entry_temperature,
        )
    )

    # ===== BASELINES (E-10 control set) =====
    if not with_baselines:
        return results
    print("  Baselines: frozen floor / random-ψ / θ fine-tune...")
    results["baselines"] = _run_baselines(
        model,
        meta_state,
        tasks,
        criterion,
        shape,
        epochs=eval_epochs_per_task,
        probe_batches=probe_batches,
        feedback=recipe.feedback,
        adapt_entropy_beta=recipe.adapt_entropy_beta,
        adapt_temp_end=recipe.adapt_temp_end,
    )

    return results


def _git_commit() -> str:
    from computronium.utils import capture_environment

    return capture_environment()["git_commit"]


def run_z3_suite(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    coordinates: list[str],
    output_dir: Path,
    meta_train_epochs: int = 50,
    eval_epochs: int = 20,
    batch_size: int = 64,
    seq_len: int = 10,
    input_dim: int = 32,
    probe_batches: int = 16,
    seeds: int = 3,
    device: str = "auto",
    recipe: MetaRecipe = MetaRecipe(),
) -> list[dict]:
    """Run Z3 fixed weights benchmark suite."""
    device = get_device(device)

    config = {
        "coordinates": coordinates,
        "meta_train_epochs": meta_train_epochs,
        "eval_epochs": eval_epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "input_dim": input_dim,
        "probe_batches": probe_batches,
        "seeds": seeds,
        "device": str(device),
        **dataclasses.asdict(recipe),
    }

    all_results = []

    for coord in coordinates:
        print(f"\nEvaluating Z3: {coord}")
        coord_results = {"coordinate": coord, "seeds": []}

        for seed in range(seeds):
            print(f"  Seed {seed}...")
            result = evaluate_z3(
                coordinate=coord,
                meta_train_epochs=meta_train_epochs,
                eval_epochs_per_task=eval_epochs,
                batch_size=batch_size,
                seq_len=seq_len,
                input_dim=input_dim,
                probe_batches=probe_batches,
                device=device,
                seed=seed,
                recipe=recipe,
            )
            coord_results["seeds"].append(result)

            # Print task accuracies
            for task_name, task_result in result["tasks"].items():
                print(f"    {task_name}: {task_result['accuracy']:.4f}")
            print(
                f"    θ-change: {result['theta_change']:.8f}, Invariant: {result['theta_invariant']}"
            )

        # Aggregate
        if coord_results["seeds"]:
            theta_changes = [s["theta_change"] for s in coord_results["seeds"]]
            diversities = [s["operator_diversity"] for s in coord_results["seeds"]]
            coord_results["mean_theta_change"] = sum(theta_changes) / len(theta_changes)
            coord_results["mean_operator_diversity"] = sum(diversities) / len(
                diversities
            )

            # Task accuracies
            for task_name in ["parity", "last_symbol", "threshold"]:
                accs = [
                    s["tasks"][task_name]["accuracy"] for s in coord_results["seeds"]
                ]
                coord_results[f"mean_{task_name}_accuracy"] = sum(accs) / len(accs)

            coord_results["gate_passed"] = all(
                s["psi_gate"]["passed"] for s in coord_results["seeds"]
            )
            coord_results["retention_gate_passed"] = all(
                s["retention_gate"]["passed"] for s in coord_results["seeds"]
            )
            coord_results["mean_retained_task_a_accuracy"] = sum(
                s["retention"]["restored"]["task_a_accuracy"]
                for s in coord_results["seeds"]
            ) / len(coord_results["seeds"])

        all_results.append(coord_results)

    # Save results (E-3 manifest: pinned config hash + git commit next to data)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "z3_fixed_weights_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2)
    manifest_file = output_dir / "manifest.json"
    with manifest_file.open("w") as f:
        json.dump(
            {
                "config": config,
                "config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ).hexdigest(),
                "git_commit": _git_commit(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Z3 Fixed Weights Benchmark Summary (Level 4)")
    print("=" * 80)
    print(
        f"{'Coordinate':<50} {'Parity':<8} {'LastSym':<8} {'Thresh':<8} "
        f"{'θ-change':<10} {'Diversity':<10} {'Gate':<6}"
    )
    print("-" * 80)
    for r in all_results:
        coord_short = (
            r["coordinate"][:48] + ".."
            if len(r["coordinate"]) > 50
            else r["coordinate"]
        )
        prim = r["coordinate"].split("/")[3]
        print(
            f"{coord_short:<50} "
            f"{r.get('mean_parity_accuracy', 0):<8.4f} "
            f"{r.get('mean_last_symbol_accuracy', 0):<8.4f} "
            f"{r.get('mean_threshold_accuracy', 0):<8.4f} "
            f"{r.get('mean_theta_change', 0):<10.8f} "
            f"{r.get('mean_operator_diversity', 0):<10.4f} "
            f"{'PASS' if r.get('gate_passed') else 'FAIL':<6} {prim}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Z3 Fixed Weights Benchmark (Level 4)")
    parser.add_argument("--coordinates", nargs="+", help="Coordinates to test")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/z3_fixed_weights",
        help="Output directory",
    )
    parser.add_argument(
        "--meta-train-epochs", type=int, default=50, help="Meta-training epochs"
    )
    parser.add_argument(
        "--eval-epochs", type=int, default=20, help="Evaluation epochs per task"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length")
    parser.add_argument("--input-dim", type=int, default=32, help="Input dimension")
    parser.add_argument(
        "--probe-batches",
        type=int,
        default=16,
        help="Fixed probe batches scored per adaptation step",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument(
        "--episode-len",
        type=int,
        default=8,
        help="Consecutive same-task batches per meta-training episode",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disable loss/gate feedback into ψ (attack (a) off)",
    )
    parser.add_argument(
        "--entropy-beta",
        type=float,
        default=0.05,
        help="Gate-entropy bonus weight during controller meta-training",
    )
    parser.add_argument(
        "--temp-start", type=float, default=2.0, help="Gating temperature at start"
    )
    parser.add_argument(
        "--temp-end", type=float, default=0.5, help="Gating temperature at end"
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.4,
        help="Share of meta epochs spent in forced-operator θ warm-up",
    )
    parser.add_argument(
        "--entropy-end",
        type=float,
        default=None,
        help="Entropy-bonus curriculum target (anneals beta→end; None = constant)",
    )
    parser.add_argument(
        "--replay-steps",
        type=int,
        default=0,
        help="Supervised replay distillation passes per meta epoch (attack c)",
    )
    parser.add_argument(
        "--adapt-temp",
        type=float,
        default=None,
        help="Gating temperature during ψ adaptation (None = end-of-anneal temp)",
    )
    parser.add_argument(
        "--adapt-entropy-beta",
        type=float,
        default=0.0,
        help="Gate-entropy bonus during ψ adaptation (exploration floor)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (10 meta, 5 eval, 1 seed)"
    )
    args = parser.parse_args()

    if args.quick:
        args.meta_train_epochs = 10
        args.eval_epochs = 5
        args.seeds = 1

    coordinates = args.coordinates or [
        "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean",
    ]

    run_z3_suite(
        coordinates=coordinates,
        output_dir=Path(args.output_dir),
        meta_train_epochs=args.meta_train_epochs,
        eval_epochs=args.eval_epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        probe_batches=args.probe_batches,
        seeds=args.seeds,
        device=args.device,
        recipe=MetaRecipe(
            episode_len=args.episode_len,
            feedback=not args.no_feedback,
            entropy_beta=args.entropy_beta,
            temp_start=args.temp_start,
            temp_end=args.temp_end,
            warmup_fraction=args.warmup_fraction,
            entropy_end=args.entropy_end,
            replay_steps=args.replay_steps,
            adapt_temp=args.adapt_temp,
            adapt_entropy_beta=args.adapt_entropy_beta,
        ),
    )


if __name__ == "__main__":
    main()
