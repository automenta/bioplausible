"""D22 pre-registration probe: ψ-only adaptation across an A→B switch.

Pre-registered predictions (written BEFORE the run, TODO12 rev 12):

- P1 (mechanism): with θ bitwise frozen (SHA-256 identical before/after
  stage B), routing-ψ adaptation on Task B moves stage-B probe accuracy
  OFF the frozen-null floor, and ψ is task-conditioned
  (‖gate_logits_B‖ > 0, per-unit mask std > 0).
- P2 (value): lr/effective-step-matched θ fine-tuning on the SAME switch
  reaches higher B accuracy than ψ-only at the same episode budget, but
  forgets more Task A (retention A: psi_only ≥ finetune). If ψ-only
  beats fine-tune on B accuracy, that is a headline — record either way.
- P3 (boundary): the frozen-null arm (θ frozen, ψ zero, no modulation)
  is the stage-A θ's raw Task-B performance — the probe-the-probe.

Arms: parity (Task A) → last-symbol (Task B), Z3 task helpers flattened
to (batch, seq*input). Stage A: gradient credit × euclidean. Stage B:
θ frozen via requires_grad_(False); ψ steps via RoutingPlasticity (fixed
per-gate drive G, per-unit masks U_ℓ — the F3 realization). lr-matched
control: the routing mask halves the effective forward gain (F3 lesson:
matched lr ≈ lr/2) — the finetune arm runs at lr/2, same episode count.

Walltime: printed, never recorded.

VERDICT (2026-09-06, probe run ~2 s CPU; both pre-registered predictions
FALSIFIED + fast-weight extension negative):

- Stage A mastery 0.977 (cumsum rule). Frozen-null: B 0.644 (transfer
  floor), A 0.973.
- P1 FALSIFIED: routing-ψ does NOT move Task B off the floor —
  b 0.656 → {0.637–0.668} across psi_lr {0.05, 0.2, 0.5} × episodes
  {200, 600} (all within probe noise of the 0.656 start; higher
  psi_lr only inflates ‖ψ‖ with no accuracy gain). Gate logits DO
  differentiate per-sample (std 0.35–3.5, mask machinery live) but the
  input-driven gate law carries no task signal.
- FastWeight-ψ extension (D1's other named ψ): also no acquisition —
  b 0.656 → {0.637–0.648} at lr 0.1 (600 ep), degrades Task A at lr 0.5
  (A 0.891 → 0.566). The Hebbian outer x ⊗ free-settled-output is a
  FIXED function of x under frozen θ — no episode-adaptive learning.
- P2 RECORDS HONESTLY: lr-matched θ fine-tuning acquires B decisively
  (0.984) at real forgetting cost (A 0.660 vs ψ-only's trivially
  preserved 0.969 ≈ null 0.973). The retention "advantage" is not an
  adaptation mechanism — ψ-only simply never changes behavior.
- ROOT CAUSE: the ψ-step contract feeds ψ the FIRST-phase (FREE,
  target-free) settled activity; no landed plasticity law consumes a
  loss/target term — the same missing-supervised-term disease F2 found
  on the STDP path (B5), now confirmed on the P-axis. A supervised
  ψ term (B5-adjacent, generalized to the P-axis) or metaplasticity
  (D2) is the indicated lever before D22 can land as a positive demo.
- Mechanism claim as pre-registered (‖Δθ‖ = 0 exact) is demonstrated
  and trivially true — but per probe-first rule 3, a demo pinning
  "solves the switch" would be FALSE; D22 is NOT promoted. The
  ψ-timescale boundary is mapped (TODO12 Open Questions row resolved
  NO).
"""

from __future__ import annotations

import hashlib
import time

import torch
from torch import Tensor

from computronium import (
    BackpropCredit,
    CreditAssignmentConfig,
    EuclideanUpdate,
    GeometryConfig,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    compose_system_from_configs,
)
from computronium.core.pipeline import forward_pass, run_train_step
from computronium.core.plasticity.routing import RoutingPlasticity
from computronium.experiments.joint.adaptation_efficiency import (
    create_switching_task,
)

SEED = 0
STAGE_A_EPISODES = 300
STAGE_B_EPISODES = 200
BATCH = 32
SEQ = 4
INPUT_DIM = 8
PROBE_BATCHES = 8
LR = 0.05
GATE_DIM = 64
PSI_LR = 0.05


def _theta_sha256(system) -> str:
    raw = b"".join(
        p.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        for p in system.geometry.params.values()
    )
    return hashlib.sha256(raw).hexdigest()


def _batch(task: str) -> tuple[Tensor, Tensor]:
    x, y = create_switching_task(BATCH, SEQ, INPUT_DIM, phase=task)
    return x.reshape(BATCH, -1), y


def _probe(system, task: str) -> float:
    correct = total = 0
    with torch.no_grad():
        for _ in range(PROBE_BATCHES):
            x, y = _batch(task)
            correct += (_settled_out(system, x).argmax(-1) == y).sum().item()
            total += len(y)
    return correct / total


def _settled_out(system, x: Tensor) -> Tensor:
    state = SystemState(x=x)
    state.activations = forward_pass(system.substrate, system.geometry, x)
    settled = system.dynamics.settle(
        state, system.geometry, system.substrate, target=None
    )
    acts = settled.activations
    return acts[-1] if isinstance(acts, list) else acts


def _probe_modulated(system, plasticity, psi, task: str) -> float:
    correct = total = 0
    with torch.no_grad():
        for _ in range(PROBE_BATCHES):
            x, y = _batch(task)
            acts = plasticity.modulate(_settled_out(system, x), psi)
            out = acts[-1] if isinstance(acts, list) else acts
            correct += (out.argmax(-1) == y).sum().item()
            total += len(y)
    return correct / total


def _mask_std(plasticity, psi) -> float:
    acts = [torch.zeros(4, 32), torch.zeros(4, 2)]
    mod = plasticity.modulate(acts, psi)
    return float(torch.stack([m.std(dim=0).mean() for m in mod]).mean())


def _freeze(system) -> None:
    for p in system.geometry.params.values():
        p.requires_grad_(False)


def _train_stage_a(system) -> float:
    for _ in range(STAGE_A_EPISODES):
        x, y = _batch("A")
        run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x,
            y,
        )
    return _probe(system, "A")


def _stage_b_psi(system) -> dict[str, float | bool]:
    plasticity = RoutingPlasticity(gate_dim=GATE_DIM, learning_rate=PSI_LR)
    psi = plasticity.initial_psi(None, 1)
    context = _Context(system.geometry.params)  # type: ignore[assignment]
    _freeze(system)
    hash_before = _theta_sha256(system)
    b_start = _probe_modulated(system, plasticity, psi, "B")
    from computronium.state import CompositeState

    for _ in range(STAGE_B_EPISODES):
        x, y = _batch("B")
        with torch.no_grad():
            state = SystemState(x=x, y=y)
            state.activations = forward_pass(system.substrate, system.geometry, x)
            settled = system.dynamics.settle(
                state, system.geometry, system.substrate, target=None
            )
            z = CompositeState(
                activity={"x": x, "y": settled.activations[-1]},
                plastic=psi,
                substrate={},
            )
            psi = plasticity.step(psi, z, context)  # type: ignore[arg-type]
            settled.activations = plasticity.modulate(settled.activations, psi)
    return {
        "b_start": b_start,
        "b_final": _probe_modulated(system, plasticity, psi, "B"),
        "a_retained": _probe_modulated(system, plasticity, psi, "A"),
        "theta_invariant": hash_before == _theta_sha256(system),
        "psi_norm": float(psi["gate_logits"].norm()),
        "mask_std": _mask_std(plasticity, psi),
    }


def _stage_b_null(system) -> dict[str, float | bool]:
    _freeze(system)
    return {
        "b_final": _probe(system, "B"),
        "a_retained": _probe(system, "A"),
    }


def _stage_b_finetune(system) -> dict[str, float | bool]:
    b_start = _probe(system, "B")
    credit = BackpropCredit(CreditAssignmentConfig.gradient())
    euclid = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR / 2))
    for _ in range(STAGE_B_EPISODES):
        x, y = _batch("B")
        run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            credit,
            euclid,
            x,
            y,
        )
    return {
        "b_start": b_start,
        "b_final": _probe(system, "B"),
        "a_retained": _probe(system, "A"),
    }


class _Context:
    """Minimal SystemContext stand-in: plasticity.step reads .theta/.device."""

    def __init__(self, theta):
        self.theta = theta
        self.device = torch.device("cpu")


def main() -> int:
    torch.manual_seed(SEED)
    t0 = time.time()
    stage_a: list[float] = []
    results: dict[str, dict[str, float | bool]] = {}
    builders = {
        "psi_only": _stage_b_psi,
        "null": _stage_b_null,
        "finetune": _stage_b_finetune,
    }
    for arm, stage_b in builders.items():
        torch.manual_seed(SEED)
        system = compose_system_from_configs(
            SubstrateConfig.digital(),
            GeometryConfig.feedforward(
                input_dim=SEQ * INPUT_DIM, output_dim=2, hidden_dims=(32,)
            ),
            StateDynamicsConfig.instantaneous(),
            CreditAssignmentConfig.gradient(),
            ParameterUpdateConfig.euclidean(step_size=LR),
        )
        a_mastery = _train_stage_a(system)
        stage_a.append(a_mastery)
        results[arm] = stage_b(system)
        print(f"{arm}: stage A {a_mastery:.3f}  {results[arm]}", flush=True)

    print(f"\nwalltime {time.time() - t0:.1f}s (printed, never recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
