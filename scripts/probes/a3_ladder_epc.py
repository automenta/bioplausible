"""A3 — ePC depth axis under the ablation ladder (TODO12 Workstream A).

Question (TODO12 A3/A4 rows): does ePC train at depth 8-20 WITHOUT
Muon? D14's faithful regime already trains depth 20; the sharper test
is the simple F1 regime (MNIST quick, width 64, batch-capped, one
epoch) with the ladder updates: unit_rms (magnitude-only), Muon
(orthogonalizing), Euclid (the F1 baseline).

Pre-registered prediction (TODO12, after A2): unit_rms carries ePC to
depth >= 8 — the crutch is magnitude, not direction, so orthogonalizing
the update should not be load-bearing once the step is unit-RMS
normalized. Falsification: depth walls reappear under unit_rms but not
under Muon => orthogonalization carries depth-side direction signal.

Also records per-layer credit norms at depth 20 (the F1 instrument's
attenuation reading — the ~4x/layer decay A4 targets).

A3 VERDICT (2026-09-06, same session; MNIST quick, w64, 60 batches,
1 epoch — the simple F1 regime):
- The simple-regime depth wall PERSISTS under the ladder: unit_rms
  best (lr 0.02) reaches 0.206 at depth 2 and walls at depth 8+;
  Muon@0.01 does better at depth 8 (0.276) but walls at 16/20 too;
  Euclid@0.2 walls from depth 8 (0.113). Consistent with F1's own
  record (ePC simple-regime depth 8 = 0.108 ≈ chance) — this probe
  CONFIRMS the known boundary, it does not overturn it.
- unit_rms lr response is non-monotonic on MNIST (0.02 > 0.05/0.1 >
  0.2-diverges) — the 60-batch budget limits conclusions; the F1
  lesson stands: trainer-regime artifacts before mechanism claims.
- Split with A2: magnitude normalization fixed the WIDTH axis on LM
  (ePC width-robust, D18 pinned) but NOT the DEPTH axis in the simple
  regime. The depth wall is a credit-channel property — A4
  (credit-space normalization on the propagated error) or the D14
  faithful-regime composition (residual + muPC + Adam) are the
  indicated levers, matching TODO12's adaptive-branch table.
- Credit norms at depth 20 reproduce the F1 attenuation signature
  (per-layer norms decay / trap at 0.0 by mid-stack; unit_rms@0.02
  shows the geometric-decay profile with a final-layer spike 2967).

A5 GAIN-CONTROL RUNG (TODO12, 2026-09-06, ``--gain-control``): ePC depth
sweep with settle-path hidden-layer unit-RMS renorm at emit.
Pre-registered prediction: if the simple-regime depth wall is carried by
the unnormalized hidden-activity channel, gain_control=unit_rms lifts
depth-8+ accuracy above the plain-ladder baselines (unit_rms 0.206@2,
walls at 8+; euclid 0.113@8). Falsification: no lift, or worse —
renormalizing emitted acts distorts the ε semantics the way credit_norm
harmed the faithful regime (A4×D14 precedent).

A5 DEPTH VERDICT (2026-09-06, 48 s wall, CPU): prediction FALSIFIED —
gain_control=unit_rms gives no depth lift (unit_rms 0.245@2 vs 0.206
baseline, then walls at 8+: 0.099/0.118/0.098; muon/euclid unchanged).
The credit norms still blow up under gain_control at depth 8 (1.1e7 at
the output layer) — activity-side renorm does not condition the credit
channel. The depth wall is a CREDIT-side property (A4's finding, now
confirmed from the activity side); the faithful regime needs nothing.
A5's residual value: bounded hidden acts as infrastructure, not a wall
repair.
"""

import argparse
from itertools import islice

import torch

from computronium import (
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_system,
    create_task,
)
from computronium.ontology.credit import Phase
from computronium.ontology.update import RiemannianOrthogonalUpdate, UnitRMSUpdate

WIDTH = 64
DEPTHS = (2, 8, 16, 20)
BATCH_CAP = 60
BUDGET = 5  # ePC settle budget (the D12 1/3 regime)
CHANCE = 0.1


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _build(update: str, lr: float, depth: int, gain_control: str = "none"):
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * depth
        )
    )
    dynamics = ErrorPredictiveCodingDynamics(
        StateDynamicsConfig.error_predictive_coding(
            max_steps=BUDGET,
            step_size=0.1,
            gain_control=gain_control,  # type: ignore[arg-type]
        )
    )
    if update == "unit_rms":
        update_obj = UnitRMSUpdate(
            ParameterUpdateConfig.unit_rms(step_size=lr, momentum=0.9)
        )
    elif update == "muon":
        update_obj = RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=lr, momentum=0.9)
        )
    else:
        update_obj = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=lr))
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry,
        dynamics=dynamics,
        credit=ThermodynamicContrast(),
        update=update_obj,
    )


def _credit_norms(system, substrate, x, y) -> list[float]:
    free = system.dynamics.settle(SystemState(x=x), system.geometry, substrate, None)
    nudged = system.dynamics.settle(
        SystemState(x=x), system.geometry, substrate, target=y
    )
    grads = system.credit.compute_pseudo_gradient(
        {Phase.FREE: free, Phase.NUDGED: nudged}, None, system.geometry
    )
    return [round(g.norm().item(), 3) for g in grads]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gain-control",
        choices=("none", "unit_rms", "spectral"),
        default="none",
        help="A5: settle-path hidden-layer gain control",
    )
    args = ap.parse_args()
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()  # type: ignore[attr-defined]
    torch.manual_seed(0)
    train_data = list(_flatten(task.get_dataloader("train"), BATCH_CAP))  # type: ignore[attr-defined]
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    x, y = train_data[0]

    lrs = {"unit_rms": (0.2, 0.02), "muon": (0.01,), "euclid": (0.2,)}
    print(f"{'update':>9} {'lr':>6} {'depth':>5}  train_acc  credit_norms")
    for update, lrs_u in lrs.items():
        for lr in lrs_u:
            for depth in DEPTHS:
                torch.manual_seed(0)
                system = _build(update, lr, depth, gain_control=args.gain_control)
                trainer = SystemTrainer(
                    system=system, config=config, train_data=train_data
                )
                t0 = __import__("time").time()
                trainer.fit()
                hist = trainer.history[-1]
                acc = hist.get("train_acc", hist.get("accuracy", float("nan")))
                norms = _credit_norms(system, substrate, x, y)
                print(
                    f"{update:>9} {lr:>6} {depth:>5}  {acc:>9.3f}  "
                    f"{norms}  ({round(__import__('time').time() - t0, 1)} s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
