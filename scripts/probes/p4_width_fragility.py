"""P4 — width fragility of local feedback on LM (Fundamental-Research Focus).

Measured facts to explain: ePC×Muon and pepita explode at w64 (smoke
shape) and train at w256+; pure FF is flat at every width. One sweep:
width ∈ {32, 64, 128, 256} × credit ∈ {ff, ff_hybrid, pepita, epc_thermo},
depth 4 (fixed — isolates width), Muon at each credit's tuned lm lr,
fixed 75 s budget per arm, ctx 32, batch 32.

Recorded per arm: final val ppl (chance 65), stability (finite final
loss), and per-layer hidden-activation std at the end (the signal-to-noise
reading: local feedback's usable signal vs the activity scale it rides).

Caveat (scope-honest): lrs are the registered per-credit values, not
re-tuned per width — the claim is fragility at the working regime, and
width-dependent lr sensitivity is itself part of the fragility.

Run: uv run python scripts/probes/p4_width_fragility.py

A0 extension (TODO12): ``--update natural_gradient`` re-runs the fragile
cells (pepita w32/w128, epc w32) with the per-tensor mean-|grad| normalizer
(MeanNormUpdate — effective step == step_size, the D16 lesson) and a
per-cell lr micro-sweep, since its lr IS its step size.

Pre-registered prediction (RESEARCH4 / TODO12 A0): if magnitude
normalization alone widens the stable band, the "credit direction is
approximately right; only the magnitude is broken" hypothesis is
half-confirmed with zero new library primitives.

A0 VERDICT (2026-09-06, matched-step controls per P3 discipline):
- epc_thermo w32: natural_gradient @ 1e-4 trains — val_ppl 41.2 (chance
  65), act_std bounded (max 4.05, ~1.5x/layer). Muon @ matched steps
  (1e-4, 1e-5) is STABLE-BUT-FLAT at chance (65.1-65.2, signal
  collapses). Muon @ registered lr 0.01 explodes (P4: std -> 2028).
  => ePC's credit direction is approximately right; magnitude
  normalization alone (no momentum, no orthogonalization) rescues w32
  (subject to the same defect-audit caveat below).
- pepita w32/w128: natural_gradient EXPLODES at every lr (1e-4 .. 1e-5;
  act_std grows 20x/layer even at 1e-5) while Muon @ 1e-5 is
  stable-but-flat. => Removing orthogonalization un-masks pepita's
  directional collapse (P4/P5: fixed B is directionally random).
  RESEARCH4's "per-layer normalization fixes PEPITA" prediction is
  FALSIFIED for the mean-|grad| normalizer: pepita needs DIRECTION
  (learned feedback, B1), not magnitude.
- Split terminus: the unifying hypothesis holds per-rule — ePC fails on
  magnitude, PEPITA fails on direction. A1's UnitRMS rung will test
  whether orthogonalization-vs-magnitude is the whole story for ePC.

A1 LADDER VERDICT (2026-09-06, A2 partial: P4-harness fragile cells,
75 s/arm, single seed — multi-seed confirmation pending for D18):
- epc_thermo w32 trains WITHOUT Muon on every magnitude rung:
  natural_gradient 41.2 @ 1e-4; local_adam 44.5 @ 3e-4; unit_rms
  34.9 @ 3e-4 — the best w32 result in the ladder (momentum EMA +
  unit-RMS normalization beats instantaneous normalization).
- epc_thermo w64 — the cell whose Muon run exploded (P4: std -> 2028) —
  trains under unit_rms: ppl 28.3-28.5 @ 1e-4/3e-4. THE OPTIMIZER
  CRUTCH IS DEAD FOR ePC IN THE w32-64 BAND.
- pepita w32/w128: every magnitude rung explodes at every lr
  (natural_gradient, unit_rms, local_adam — act_std grows 20x/layer);
  Muon small-lr is stable-flat. PEPITA fails at HEAD under the current
  fixed-B configuration at these step shapes — CAUTION (user directive
  2026-09-06): this is an observed behavior, not a settled mechanism
  claim. Open defect candidates before concluding direction is
  fundamentally wrong: (a) feedback_scale=0.01 was tuned for Muon's
  step shape — retune per rung; (b) Muon's orthogonalization may have
  been acting as the credit channel's missing gain control — test
  per-hop credit normalization + unit_rms before condemning fixed-B;
  (c) step-0 vs post-training act stds to separate cause from symptom.
  RESEARCH4's Fix-4 prediction is UNCONFIRMED for PEPITA at HEAD, not
  falsified in principle.
- Optimizer–credit co-design reading (A6 input): momentum EMA helps the
  normalized rungs (unit_rms > natural_gradient on ePC); per-coordinate
  Adam structure is not needed (local_adam < unit_rms).

A2 COMPLETION + DEFECT AUDIT (2026-09-06, same session):
- Width sweep completed for ePC under unit_rms: w128 25.3-27.4,
  w256 24.8-29.6 (lr 1e-4/3e-4) — ePC is WIDTH-ROBUST under unit_rms,
  matching ff_hybrid's robustness profile; Muon required w >= 256
  (P4: epc w32 std -> 2028). Multi-seed confirmed at w64:
  ppl 28.1/28.4 (seeds 1/2); PEPITA explodes identically across seeds.
- PEPITA defect audit (user caution: audit before condemning):
  (a) feedback_scale {1e-4, 1e-3, 0.1} — INERT under the normalized
      rungs (fscale scales the pseudo-gradient linearly; per-tensor
      normalization removes it). Not a tuning artifact.
  (c) step-0 act stds healthy and credit-independent — explosion
      develops through training, update-rule-coupled.
  (b) centered-e1 variant (lm_local_audit's CenteredPepitaCredit) —
      does NOT tame the explosion (unit_rms 1e-4..1e-3 all explode).
      The constant-one-hot-term candidate is ruled out.
  REMAINING interpretation: the runaway is structural to the current
  DFA-style PEPITA realization — fixed random row-space per weight
  (B frozen at first use) + unbounded activity loop; Muon's
  orthogonalization was implicitly suppressing it. Honest repairs:
  B1 (learned/adaptive B changes the update's row space) or A5
  (settle-path gain homeostasis bounds the loop). This is an
  audit-backed observation at HEAD for these realizations, NOT a
  claim about PEPITA-in-principle (a faithful forward-modulation
  PEPITA is a different realization).

B1 LEARNED-FEEDBACK EXTENSION (TODO12, 2026-09-06, ``--learned-feedback``):
P4 cells with learned_feedback=True (transport-free reconstruction B:
closed-form ridge post @ C ≈ e1 per weight, B = Cᵀ·fscale, EMA 0.5,
every step; never reads param.data).
Pre-registered prediction: if the fixed-B row space is the runaway
driver (A2 audit's remaining interpretation), pepita w32/w128 under
unit_rms — which explodes on every magnitude rung at every lr — now
  TRAINS (val_ppl < 65, bounded act_std). If it still explodes, the
  runaway is the unbounded activity loop itself and A5 (settle-path gain
  homeostasis), not B1, is the honest repair.

A5 GAIN-CONTROL EXTENSION (TODO12, 2026-09-06, ``--gain-control``):
settle-path hidden-layer renorm at emit (unit_rms = μPC per-sample
unit RMS; output/input untouched). B1's falsified prediction hands the
PEPITA-repair mandate here: the unbounded settle-activity loop is the
last remaining candidate driver.
Pre-registered prediction: if the loop is the driver, pepita w32/w128
under unit_rms + gain_control=unit_rms TRAINS (val_ppl < 65, finite
loss). CAVEAT: post-settle act_std is bounded by construction under
gain_control — the stability read is val_ppl/finite loss, not act_std;
the output layer stays unnormalized so the loop can still explode
through the readout.

READOUT-NORM EXTENSION (TODO12, 2026-09-06, ``--readout-norm unit_rms``):
update-side unit-RMS normalization of the OUTPUT-weight pseudo-gradient
(per-tensor; bias and hidden grads untouched; CE landscape untouched —
only step shape changes). Targets the readout-path suspect left after
B1+A5: persistent saturated-e1 update direction on the output weights.
Pre-registered prediction: if the runaway is readout-path (output-weight
step shape), pepita w32/w128 under unit_rms + --readout-norm unit_rms
TRAIN (val_ppl < 65, finite loss) at some swept lr; combining with
--gain-control unit_rms closes the loop if BOTH hidden and readout
paths must be bounded. If it still diverges, the runaway is neither
hidden-activity nor output-step-shape and the realization is retired
honestly (faithful forward-modulation PEPITA remains untested).

READOUT-NORM VERDICT (2026-09-06, both predictions FALSIFIED, ~5 min
RTX 3080 per grid): (1) ``--readout-norm unit_rms`` alone — hidden
loop explodes as always (~20×/layer). (2) ``--readout-norm unit_rms
--gain-control unit_rms`` — hidden acts bounded (~0.8) AND output
step unit-normalized, yet output act_std still explodes (5.1e4–7.1e10)
and val_ppl saturates at the sentinel. The runaway therefore lives in
the weight trajectory itself (output-weight magnitude growth), not in
per-step direction shape or hidden activity. Ruled out for PEPITA
divergence at HEAD: feedback_scale (A2), centered-e1 (A2), fixed-B row
space (B1), hidden gain (A5), output step shape (this rung). The
realization stays retired for the local-credit story; the
faithful-forward-modulation variant remains the untested alternative.
"""

import argparse
import math
import time

import lm_comparison as lmc
import torch

from computronium import (
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    ThermodynamicContrast,
    compose_system,
)
from computronium.core.pipeline import run_train_step
from computronium.ontology.update import (
    LocalAdamUpdate,
    MeanNormUpdate,
    RiemannianOrthogonalUpdate,
    UnitRMSUpdate,
)

WIDTHS = (32, 64, 128, 256)
DEPTH = 4
BUDGET_S = 75.0
CTX = 32
BATCH = 32
LR = {"ff": 0.01, "ff_hybrid": 0.02, "pepita": 5e-4, "epc_thermo": 0.01}


def build(
    credit: str,
    width: int,
    update: str = "muon",
    lr: float | None = None,
    fscale: float = 0.01,
    learned_fb: bool = False,
    gain_control: str = "none",
):
    if credit == "ff_hybrid":
        credit_obj = LocalGoodnessCredit(
            lmc.CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff", readout_error=True
            )
        )
    elif credit == "pepita":
        credit_obj = LocalGoodnessCredit(
            lmc.CreditAssignmentConfig.local_goodness(
                feedback_scale=fscale,
                local_objective="pepita",
                learned_feedback=learned_fb,
                feedback_lr=0.5,
                feedback_update_every=1,
            )
        )
    elif credit == "ff":
        credit_obj = LocalGoodnessCredit(
            lmc.CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff"
            )
        )
    else:
        credit_obj = ThermodynamicContrast()
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=CTX * lmc.VOCAB,
            output_dim=lmc.VOCAB,
            hidden_dims=(width,) * DEPTH,
        )
    )
    if credit == "epc_thermo":
        dynamics = ErrorPredictiveCodingDynamics(
            StateDynamicsConfig.error_predictive_coding(
                max_steps=10,
                step_size=0.1,
                gain_control=gain_control,  # type: ignore[arg-type]
            )
        )
    else:
        dynamics = InstantaneousDynamics(
            StateDynamicsConfig.instantaneous(gain_control=gain_control)  # type: ignore[arg-type]
        )
    step = lr if lr is not None else LR[credit]
    update_obj: object
    match update:
        case "mean_norm":
            update_obj = MeanNormUpdate(
                lmc.ParameterUpdateConfig.mean_norm(step_size=step)
            )
        case "unit_rms":
            update_obj = UnitRMSUpdate(
                lmc.ParameterUpdateConfig.unit_rms(step_size=step, momentum=0.9)
            )
        case "local_adam":
            update_obj = LocalAdamUpdate(
                lmc.ParameterUpdateConfig.local_adam(step_size=step, momentum=0.9)
            )
        case _:
            update_obj = RiemannianOrthogonalUpdate(
                lmc.ParameterUpdateConfig.riemannian_orthogonal(
                    step_size=step, momentum=0.9
                )
            )
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=lmc.DEVICE)),
        geometry=geometry,
        dynamics=dynamics,
        credit=credit_obj,
        update=update_obj,
    )


class _ReadoutNormProxy:
    """Unit-RMS-normalize the output-weight pseudo-gradient before the inner step."""

    def __init__(self, inner: object):
        self.inner = inner

    def step(self, params, pseudo_grads, geometry):
        grads = list(pseudo_grads)
        # pseudo-grads are one per weight matrix (biases excluded), in
        # weight order — the last grad is the output weight.
        if grads:
            g = grads[-1]
            rms = g.pow(2).mean().sqrt()
            if torch.isfinite(rms) and rms > 0:
                grads[-1] = g / (rms + 1e-8)
        return self.inner.step(params, grads, geometry)  # type: ignore[attr-defined]


def act_stds(system) -> list[float]:
    x = torch.randn(8, CTX * lmc.VOCAB, device=lmc.DEVICE)
    with torch.no_grad():
        state = system.dynamics.settle(
            SystemState(x=x), system.geometry, system.substrate, None
        )
    acts = state.activations
    return [float(a.std()) for a in (acts if isinstance(acts, list) else [acts])]


def run_arm(system, tokens, val, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    step = 0
    train_loss = float("nan")
    while time.time() - t0 < BUDGET_S:
        idx = torch.randint(0, len(tokens) - CTX - 1, (BATCH,), generator=gen)
        win = tokens[idx.unsqueeze(1) + torch.arange(CTX + 1)]
        x = (
            torch.nn.functional
            .one_hot(win[:, :-1], lmc.VOCAB)
            .float()
            .reshape(BATCH, CTX * lmc.VOCAB)
            .to(lmc.DEVICE)
        )
        y = win[:, -1].to(lmc.DEVICE)
        train_loss = run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x,
            y,
        )["loss"]
        step += 1
    stds = act_stds(system)
    stable = math.isfinite(train_loss) and train_loss < 1e4
    return {
        "steps": step,
        "train_loss": round(train_loss, 4),
        "stable": stable,
        "act_std": [round(s, 4) for s in stds],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--update",
        choices=("muon", "mean_norm", "unit_rms", "local_adam"),
        default="muon",
    )
    ap.add_argument("--cells", nargs="*", help="credit:width filters, e.g. pepita:32")
    ap.add_argument("--lrs", nargs="*", type=float, help="lr micro-sweep (A0)")
    ap.add_argument("--fscale", type=float, default=0.01, help="PEPITA feedback_scale")
    ap.add_argument(
        "--learned-feedback",
        action="store_true",
        help="B1: learned transport-free reconstruction B (pepita cells)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--gain-control",
        choices=("none", "unit_rms", "spectral"),
        default="none",
        help="A5: settle-path gain homeostasis (hidden-layer renorm at emit)",
    )
    ap.add_argument(
        "--readout-norm",
        choices=("none", "unit_rms"),
        default="none",
        help="Readout rung: unit-RMS-normalize the output-weight pseudo-gradient",
    )
    args = ap.parse_args()
    torch.manual_seed(0)
    train_t, val_t = lmc.load_tokens()
    _, m_val = lmc._val_sets(val_t, CTX)
    cells = [
        (c, w) for c in ("ff", "ff_hybrid", "pepita", "epc_thermo") for w in WIDTHS
    ]
    if args.cells:
        want = {tuple(c.split(":")) for c in args.cells}
        cells = [(c, w) for c, w in cells if (c, str(w)) in want]
    lrs = args.lrs or [LR.get("pepita", 5e-4)]
    print(f"{'credit':>12} {'width':>5} {'update':>16} {'lr':>9}  val_ppl  act_std")
    for credit, width in cells:
        for lr in lrs:
            torch.manual_seed(0)
            system = build(
                credit,
                width,
                update=args.update,
                lr=lr,
                fscale=args.fscale,
                learned_fb=args.learned_feedback,
                gain_control=args.gain_control,
            )
            system.geometry.to(lmc.DEVICE)  # type: ignore[attr-defined]
            if args.readout_norm != "none":
                system.update = _ReadoutNormProxy(system.update)  # type: ignore[attr-defined]
            r = run_arm(system, train_t, m_val, seed=args.seed)
            val = lmc._eval(system, m_val, "mlp")
            print(
                f"{credit:>12} {width:>5} {args.update:>16} {lr:>9}  "
                f"{val['val_ppl']:>7}  {r['act_std']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
