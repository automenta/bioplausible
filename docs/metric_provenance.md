# Metric Provenance Table (R7 probe #4 — imp-46)

> Census of every metric that can influence a claim, with the state it is
> computed on. Machine counterpart: `tests/property/test_metric_provenance.py`
> (closed pipeline schema + strict free-read claim extraction). A schema change
> must update both.
>
> **Rule (hard):** any metric computed on a nudged/target-conditioned state is
> named `nudged_*`/`*_fit_*` and is excluded from learning claims, adaptation
> claims, resource-efficiency claims, frontier ranking, and discovery locks
> unless a pre-registration says otherwise. The evidence chain consumes only
> `free_*` (post-update target-free settle) claim fields.

## Verdict legend

| Verdict | Meaning |
|---|---|
| **claim-grade** | Target-free post-update (or held-out) state; may carry learning/adaptation claims |
| **diagnostic** | Honest about what it measures but supervised/fit/state-conditioned; never a claim metric |
| **quarantined** | Target-conditioned fit; metadata/diagnostic only (imp-20/imp-46) |

## Canonical pipeline (`core/pipeline.py::run_train_step`)

Schema is closed (`METRIC_SCHEMA`); a bare `accuracy` key must never reappear.

| Metric | Computed state | Target-free | Labels | Nudged/target-conditioned | Verdict |
|---|---|---|---|---|---|
| `free_loss` | post-update forward + target-free settle, CE | yes | scoring only | no | **claim-grade** |
| `free_accuracy` | same state, argmax | yes | scoring only | no | **claim-grade** |
| `free_energy` | same state, `compute_energy` | yes | no | no | **claim-grade** (state variable; may be negative) |
| `loss` | output-phase CE (nudged state for contrastive credits) | no | yes | yes | diagnostic (training loss) |
| `energy` | output-phase state energy | no | no | yes for contrastive | diagnostic |
| `nudged_fit_accuracy` | output-phase (nudged) fit, argmax | no | yes | **yes** | **quarantined** |

## Campaign episode path (`core/campaign/evaluation.py`)

| Metric | Emitter | Computed state | Target-free | Claim consumer | Verdict |
|---|---|---|---|---|---|
| `task_loss` | `evaluate_episode` | strict read of `free_loss` | yes | FrontierRecord → discovery/pareto/fidelity/replication | **claim-grade** |
| `task_accuracy` | `evaluate_episode` | strict read of `free_accuracy` | yes | FrontierRecord → all campaign claims | **claim-grade** |
| `state_energy_j` | `_episode_resources` | strict read of `free_energy` | yes | resource records (state variable, not consumption) | **claim-grade** (state) |
| `compute`/`energy`/`memory`/`latency`/`plastic_state_capacity` | `_episode_resources` | work-derived (MACs incl. documented 2× backward), param count, wall clock, ψ dims | n/a (no labels) | ResourceUsage Pareto axes | **claim-grade** post imp-45; pre-fix r5b_b/r51c records quarantined for resource claims |
| `rho_jacobian` (windowed growth), `settling_time`, `basin_stability` | StabilityGuard probe | activity transition on free system | yes | stability fields of FrontierRecord | diagnostic — axis saturated (imp-36: one distinct tuple across 480 records) |
| `nudged_fit_accuracy` (metadata) | `evaluate_episode` | nudged-settle fit | **no** | none (comparison only) | **quarantined** |

## Benchmark suites (`experiments/joint/`)

| Metric | Suite | Computed state | Target-free | Verdict |
|---|---|---|---|---|
| `final_accuracy` | L1 adaptation, L2 compute | `model.eval()` + `no_grad()` on fresh task batches | yes | **claim-grade** (M-arm contrast gated by imp-43) |
| `adaptation_time` (L1), `migration_time` (L3.5) | threshold = first epoch with training CE < 0.5 | training-batch CE on target-free forward logits — label-conditioned threshold, state-clean | no (labels set the bar) | diagnostic — measures training-fit crossing, not held-out adaptation; any adaptation *claim* must re-define the trigger on a target-free probe metric first |
| `phase_a/b_final_loss`, `a0/a1_losses` | L1/L3.5 | training CE | no | diagnostic |
| `theta_change` | L3.5 | direct ‖θ_after − θ_before‖ | n/a | **claim-grade** (exact invariance check) |
| `active_routes`, `gate_entropy` | L2 | gate logits (`no_grad`) | yes | **claim-grade** — the one genuine M-axis discriminator |
| `dense_flops`/`effective_flops`/`flops_reduction` | L2 | structural estimate | n/a | **claim-grade** (structural) |
| `pre_damage_accuracy`/`initial_accuracy`/`final_accuracy` (L3) | eval + `no_grad` | yes | **claim-grade** (imp-42 fixed degenerate pre-damage) |
| `recovery_ratio` | L3 | ratio of two target-free evals | yes | **claim-grade** (same semantics both sides) |
| `recovery_accuracies` | L3 | training-batch argmax during recovery | no | diagnostic |
| Z3 `_eval_task_accuracy`, `_probe_accuracy`, adaptation curve | Z3 | fresh batches / fixed held-out probe, `eval()` + `no_grad()` | yes | **claim-grade** — flagship-ready *pending* imp-43 engagement lock + imp-52 positive control |
| Z3 episode losses | Z3 | training CE | no | diagnostic |
| Z3 θ-invariance | `ThetaInvarianceAudit` | exact param comparison | n/a | **claim-grade** |

## Trainers / other emitters

| Metric | Emitter | Computed state | Verdict |
|---|---|---|---|
| `train_acc` | `SystemTrainer` epoch | **fixed (imp-46):** now free-first (`free_accuracy`); was nudged-fit | **claim-grade** |
| `train_loss`/`train_energy` | `SystemTrainer` epoch | output-phase (nudged) diagnostics | diagnostic |
| `val_loss`/`val_acc` | `SystemTrainer.validate` | `system.forward` + `no_grad`, CE/argmax | **claim-grade** |
| `train_acc` | `DistributedSystemTrainer` epoch | **defect found + fixed (imp-46):** read `free_state.metrics["accuracy"]` which is never written — silent constant 0.0; now computed from free-state activations (`_accuracy_from_state`) | **claim-grade** |
| `nudged_fit_accuracy` | `core/continual/training.py` step dicts | renamed from `accuracy` (output-phase fit) | **quarantined** |
| `accuracy_matrix`, backward/forward transfer | `core/continual/runner.py` | test-loader `eval()` + `no_grad` | **claim-grade** |
| `train_accuracy`/`val_accuracy` | `domains/trainer.py` | plain autograd forward (`no_grad` on val) | **claim-grade** |
| `accuracy` (BPTT fallback, TileAlgorithm BPTT) | `core/trainer.py`, `core/local_learning/builder.py` | plain target-free forward argmax | **claim-grade** (pre-update forward fit) |
| kernel `accuracy` | `acceleration/kernels.py` (EqProp) | free-equilibrium logits (pre-update) | diagnostic (parity) |
| `best_accuracy`/`final_accuracy` | `experiments/joint/memory_wall.py` | eval forward | **claim-grade** |
| `final_acc`/`accuracy` consumers (knowledge, hyperopt, p2p, execution) | various | consume the target-free emitters above | **claim-grade** by inheritance |

## Quarantine status (claim chain)

- `nudged_fit_accuracy`: metadata/diagnostic only — excluded from all claims (locked).
- Pre-fix r5b_b/r51c resource records: quarantined for **resource claims** (imp-45).
- Stability axis fields: saturated — not discriminating (imp-36 data); quarantined for stability claims.
- L1/L3 M-arm means: identical across plasticity primitives (imp-43) — no M-axis claim until ψ engagement locks pass.
- Task-loss attribution: **provisional → upgraded to claim-grade by this census** for the campaign chain (strict free reads, closed schema, locks pinned). Suite-level training-loss-threshold metrics (adaptation/migration time) remain diagnostics.
