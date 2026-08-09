# EXPERIMENT_PLAN8.md — Credit Assignment Reality Plan  
**Status:** Replacement/addendum to `EXPERIMENT_PLAN7.md`.  
**Scope:** Supersedes Plan 7 Phase 4+ while preserving Plan 7 Session 1 fixes as the baseline.  
**Goal:** Turn the current empirical findings into concrete codebase improvements: fix real defects, add honest diagnostics, stop spending time on mathematically misleading fixes, and produce credible compute-matched results for the families that actually work.

---

## 1. Executive Summary

Plan 7 established the key scientific fact:

> **1-layer EqProp works. Multi-layer vanilla EqProp does not learn on digits/MNIST under the current contrastive rule. FA / Target Prop / Predictive Coding do learn deep architectures.**

The original Phase 4 hypothesis was that multi-layer EqProp could be rescued by:
1. Initializing recurrent weights non-zero.
2. Adding per-layer β.
3. Logging layer-wise energy gaps.

After evaluation, the correct interpretation is:

- **Non-zero `W_rec` initialization** is reasonable but unlikely to solve deep credit assignment.
- **Per-layer β**, if implemented as a different denominator in the contrastive update, is not truly β. It is a per-layer learning-rate/update-scale hack. It may help empirically but does not solve the structural EqProp problem.
- **Logging layer-wise gaps** is essential, but it is a diagnostic, not a fix.

Therefore, this plan replaces the previous Phase 4 with a more honest and effective program:

1. **Instrument the contrastive rule properly.**
2. **Separate true EqProp β from per-layer update scaling.**
3. **Test principled EqProp salvage variants, especially explicit feedback.**
4. **Quarantine or fix phantom-knob models.**
5. **Run compute-matched parity on the families that already work.**
6. **Update registry, tests, docs, and knowledge base to reflect reality.**

This plan is intentionally conservative: it does not build speculative infrastructure, does not invent new algorithms prematurely, and does not force EqProp to be the flagship if the evidence says otherwise.

---

## 2. Baseline Ground Truth

Retain the following from Plan 7:

| Item | Status |
|---|---:|
| Backprop parity synthetic suite | ✅ Existing |
| Gradient equivalence suite | ✅ Existing |
| Registry metadata audit | ✅ Existing |
| Statistics utilities | ✅ Existing |
| Broad sweep script | ✅ Existing |
| EqProp search-space fix | ✅ Done |
| Phantom `num_layers` root cause for core EqProp | ✅ Fixed |
| Consolidated deep EqProp engine | ✅ Done |
| Fast implicit 1-layer EqProp path | ✅ Done |
| Early abort on epoch budget | ✅ Done |
| Phase 1–3 sweeps | ✅ Complete |

Key empirical findings:

| Family / Model | Result |
|---|---:|
| 1-layer EqProp | Works, especially implicit path |
| Multi-layer vanilla EqProp | Broken / near chance on digits |
| FA / DFA | Strong, up to ~94% |
| Target Prop | Strong, ~80%+ |
| Predictive Coding | Strong, ~80%+ |
| DirectedEP / feedback EqProp | Needs honest evaluation |

---

## 3. Core Diagnosis

The likely failure mode of deep vanilla EqProp is **vanishing contrastive state difference**.

In standard EqProp, the update is approximately:

\[
\Delta W \propto \frac{1}{\beta}
\left(
h^+_{\text{post}} h^{+T}_{\text{pre}}
-
h^-_{\text{post}} h^{-T}_{\text{pre}}
\right)
\]

where:

- \(h^-\) is the free-phase settled state.
- \(h^+\) is the nudged-phase settled state.
- \(\beta\) is the global output nudge strength.

For deep networks, the nudging signal applied at the output must propagate backward through the settled dynamics. If the dynamics are contractive, that signal can decay exponentially. Early hidden layers then satisfy:

\[
h^+ \approx h^-
\]

and therefore:

\[
h^+ h^{+T} - h^- h^{-T} \approx 0
\]

Dividing by a smaller per-layer β does not fix this. If the numerator is zero, scaling the denominator still gives zero. What such a change actually implements is a per-layer update scale, not a true energy-based β.

Therefore, the codebase should distinguish clearly between:

| Concept | Meaning | Where it acts |
|---|---|---|
| `beta` | Global output nudge in the energy | Settling / nudged phase |
| `update_scale` | Multiplier on the computed contrastive update | Optimizer / gradient application |
| `feedback_gain` | Strength of explicit top-down feedback pathway | Dynamics / nudged phase |
| `w_rec_init` | Initialization of recurrent/self-connections | Model construction |

This distinction is one of the central improvements in this plan.

---

## 4. Objectives

### Primary Objectives

1. **Make EqProp diagnostics first-class.**
   - Measure per-layer state deltas.
   - Measure per-layer gradient norms.
   - Measure convergence behavior.
   - Store results in machine-readable form.

2. **Make EqProp hyperparameters mathematically honest.**
   - Keep `beta` global.
   - Replace misleading `beta_scale_by_depth` with `update_scale_by_depth`.
   - Add explicit `w_rec_init` knob.
   - Add explicit feedback knobs where relevant.

3. **Decide the fate of deep EqProp using evidence.**
   - If feedback-based EqProp works, keep it.
   - If it does not, mark deep vanilla EqProp as experimentally broken and stop treating it as a flagship deep-learning rule.

4. **Produce credible compute-matched parity results.**
   - Compare backprop against top bio-plausible families.
   - Use matched architectures, seeds, epochs, and compute budgets.
   - Report confidence intervals and effect sizes.

5. **Improve codebase hygiene.**
   - Remove or quarantine phantom-knob models.
   - Add registry status tags.
   - Add regression tests.
   - Document limitations clearly.

---

## 5. Definition of Done

This plan is complete when all of the following are true:

- [ ] All active search spaces contain no phantom knobs.
- [x] EqProp diagnostics are implemented and tested.
- [x] `beta` and `update_scale` are separated.
- [x] `w_rec_init` is implemented as an explicit knob.
- [ ] DirectedEP / feedback EqProp is honestly evaluated.
- [x] A contrastive profiling script produces per-layer diagnostic reports.
- [ ] Registry status tags or equivalent flags distinguish stable, experimental, and broken models.
- [ ] Known phantom-knob models are either fixed or quarantined.
- [ ] A compute-matched parity runner produces a reproducible report.
- [x] Unit and integration tests cover the new diagnostics and update-scaling behavior.
- [ ] Documentation reflects the actual limitations of deep EqProp.
- [ ] A final decision is recorded: keep, demote, or quarantine deep EqProp variants.

---

## 6. Workstreams

The plan is divided into four tracks:

- **Track A — Instrumentation and correctness**
- **Track B — EqProp autopsy and salvage**
- **Track C — Compute-matched parity for working families**
- **Track D — Registry, testing, documentation, and knowledge capture**

Tracks A and B are the immediate next steps. Track C should begin as soon as diagnostics are available. Track D runs throughout.

---

# Track A — Instrumentation and Correctness

## A1. Add first-class contrastive diagnostics

### Goal

Make it possible to see whether deep EqProp layers are receiving any meaningful nudged/free state difference.

### Files

- `bioplausible/zoo/models/eqprop/_contrastive.py`
- `bioplausible/zoo/models/eqprop/_energy.py`
- `bioplausible/core/trainer.py`, if metrics flattening is needed

### Required Behavior

Modify `_contrastive_step` to optionally collect diagnostics.

Suggested signature change:

```python
def _contrastive_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    layer_list: list[nn.Module],
    beta: float,
    update_scales: list[float] | None = None,
    diagnostics: bool = False,
    use_conj: bool = False,
    feedback_layer_list: list[nn.Module] | None = None,
    recurrent_layer_list: list[nn.Module] | None = None,
) -> dict[str, object]:
```

For each forward layer `i`, record at least:

```python
{
    "layer": i,
    "pre_state_delta_norm": (h_prev_nudge - h_prev_free).norm().item(),
    "post_state_delta_norm": (h_post_nudge - h_post_free).norm().item(),
    "weight_grad_norm": dW.norm().item(),
    "bias_grad_norm": db.norm().item() if db is not None else 0.0,
    "update_scale": scale,
}
```

Also record global information:

```python
{
    "output_state_delta_norm": (nudged_acts[-1] - free_acts[-1]).norm().item(),
    "beta": beta,
    "loss": loss,
    "accuracy": acc,
}
```

### Important Design Rule

Diagnostics should not spam stdout during large sweeps. Use one of:

- `diagnostics=True` only for profiling scripts.
- Log every N steps.
- Store diagnostics in `model._last_contrastive_diagnostics`.
- Return nested metrics only when explicitly enabled.

### Acceptance Criteria

- [ ] `_contrastive_step` can return diagnostics without changing default training behavior.
- [ ] Diagnostic values are finite for a small 2-layer EqProp model.
- [ ] A unit test asserts expected diagnostic keys exist.
- [ ] Diagnostics can be enabled via config extra, e.g. `contrastive_diagnostics: true`.

---

## A2. Separate true β from update scaling

### Goal

Avoid mathematically misleading “per-layer β” changes.

### Required Change

Keep `beta` as the single nudge parameter used during the nudged settle.

Introduce separate update scaling:

```python
update_scale = float(config.extra.get("update_scale", 1.0))
update_scale_by_depth = float(config.extra.get("update_scale_by_depth", 1.0))

update_scales = [
    update_scale * (update_scale_by_depth ** i)
    for i in range(len(self.layers))
]
```

Then in `_contrastive_step`, apply it after the EqProp gradient has been computed:

```python
scale = update_scales[i] if update_scales is not None else 1.0

dW = (prod_nudge - prod_free) / beta
dW = dW * scale / batch_size

db = (h_post_nudge - h_post_free).sum(0) / beta
db = db * scale / batch_size
```

### Files

- `bioplausible/zoo/models/eqprop/_contrastive.py`
- `bioplausible/zoo/models/eqprop/_energy.py`
- `bioplausible/hyperopt/search_space.py`

### Search Space Update

Replace or rename any `beta_scale_by_depth` knob.

Use:

```python
"update_scale": (1e-2, 1e1, "log"),
"update_scale_by_depth": (1e-1, 1e1, "log"),
```

Do **not** advertise this as β.

### Acceptance Criteria

- [ ] `beta` remains global and is used in `_run_free_nudged`.
- [ ] `update_scales` only multiplies the computed update.
- [ ] A test verifies that changing `update_scale` scales the gradient norm approximately linearly.
- [ ] A test verifies that changing `beta` affects the nudged/free state difference, not merely the denominator.

---

## A3. Make recurrent initialization an explicit knob

### Goal

Test the `W_rec` hypothesis honestly instead of hardcoding a change.

### Required Change

In `EquilibriumMLP._init_weights`, initialize `W_rec` according to config:

```python
w_rec_init = self.config.extra.get("w_rec_init", "zero")
w_rec_gain = float(self.config.extra.get("w_rec_gain", 0.1))

if w_rec_init == "xavier":
    nn.init.xavier_uniform_(actual.weight, gain=w_rec_gain)
else:
    nn.init.zeros_(actual.weight)
```

### Search Space

```python
"w_rec_init": ["zero", "xavier"],
"w_rec_gain": (1e-3, 1e0, "log"),
```

### Acceptance Criteria

- [ ] `w_rec_init="zero"` produces all-zero `W_rec` weights.
- [ ] `w_rec_init="xavier"` produces non-zero `W_rec` weights.
- [ ] The knob is visible to the construction/search-space audit.
- [ ] A unit test checks both modes.

---

# Track B — EqProp Autopsy and Salvage

## B1. Create an EqProp profiling script

### Goal

Provide a single command that reveals whether deep EqProp is suffering from vanishing contrastive signal.

### New Script

```text
scripts/contrastive_profile.py
```

### Required Arguments

```bash
--model
--task
--num-layers
--hidden-dim
--beta
--learning-rate
--epochs
--batch-size
--seed
--device
--output-dir
```

### Required Output

For each profiled run, write:

```text
runs/contrastive_profile/<timestamp>/<model>_<depth>_<task>/
    diagnostics.json
    summary.md
```

The summary should include a table like:

| Step | Layer | Pre Δ Norm | Post Δ Norm | Grad Norm | Update Scale |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | ... | ... | ... | 1.0 |
| 0 | 1 | ... | ... | ... | 1.0 |
| 0 | 2 | ... | ... | ... | 1.0 |

### Acceptance Criteria

- [ ] Script runs on CPU for a tiny config.
- [ ] Script runs on GPU for normal configs.
- [ ] Output JSON is loadable.
- [ ] Summary markdown is human-readable.
- [ ] The script can compare at least two models in separate runs.

---

## B2. Run the EqProp autopsy

### Models to Profile

At minimum:

```text
eqprop
directed_ep
momentum_equilibrium
sparse_equilibrium
finite_nudge_ep
lazy_eqprop
```

### Depths

```text
1, 2, 3, 4
```

### Tasks

```text
digits
mnist
```

### Diagnostic Questions

Answer these explicitly:

1. Does the output layer have a non-zero nudged/free state difference?
2. Does the last hidden layer have a non-zero difference?
3. Do early hidden layers have near-zero differences?
4. Are gradient norms near zero in early layers?
5. Does non-zero `W_rec` initialization change early-layer state deltas?
6. Does `update_scale_by_depth` improve accuracy, or only amplify noise?
7. Does DirectedEP / feedback materially improve deep-layer state deltas?

### Gate G1 — Vanishing Signal

Declare vanishing contrastive signal if, for depth ≥ 3:

```text
early_layer_post_state_delta_norm / output_state_delta_norm < 1e-3
```

and:

```text
early_layer_grad_norm / output_layer_grad_norm < 1e-3
```

for the majority of early training steps.

If G1 triggers, the codebase should record:

```text
Finding: Deep vanilla EqProp suffers from vanishing contrastive signal under current dynamics.
```

---

## B3. Evaluate feedback-based EqProp seriously

### Rationale

If deep EqProp lacks an error-feedback pathway, DirectedEP or a similar explicit feedback variant is the most principled salvage route.

### Required Work

Ensure `DirectedEP` / `variant="feedback"` has:

- [ ] Trainable feedback layers.
- [ ] Feedback gradients recorded in diagnostics.
- [ ] Honest feedback knobs.
- [ ] No phantom knobs.
- [ ] Correct behavior for `num_layers > 1`.

### Suggested Feedback Knobs

```python
"feedback_gain": (1e-2, 1e1, "log"),
"feedback_init_gain": (1e-3, 1e0, "log"),
```

### Important Question

Does feedback actually change hidden-layer state deltas?

If feedback improves accuracy but hidden-layer deltas remain near zero, treat the result with suspicion and inspect whether learning is occurring through an unintended shortcut.

### Gate G2 — Feedback Salvage

A feedback EqProp variant is considered salvaged for deep tasks if:

- [ ] It reaches > 50% accuracy on digits with depth ≥ 3 after 5 epochs.
- [ ] Or > 75% accuracy on MNIST with depth ≥ 3 after 10 epochs.
- [ ] With at least 3 seeds.
- [ ] And diagnostics show non-zero early-layer state deltas or gradient norms.

If G2 fails, deep EqProp should be demoted or quarantined.

---

## B4. Do not overfit to misleading knobs

### Rule

If a variant only works after adding a knob that is mathematically inconsistent with the original algorithm, document the knob as an optimization hack.

Examples:

- Per-layer update scaling can be useful, but it is not per-layer β.
- Adding external feedback may be biologically plausible, but it changes the credit assignment mechanism.
- Initializing recurrent weights may help optimization, but it does not prove the original rule scales.

### Required Documentation

For every successful knob, record:

```text
Knob:
Mechanism:
Is it algorithmically faithful?
Does it change the energy function?
Does it change only optimization?
Evidence:
```

---

# Track C — Compute-Matched Parity for Working Families

## C1. Select the parity portfolio

Based on Plan 7 results, the initial portfolio should be:

```text
backprop_mlp
dfa_deep
direct_feedback_alignment_eqprop
diff_target_prop
fabricpc_graph_pcn
directed_ep  # only if it passes Gate G2
```

Do not include known-broken deep vanilla EqProp in final parity unless explicitly requested for negative-result comparison.

---

## C2. Implement or harden the parity runner

### File

```text
bioplausible/validation/backprop_parity.py
```

If this already exists, extend it. If not, create it.

### Required Capabilities

The parity runner must:

- [ ] Accept a list of families or model names.
- [ ] Use the same dataset and split.
- [ ] Use the same architecture depth and width where possible.
- [ ] Match parameter counts within a tolerance, e.g. ±10%.
- [ ] Use the same seed set.
- [ ] Use the same epoch budget.
- [ ] Use the same max wall-clock budget per epoch.
- [ ] Record peak memory and epoch time.
- [ ] Use existing statistics utilities for CIs and effect sizes.
- [ ] Emit JSON and markdown reports.

### Suggested CLI

```bash
uv run python -m bioplausible.validation.backprop_parity \
  --task digits \
  --arch mlp \
  --depths 2,3 \
  --hidden-dims 256,512 \
  --seeds 5 \
  --epochs 5 \
  --device cuda \
  --families backprop,fa,target_prop,predictive_coding,eqprop_feedback \
  --output-dir runs/parity/digits_mlp
```

### Required Report Contents

For each model/family:

```text
model
family
depth
hidden_dim
params
epochs
seed_count
mean_accuracy
accuracy_ci95
mean_loss
mean_epoch_time
peak_memory
status
notes
```

For each comparison against backprop:

```text
delta_accuracy
cohen_d
cliff_delta
bootstrap_p
compute_matched
```

### Acceptance Criteria

- [ ] A tiny CPU smoke test passes.
- [ ] A real GPU run produces a markdown report.
- [ ] The report includes confidence intervals.
- [ ] The runner fails loudly if parameter budgets are not matched.

---

## C3. Tune before comparing

Do not compare models at arbitrary defaults.

For each family:

1. Run a small sweep.
2. Select the best non-broken config under the same compute budget.
3. Then run final seeded parity.

### Suggested Tuning Budget

For each family:

```text
Probes: 10–30
Epochs: 2–3
Task: digits
Max params: 100k
Max epoch time: 30s
```

Then final evaluation:

```text
Seeds: 5–10
Epochs: 5–10
Task: digits or MNIST
```

### Rule

The final parity report must include both:

- The tuned hyperparameters.
- The final seeded results.

---

## C4. Define parity success criteria

Use different tiers:

### Tier 1 — Strong Parity

```text
Bio model accuracy within 2% absolute of backprop,
with matched params and compute.
```

### Tier 2 — Acceptable Parity

```text
Bio model accuracy within 5% absolute of backprop,
and has memory/time/locality advantages.
```

### Tier 3 — Negative Result

```text
Bio model accuracy more than 5% below backprop,
with no compensating advantage.
```

Record the tier for each family.

---

# Track D — Registry, Testing, Documentation, and Knowledge Capture

## D1. Add model status metadata

### Goal

Prevent sweeps from wasting time on known-broken models.

### Implementation

If registry metadata supports a `status` field, use:

```text
stable
experimental
broken
deprecated
```

If not, use tags:

```text
status:stable
status:experimental
status:broken
status:deprecated
```

### Suggested Initial Status

| Model / Family | Suggested Status |
|---|---|
| `backprop_mlp` | stable |
| 1-layer `eqprop_mlp` implicit path | stable |
| deep vanilla `eqprop` | experimental or broken pending Gate G2 |
| `directed_ep` | experimental |
| `dfa_deep` | stable candidate |
| `diff_target_prop` | stable candidate |
| `fabricpc_graph_pcn` | stable candidate |
| phantom-knob conv/graph/cube EqProp variants | broken or deprecated until fixed |

### Acceptance Criteria

- [ ] Sweep can exclude broken models by default.
- [ ] Sweep can include broken models with `--include-broken`.
- [ ] Registry audit reports status metadata.

---

## D2. Resolve phantom-knob models

Plan 7 flagged several models with phantom `num_layers`, including:

```text
graph_eqprop
conv_eqprop
modern_conv_eqprop
neural_cube
eqprop_diffusion
direct_feedback_alignment_eqprop
dfa_deep
equilibrium_alignment
hebbian_chain
deep_hebbian
hebbian_3d
```

For each model, choose one of three actions:

### Option 1 — Fix

Thread `num_layers` into `hidden_dims` and use canonical construction.

### Option 2 — Quarantine

Mark as `status:broken` and exclude from default sweeps.

### Option 3 — Deprecate

Move to legacy/deprecated if not scientifically important.

### Required Artifact

Create a table in `RESEARCH.md` or `docs/phantom_knob_audit.md`:

| Model | Phantom Knob | Action | Evidence | Test |
|---|---|---|---|---|

### Acceptance Criteria

- [ ] No active default sweep model has phantom knobs.
- [ ] Regression test fails if a sampled config knob is silently dropped.
- [ ] All quarantined models are clearly documented.

---

## D3. Add targeted tests

Add or extend the following tests:

### EqProp Diagnostics

```text
tests/unit/models/test_eqprop_diagnostics.py
```

Checks:

- [ ] Diagnostic keys exist.
- [ ] Diagnostic values are finite.
- [ ] Output-layer delta is non-zero when beta > 0.

### Update Scaling

```text
tests/unit/models/test_eqprop_update_scale.py
```

Checks:

- [ ] `update_scale=2.0` approximately doubles update norm.
- [ ] `update_scale_by_depth` creates expected scale list.
- [ ] β remains global.

### Recurrent Initialization

```text
tests/unit/models/test_eqprop_wrec_init.py
```

Checks:

- [ ] `zero` mode yields zeros.
- [ ] `xavier` mode yields non-zeros.

### Feedback Pathway

```text
tests/unit/models/test_directed_ep_feedback.py
```

Checks:

- [ ] Feedback layers exist for depth ≥ 2.
- [ ] Feedback layers receive gradients.
- [ ] Feedback layers are included in diagnostics.

### Registry Status

```text
tests/unit/validation/test_registry_status.py
```

Checks:

- [ ] Every registered model has status or status tag.
- [ ] Broken models are excluded by default sweep filters.

### Parity Smoke Test

```text
tests/unit/validation/test_backprop_parity_smoke.py
```

Checks:

- [ ] Tiny parity run completes on CPU.
- [ ] Report files are created.
- [ ] Statistics utilities are invoked.

---

## D4. Improve sweep result provenance

Every sweep result should record:

```text
git_sha
timestamp
python_version
torch_version
device
task
model
family
config
seed
accuracy
loss
epoch_time
peak_memory
status
defects
diagnostics_summary
```

### Files

- `scripts/broad_sweep.py`
- `bioplausible/validation/statistics.py`
- Optional new file: `bioplausible/validation/results.py`

### Acceptance Criteria

- [ ] Sweep JSON includes git SHA.
- [ ] Sweep JSON includes wall-clock and memory metrics.
- [ ] Failed probes include defect reason.
- [ ] A summarizer can produce a top-N table from sweep JSON.

---

## D5. Write the scientific limitation document

Create:

```text
docs/eqprop_deep_limitation.md
```

Contents:

1. Summary of Plan 7 findings.
2. Explanation of contrastive state difference.
3. Why per-layer β is not a true fix.
4. Diagnostic evidence.
5. Which EqProp variants remain viable.
6. Which families are recommended for deep credit assignment.

This document should be written even if EqProp is later rescued, because it records the boundary conditions.

---

# 7. Execution Order

This plan is designed to be executed in small, verifiable increments.

---

## Session 2 — Instrumentation and Honest Knobs

### Goals

- Add diagnostics.
- Separate β from update scaling.
- Add `w_rec_init`.
- Add unit tests.

### Tasks

```text
A1
A2
A3
D3 partial
```

### Exit Criteria

```bash
uv run pytest tests/unit/models/test_eqprop_diagnostics.py -q --no-cov
uv run pytest tests/unit/models/test_eqprop_update_scale.py -q --no-cov
uv run pytest tests/unit/models/test_eqprop_wrec_init.py -q --no-cov
```

All pass.

---

## Session 3 — EqProp Autopsy

### Goals

- Build profiling script.
- Run diagnostics on EqProp variants.
- Determine whether deep EqProp has vanishing contrastive signal.
- Evaluate feedback salvage.

### Tasks

```text
B1
B2
B3
```

### Exit Criteria

The following reports exist:

```text
runs/contrastive_profile/eqprop_depth3_digits/
runs/contrastive_profile/directed_ep_depth3_digits/
```

And a written answer exists for:

```text
Does deep vanilla EqProp have vanishing contrastive state deltas?
Does DirectedEP materially fix the issue?
```

---

## Session 4 — Registry and Phantom Cleanup

### Goals

- Add status metadata.
- Quarantine or fix phantom-knob models.
- Prevent default sweeps from running known-broken models.

### Tasks

```text
D1
D2
D4 partial
```

### Exit Criteria

```bash
uv run pytest tests/unit/validation/test_registry_audit.py -q --no-cov
uv run pytest tests/unit/experiment/test_config_knobs.py -q --no-cov
uv run pytest tests/unit/validation/test_registry_status.py -q --no-cov
```

All pass.

---

## Session 5 — Compute-Matched Parity

### Goals

- Implement or harden parity runner.
- Tune top families.
- Run seeded parity.
- Produce final report.

### Tasks

```text
C1
C2
C3
C4
```

### Exit Criteria

A report exists at:

```text
runs/parity/digits_mlp/report.md
runs/parity/digits_mlp/results.json
```

It includes:

- [ ] Backprop baseline.
- [ ] FA/DFA.
- [ ] Target Prop.
- [ ] Predictive Coding.
- [ ] DirectedEP, if it passed Gate G2.
- [ ] Confidence intervals.
- [ ] Compute metrics.
- [ ] Parity tier.

---

## Session 6 — Documentation and Decision Capture

### Goals

- Write limitation document.
- Update `RESEARCH.md`.
- Record final model statuses.
- Add knowledge-base entries.

### Tasks

```text
D5
B4
D4 final
```

### Exit Criteria

The repository contains:

```text
docs/eqprop_deep_limitation.md
docs/phantom_knob_audit.md
RESEARCH.md updated
runs/parity/*/report.md
```

---

# 8. Decision Gates

This plan uses explicit gates to avoid endless speculation.

---

## Gate G1 — Vanishing Contrastive Signal

Trigger if, for depth ≥ 3:

```text
early_layer_state_delta / output_state_delta < 1e-3
```

and:

```text
early_layer_grad_norm / output_layer_grad_norm < 1e-3
```

### Action if triggered

Record:

```text
Deep vanilla EqProp has vanishing contrastive signal.
```

Do not spend further time on ordinary hyperparameter tweaks unless they change this diagnostic.

---

## Gate G2 — Feedback Salvage

A feedback EqProp variant is kept if it meets:

```text
> 50% accuracy on digits, depth >= 3, 5 epochs, 3 seeds
```

or:

```text
> 75% accuracy on MNIST, depth >= 3, 10 epochs, 3 seeds
```

and diagnostics show non-zero early-layer signal.

### Action if passed

- Mark feedback EqProp as experimental but viable.
- Include it in parity.
- Document the required feedback mechanism.

### Action if failed

- Mark deep EqProp as broken for deep credit assignment.
- Remove it from flagship comparisons.
- Keep 1-layer EqProp for memory/implicit-path experiments.
- Shift focus to FA, Target Prop, Predictive Coding, or hybrids.

---

## Gate G3 — Compute-Matched Parity

After parity runs, classify each family.

### Strong Parity

```text
within 2% absolute of backprop
```

### Acceptable Parity

```text
within 5% absolute of backprop
and has memory/time/locality advantage
```

### Negative Result

```text
more than 5% below backprop
no compensating advantage
```

### Action

- Strong or acceptable families become the primary bio-plausible candidates.
- Negative-result families remain in the registry but are not flagships.
- Results are written into `RESEARCH.md`.

---

# 9. Verification Commands

Use these commands to verify the plan implementation.

## Core validation floor

```bash
uv run pytest tests/unit/validation/ tests/integration/test_gradient_equivalence.py -q --no-cov
```

## EqProp-specific tests

```bash
uv run pytest tests/unit/models/test_eqprop*.py -q --no-cov
```

## Config knob tests

```bash
uv run pytest tests/unit/experiment/test_config_knobs.py -q --no-cov
```

## Contrastive profiling

```bash
uv run python scripts/contrastive_profile.py \
  --model eqprop \
  --task digits \
  --num-layers 3 \
  --hidden-dim 256 \
  --epochs 1 \
  --device cuda
```

```bash
uv run python scripts/contrastive_profile.py \
  --model directed_ep \
  --task digits \
  --num-layers 3 \
  --hidden-dim 256 \
  --epochs 1 \
  --device cuda
```

## EqProp sweep after diagnostics

```bash
uv run python scripts/broad_sweep.py \
  --families eqprop \
  --probes-per-rule 3 \
  --epochs 2 \
  --device cuda \
  --max-params 32000 \
  --max-epoch-time 30 \
  --task digits
```

## Compute-matched parity

```bash
uv run python -m bioplausible.validation.backprop_parity \
  --task digits \
  --arch mlp \
  --depths 2,3 \
  --hidden-dims 256,512 \
  --seeds 5 \
  --epochs 5 \
  --device cuda \
  --families backprop,fa,target_prop,predictive_coding,eqprop_feedback \
  --output-dir runs/parity/digits_mlp
```

---

# 10. Explicit Non-Goals

To keep this plan effective, the following are intentionally excluded until they become bottlenecks.

## Not now

- Building a full AutoScientist campaign.
- Building a large analysis/plotting toolkit.
- Fixing EquiTile unless it becomes a flagship candidate.
- Adding speculative new bio-plausible algorithms.
- Implementing Progressive Locality unless EqProp diagnostics and parity results demand it.
- Expanding gradient equivalence gating to every propagator before top-family parity is complete.
- Creating a global seed manager unless reproducibility tests begin failing.
- Packaging new CLI binaries solely for spec compliance.

## Never in this plan

- Calling per-layer update scaling “β” if it is not part of the energy nudging term.
- Running known-broken models in default sweeps without a clear diagnostic reason.
- Committing infrastructure-only changes that do not improve diagnostics, correctness, or results.
- Declaring an algorithm fundamentally dead without diagnostics and honest tuning.

---

# 11. Expected Outcomes

If this plan is executed fully, the codebase will have:

1. **Honest EqProp semantics**
   - β is global.
   - Update scaling is explicit.
   - Recurrent initialization is explicit.
   - Feedback pathways are explicit.

2. **Deep EqProp evidence**
   - Per-layer state deltas.
   - Per-layer gradient norms.
   - Clear decision on whether deep EqProp is salvageable.

3. **Reduced wasted compute**
   - Broken models quarantined.
   - Phantom knobs detected.
   - Early abort preserved.
   - Default sweeps focus on viable models.

4. **Credible comparisons**
   - Compute-matched parity.
   - Bootstrap confidence intervals.
   - Effect sizes.
   - Memory/time metrics.

5. **Durable documentation**
   - EqProp limitation document.
   - Phantom-knob audit.
   - Updated research priorities.
   - Knowledge-base entries.

---

# 12. Final Decision Policy

After executing this plan, one of three outcomes should be recorded.

---

## Outcome 1 — Deep EqProp Rescued

Conditions:

- Feedback or another principled variant passes Gate G2.
- Diagnostics show non-zero deep-layer contrastive signal.
- Parity performance is competitive.

Action:

- Keep EqProp as a viable family.
- Clearly document the required feedback mechanism.
- Include the successful variant in flagship parity.

---

## Outcome 2 — Deep EqProp Limited but Useful

Conditions:

- 1-layer EqProp remains strong.
- Deep EqProp fails beyond 1–2 layers.
- Feedback variants help only slightly.

Action:

- Keep EqProp for shallow/memory-advantage experiments.
- Mark deep vanilla EqProp as broken or experimental.
- Use FA / Target Prop / Predictive Coding for deep experiments.

---

## Outcome 3 — Deep EqProp Quarantined

Conditions:

- Diagnostics confirm vanishing contrastive signal.
- Feedback variants fail Gate G2.
- Compute-matched parity shows no competitive path.

Action:

- Mark deep EqProp variants as broken/deprecated.
- Exclude them from default sweeps.
- Preserve tests and docs as a negative result.
- Focus the codebase on families with demonstrated deep credit assignment.

---

# 13. Minimal Patch Summary

If only the smallest code changes are allowed, implement these first:

## 1. `_contrastive.py`

Add:

```python
update_scales: list[float] | None = None
diagnostics: bool = False
```

Use:

```python
scale = update_scales[i] if update_scales is not None else 1.0
dW = ((prod_nudge - prod_free) / beta) * scale / batch_size
```

Record per-layer diagnostics.

---

## 2. `_energy.py`

Add:

```python
update_scale = float(cfg.extra.get("update_scale", 1.0))
update_scale_by_depth = float(cfg.extra.get("update_scale_by_depth", 1.0))
diagnostics = bool(cfg.extra.get("contrastive_diagnostics", False))

update_scales = [
    update_scale * (update_scale_by_depth ** i)
    for i in range(len(self.layers))
]
```

Pass `update_scales` and `diagnostics` to `_contrastive_step`.

Add explicit `w_rec_init` handling.

---

## 3. `search_space.py`

In the `eqprop` rule space, add:

```python
"update_scale": (1e-2, 1e1, "log"),
"update_scale_by_depth": (1e-1, 1e1, "log"),
"w_rec_init": ["zero", "xavier"],
"w_rec_gain": (1e-3, 1e0, "log"),
```

Remove or rename any misleading `beta_scale_by_depth`.

---

## 4. `scripts/contrastive_profile.py`

Create a profiling script that enables diagnostics and writes JSON/markdown reports.

---

## 5. Registry status

Mark known broken models as:

```text
status:broken
```

or equivalent tag.

---

# 14. Replacement Statement

This document replaces the forward-looking parts of `EXPERIMENT_PLAN7.md`, specifically:

- Phase 4: Debug Multi-Layer EqProp.
- Phase 5: Real-Task Compute-Matched Parity.
- The deferred-item priority list where it conflicts with the new gates.

Plan 7 Session 1 remains valid and should be preserved as the baseline.

The new priority hierarchy is:

1. Diagnostics and mathematical honesty.
2. Evidence-based EqProp salvage or quarantine.
3. Compute-matched parity for working families.
4. Registry/test/documentation hardening.
5. Deferred speculative items only when they become bottlenecks.

---

# 15. Execution Log

Append-only log of progress made against this plan. Newest entry last.

## 15.1 Session: Track A + B1 + D3 (Instrumentation, Honest Knobs, Profiler)

**Date:** 2026-08-09

**Tasks covered:** A1, A2, A3, B1, D3 (partial)

### Progress Made

#### A1 — First-class contrastive diagnostics (`_contrastive.py`)

- `_contrastive_step` accepts `update_scales: list[float] | None` and
  `diagnostics: bool = False` and returns `dict[str, object]`.
- When `diagnostics=True`, the result includes `layer_diagnostics`
  (one entry per forward layer with `layer`, `pre_state_delta_norm`,
  `post_state_delta_norm`, `weight_grad_norm`, `bias_grad_norm`,
  `update_scale`) and `global_diagnostics` (`output_state_delta_norm`,
  `beta`, `loss`, `accuracy`).
- Diagnostics are opt-in only: default training behavior and the returned
  `{loss, accuracy}` dict are unchanged when the flag is off.
- Added `_compute_layer_diagnostics` (module-level, bundled state tuple to
  respect the PLR0913 argument cap).
- `diagnostics` is enabled via `config.extra["contrastive_diagnostics"]`, so it
  is reachable from a sweep config (acceptance criterion A1).

#### A2 — Separate true β from update scaling (`_energy.py`, `_contrastive.py`, `search_space.py`)

- `beta` remains the single global nudge used by `_run_free_nudged`; it is
  never divided per-layer.
- New knobs: `update_scale` (global multiplier on the computed contrastive
  update) and `update_scale_by_depth` (geometric depth factor). `train_step`
  builds `update_scales = [update_scale * update_scale_by_depth**i ...]` per
  forward layer and passes them into `_contrastive_step`, which applies the
  scale *after* the `(prod_nudge - prod_free) / beta` EqProp gradient and the
  batch-size normalization.
- `RULE_SPACES["eqprop"]` gains `update_scale`, `update_scale_by_depth`,
  `w_rec_init`, `w_rec_gain`; no `beta_scale_by_depth` existed in code so
  nothing misleading was renamed (Plan 7 only mentioned it in prose).
- Regression tests confirm `update_scale=2.0` scales the layer gradient
  norm ≈2× (via direct `_contrastive_step` exercise), `update_scale_by_depth`
  yields the geometric scale list `[1.0, 2.0, 4.0]` for 3 layers, and β stays
  global.

#### A3 — Explicit recurrent initialization (`_energy.py`)

- `w_rec_init` (`"zero"` | `"xavier"`) and `w_rec_gain` are read from
  `config.extra` **before** `super().__init__()` because `BioModel.__init__`
  invokes `_build_layers()` → `_init_weights()` which needs them.
- `_init_weights` was refactored into small module-level helpers
  (`_unwrap_weight`, `_init_weight_storage`, `_zero_bias`,
  `_wrec_init_gain`) to keep complexity down.
- **Important discovered defect:** with `use_spectral_norm=True`, `W_rec`
  was previously always spectral-norm-parametrized and then zero-initialized.
  Spectral norm's power iteration divides by the (zero) norm → **all NaN** in
  the forward pass, and later produced an exploding `W_rec` gradient
  (‖grad‖ ≈ 2e5). Fix: spectral norm is applied to `W_rec` only on the
  implicit-equilibrium path (`gradient_method="equilibrium"`); the contrastive
  path keeps plain `nn.Linear` so `w_rec_init="zero"` (the honest default) is
  numerically safe. When spectral norm *is* present and `"zero"` is requested,
  `_wrec_init_gain` falls back to a small xavier (gain 0.1) so the power
  iteration stays finite.
- Tests cover zero mode, xavier mode, gain magnitude, knob visibility, and the
  zero default.

#### B1 — Contrastive profiling script (`scripts/contrastive_profile.py`)

- New script with the required CLI args (`--model --task --num-layers
  --hidden-dim --beta --learning-rate --epochs --batch-size --seed --device
  --output-dir`).
- Runs `gradient_method="contrastive"` with `contrastive_diagnostics=True`,
  profiles the first 10 train steps, and writes
  `runs/contrastive_profile/<ts>/<model>_depth<N>_<task>/{diagnostics.json,
  summary.md}`.
- `summary.md` contains the per-step/per-layer table (Step, Layer, Pre Δ Norm,
  Post Δ Norm, Grad Norm, Update Scale) plus a global diagnostics table.
- Implements **Gate G1** (`_check_gate_g1`): triggers for depth ≥ 3 when, on a
  majority of early steps, `early_post_state_delta/output_state_delta < 1e-3`
  AND `early_grad/output_grad < 1e-3`.

**Initial evidence from tiny runs (CPU, digits, 1 epoch, 10 steps):**

- `eqprop` depth 3, hidden 32: early-layer `post_state_delta_norm` ≈ 1e-4 …
  7e-4 vs output ≈ 0.76–2.7 on steps 1–9 (step 0 retains a transient
  feedforward-init signal). The deep vanilla rule is visibly starved of
  contrastive signal in early layers — G1's *majority-of-steps* criteria is on
  the boundary (both ratios < 1e-3 on some steps, not all), so G1 did not
  formally fire for this tiny run but the trend matches the vanishing-signal
  hypothesis.
- `directed_ep` depth 3, hidden 32: G1 not triggered (feedback keeps
  early-layer signal alive), consistent with feedback being a salvage path.
- These are shallow diagnostic observations, not conclusions; full scans
  (depth 1–4 × the six B2 models × digits/mnist) are still required before
  recording Gate G1 / G2 findings.

#### D3 (partial) — new tests

- `tests/unit/models/test_eqprop_diagnostics.py` — diagnostic keys exist,
  values finite, output delta non-zero with β>0, disabled-by-default, enabled
  via config extra, DirectedEP diagnostics include feedback.
- `tests/unit/models/test_eqprop_update_scale.py` — linear update scaling,
  geometric depth list, β stays global.
- `tests/unit/models/test_eqprop_wrec_init.py` — zero/xavier/gain/visibility/
  default.
- Fixed pre-existing broken tests that were exercising `train_step` with
  single-hidden `gradient_method="equilibrium"` (which intentionally returns
  `None` to delegate to the implicit path):
  - `tests/unit/models/test_eqprop_energy_gradients.py` now uses
    `gradient_method="contrastive"`.
  - `tests/unit/models/test_eqprop_models.py` `_make_config` adds
    `gradient_method="contrastive"`.
- `tests/integration/test_equilibrium_implicit_learns.py`: the learnability
  test regenerated the weight matrix `w` inside the training loop (moving
  target — cannot learn); moved `w` outside the loop.

### Lint / type status

- `ruff format` clean on all touched files.
- `ruff check`: 0 **new** violations introduced. Remaining findings in
  `_energy.py` (12) and `_contrastive.py` (9) are all pre-existing (verified by
  counting against `git stash` baseline). `search_space.py` (2) pre-existing.
  `scripts/contrastive_profile.py` is fully clean.
- `pyright` (strict): 0 errors; warnings reduced 33 → 27 on the touched
  modules vs baseline.

### Test verification

```bash
uv run pytest tests/unit/models/test_eqprop_diagnostics.py \
  tests/unit/models/test_eqprop_update_scale.py \
  tests/unit/models/test_eqprop_wrec_init.py -q --no-cov            # all pass
uv run pytest tests/unit/models/ -q --no-cov                        # 327 pass
uv run pytest tests/unit/experiment/ -q --no-cov                    # 135 pass
uv run pytest tests/unit/validation/test_registry_audit.py tests/unit/experiment/test_config_knobs.py tests/unit/test_rule_space_integrity.py -q --no-cov  # pass
uv run pytest tests/integration/test_equilibrium_implicit_learns.py -q --no-cov  # 4 pass (2 pre-existing parities excluded)
```

Pre-existing failures confirmed unrelated (verified via `git stash`):
`tests/integration/test_equilibrium_parity.py::test_mlp_gradient_parity` and
`tests/unit/validation/test_backprop_parity.py[eqprop_mlp, directed_ep]`.

### Known follow-ups / improvement opportunities

1. **`_contrastive_step` complexity** (pre-existing `PLR0913`/`PLR0917`/local
   count): the function now has 30+ locals. Next session could bundle the
   free/nudged activation lists into a small frozen dataclass (or per-layer
   state object) to shrink the signature and locals.
2. **W_rec spectral-norm asymmetry**: the contrastive path drops spectral norm
   on `W_rec` entirely, so contractivity of the *scalar* recurrence relies on
   small init/β. This is a documented plan deviation attributable to the zero-init
   NaN defect. If evidence ever shows the contrastive deep path needs
   contractivity independent of `w_rec_init`, revisit by initializing non-zero
   before applying spectral norm rather than adding the zero case back.
3. **`test_backprop_parity[eqprop_mlp / directed_ep]`** pre-existing failures
   should be triaged separately (likely parity-harness budget/setting drift,
   not the engine change here).
4. **Gate G1 conservatism**: the `1e-3` ratio threshold with "majority of
   steps" is fairly strict; tiny hidden_dim=32 runs show early-step transient
   signals. For the eventual B2 full scan, consider reporting per-step ratios
   in the summary.md (already in JSON) so the trend is visible even when the
   boolean gate does not fire.
5. **Next session target**: Session 3 (B2 autopsy scan across six models ×
   depth 1–4 × digits/mnist) using `scripts/contrastive_profile.py`, then
   B3 feedback evaluation against Gate G2 once diagnostic reports exist.

---

