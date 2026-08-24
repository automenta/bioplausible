# Phantom-Knob Audit

**Plan 8 §D2 artifact.** Status: maintained as model registry evolves.

A "phantom knob" is a search-space key that a sweep samples but the model
silently drops at construction — the probe then trains with a default instead
of the value the sweep *thinks* it sampled. Phantom knobs poison parity,
liveness, and hyperparameter searches in identical ways and are detected by
:func:`computronium.core.construction.phantom_knobs` (and enforced by the P0a
surface gate in :mod:`computronium.hyperopt.search_space`).

Every model in the default (non-``--include-broken``) sweep is verified
phantom-free via ``validate_all_rule_spaces()`` and the config-knob regression
tests. This table records the historical audit.

## Audit Table

| Model | Phantom Knob | Action | Evidence | Status |
|---|---|---|---|---|
| `graph_eqprop` | `num_layers` | Quarantine | `phantom_knobs()` flags `num_layers`; constructor builds a fixed single-GCN-loop graph regardless of requested depth | `status:broken` |
| `conv_eqprop` | `num_layers` | Quarantine | `phantom_knobs()` flags `num_layers`; single conv loop, depth not threaded | `status:broken` |
| `modern_conv_eqprop` | `num_layers` | Quarantine | `phantom_knobs()` flags `num_layers`; depth fixed by architecture | `status:broken` |
| `equilibrium_alignment` | `num_layers` | Quarantine | `phantom_knobs()` flags `num_layers` | `status:broken` |
| `hebbian_chain` | `num_layers` | Quarantine | `phantom_knobs()` flags `num_layers` | `status:broken` |
| `deep_hebbian` | `num_layers` | Quarantine | `phantom_knobs()` flags `num_layers` (alias of `hebbian_chain`) | `status:broken` |
| `hebbian_3d` | `num_layers` | Quarantine | `phantom_knobs()` flags `num_layers`; `hebbian_3d` CHL propagator incompatible with probe driver | `status:broken` |
| `eqprop_diffusion` | n/a (diffusion interface) | Quarantine | `forward(x, t)` requires a timestep that the probe driver cannot supply; not a phantom-`num_layers` case but a task-interface incompatibility (see `_forward_probe_ok`) | `status:broken` |
| `neural_cube` | `num_layers` (structural axis is `cube_size`) | Quarantine | `cube_size` defines the 3D lattice; sampled `num_layers` is absorbed via `**kwargs` and silently dropped (no constructor-surface phantom is raised because the knob is "absorbed") — caught by the registry-wide depth guard | `status:broken` |
| `equilibrium_alignment` | `num_layers` (absorbed via `**kwargs`, unused) | Quarantine | Same `**kwargs` absorption: `num_layers` lands in kwargs and never grows the architecture — caught by the registry-wide depth guard | `status:broken` |

## Resolved (depth-cap fix)

| Model | Phantom Knob | Resolution | Evidence | Status |
|---|---|---|---|---|
| `direct_feedback_alignment_eqprop` | `num_layers` | Fixed | Structural fallback depth cap (`min(..., 2)`) removed; `phantom_knobs()` now returns `frozenset()`; `num_layers` honored in `transition_modules()` | `status:experimental` |
| `dfa_deep` | `num_layers` | Fixed | Same fix as parent; depth honored | `status:experimental` |

## Registry-Wide Depth Guard

Since these last two cases are invisible to the constructor-surface gate
(`**kwargs` absorption is reported as "absorbed", not "phantom"), the durable
defense is `test_all_models_honor_depth_or_are_knowingly_phantom`: for every
registered model, either the parameter count grows with sampled `num_layers`,
or the model is tagged `status:broken` (quarantined from default sweeps).
A model doing neither fails the test — no per-model regression needed. This is
the invariant that stops the phantom-depth defect recurring across the zoo.

## Verified Clean (no phantom knobs)

The following models honor sampled `num_layers` (threaded into real hidden
layers) and have no phantom knobs. They are the default-sweep population:

- `eqprop`, `directed_ep`, `lazy_eqprop`, `finite_nudge_ep`,
  `momentum_equilibrium`, `sparse_equilibrium`, `eqprop_mlp`,
  `holomorphic_ep` — the consolidated deep-eqprop engine. Verified by
  `test_num_layers_consumed_is_not_phantom` and
  `test_param_count_varies_with_num_layers`.
- FA-family MLPs (`feedback_alignment`, `adaptive_feedback_alignment`,
  `standard_fa`, `contrastive_feedback_alignment`, `stochastic_fa`,
  `energy_guided_fa`, `energy_minimizing_fa`, `layerwise_equilibrium_fa`) and
  `diff_target_prop`, `fabricpc_graph_pcn` — the default-sweep population,
  covered by `test_all_models_honor_depth_or_are_knowingly_phantom`.

### Structural-fallback depth fix

`num_layers` was once capped at 2 by the structural fallback in
`core/construction.py:construct_model` and `core/model.py:BioModel.build`
(`min(max(len(hidden_dims), 1), 2)`). Any config-accepting model with required
positional args (e.g. `feedback_alignment`) silently built a 2-hidden-layer
architecture regardless of the sampled depth. Both sites now propagate the
full `len(hidden_dims)`; the registry-wide depth guard prevents regression.

## Default Sweep Exclusions

`scripts/broad_sweep.py` filters `status:broken` models by default
(Plan 8 §D1). Re-run them deliberately with:

```bash
uv run python scripts/broad_sweep.py --include-broken --families eqprop
```

The registry status tags are defined in `computronium/core/model_status.py`
and rendered on registrations via the `status_tag()` helper.