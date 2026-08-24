# HPO Protocol: Compute-Matched Search Across Propagator Families

This document specifies the protocol for running **hardware-matched, compute-matched**
hyperparameter optimization across propagator families, via the `biopl-hpo` CLI.

## 1. Entry point

```bash
uv run biopl-hpo search \
  --family eqprop \
  --task digits \
  --budget 8 \
  --budget-tier standard \
  --seed 42 \
  --output ./hpo_results
```

Console script `biopl-hpo` resolves to `computronium/cli/hpo.py:main`, which dispatches to
`computronium.cli.run`.

## 2. Family mapping

The `--family` flag uses CLI aliases that map to internal registry family names:

| CLI flag (`--family`)         | Registry family            |
|-------------------------------|----------------------------|
| `eqprop`                      | `eqprop`                   |
| `fa`                          | `feedback_alignment`       |
| `backprop`                    | `backprop`                 |
| `forward_only`                | `forward_only`             |
| `hebbian`                     | `hebbian`                  |
| `predictive_coding`           | `predictive_coding`        |
| `spiking`                     | `spiking`                  |
| `target_prop`                 | `target_prop`              |
| `equitile`                    | `equitile`                 |
| `all`                         | every family above         |

`mep` is registered as a propagator, not a model, and therefore has 0 models — it is
skipped automatically with a warning.

## 3. Study layout

For each compatible model in a family, **one Optuna study is created per model**
named `{reg_family}_{model}_{task}` (stored in `sqlite:///computronium.db` by
default; pass `--db <file>` to any HPO subcommand to isolate a run in a dedicated
SQLite file).

A single study spans one model only, because `create_optuna_space` derives categorical
choices from model metadata (`max_hidden`, `max_layers`); mixing models in one study
produces incompatible `CategoricalDistribution`s and Optuna aborts with
`does not support dynamic value space`.

Each study is multi-objective: **maximize accuracy, minimize loss** (TPE or NSGA-II
sampler, configurable via `--method`).

## 4. Compute matching

All models within a family run at the **same patience tier** (`--budget-tier`), so epoch
counts, batch size, and sampler warmup are identical — the only varying factor is the
propagation algorithm + its hyperparameters. Default tier mapping:

| Tier      | `n_trials` | epochs | batch size |
|-----------|-----------|--------|------------|
| `smoke`   | 1–2       | 1      | 1          |
| `standard`| 8         | 8      | 16         |
| `extended`| 32        | 16     | 16         |
| `full`    | 64        | 32     | 32         |

`--budget` overrides the per-model trial count.

## 5. Trial metadata schema

Every completed Optuna trial carries `user_attrs` for downstream analysis:

```
model_name, family, task, tier,           # provenance
param_count, iteration_time, loss,         # scalar metrics
epochs, batch_size, seed                    # config echo
```

These feed the comparison and Pareto ranking tools.

## 6. Subcommands

| Command      | Purpose                                                      |
|--------------|--------------------------------------------------------------|
| `search`     | Run compute-matched HPO across a family (one study per model)|
| `compare`    | Aggregate completed studies into a ranked CSV (`--metric`)   |
| `verify`     | Re-run a study's top-k configs with `n` seeds               |
| `pareto`     | Emit Pareto frontier plot + JSON data for a study            |
| `list`       | List registered models and families                          |

## 7. Reproducibility

- All samplers are seeded (`--seed`). `optuna_bridge.create_study` threads the seed
  into `TPESampler`, `RandomSampler`, and `NSGAIISampler`.
- Storage is SQLite (`computronium.db`), so studies resume idempotent via
  `load_if_exists=True`.
- Each trial writes `model_name` + `family` into `user_attrs`; re-running the same
  study name appends to the existing trial set.

## 8. Artifacts

With `--output <dir>`, each model study is exported as a JSONL file named
`{output}/{family}_{model}_{task}_{model}.jsonl`. Pareto front data is written as
JSON via the `pareto` subcommand to `--output-dir`.

## 9. Known limitations

- `torch.compile` / XLA backends are not yet wired into the trial config; only the
  `default`/`triton` acceleration path is exercised.
- A model's `train_step` must be implemented for the chosen credit-assignment family;
  models lacking a custom `train_step` (e.g. some EqProp variants) prune the trial
  via the bridge's exception handler rather than crashing the study.
- **Optuna 4.9 MOTPE crash guard.** When a study accumulates >= `n_startup_trials`
  trials but *every* one is PRUNED (`values=None`), the multi-objective TPE sampler
  crashes with `TypeError`. `run.py:_safe_sampler_name` detects this and falls back to
  a seeded `RandomSampler` for that model so the family run completes. The failure is
  logged (`[SAMPLER] ... falling back to random`) and is reproducible.
- **Stale RUNNING trials.** A killed process can leave a `RUNNING` trial with
  `values=None`. `run.py:_fail_stale_running` marks these as `FAILED` before each
  model's search (logged `[CLEAN]`), so they never contaminate the study or the
  comparison/portfolio reads.
- **`training_checkpoints` schema unification.** The table is created by both
  `hyperopt/storage.py` and `execution/_lifecycle.py`. Both now define the *same*
  union schema (trial_id + trajectory_id + all metric columns); previously whichever
  SQL ran first won, breaking the other with `no column named <x>`. New DBs are fine;
  pre-existing DBs with a stale table can be fixed via
  `DROP TABLE training_checkpoints;`.

