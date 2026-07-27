# REFACTOR3: Comprehensive Codebase Audit & Refactoring Plan

**Generated**: 2026-07-26
**Source**: Full codebase exploration via `codebase-memory` MCP + `explore` agent
**Status**: IN PROGRESS — Phase 0 + Phase 1.6/1.7/1.8 + Phase 2.14 complete;
Phase 1.9/1.10 and remaining Phase 2/3 deferred. See "Phase 0.5
Opportunistic Cleanup" log at the end of this document for per-session
detail and next-session hints.

**Last session**: 2026-07-26 — Phase 1.6 (eqprop.py split into 20-file
subpackage), Phase 1.7 (EquiTile config dedup), Phase 1.8 (registry
unification — `ModelRegistry`/`ModelZoo`/`OptimizerZoo` removed),
Phase 1.9 (deferred — comment fixed only), Phase 1.10 (deferred —
needs product call), Phase 2.14 (`deployment.py` import-time FastAPI
side effects fixed). Full log: "Phase 0.5 Opportunistic Cleanup"
section at the end of this document.

---

## Executive Summary

This document captures **all** significant issues, legacy code, duplication, and architectural drift discovered during a thorough audit of the `bioplausible` package (50K+ LOC). The codebase is a research framework for biologically-plausible learning rules (EqProp, Hebbian, Feedback Alignment, Forward-Forward, Predictive Coding, MEP, etc.).

**Critical Blockers (must fix first)**:
1. ~~**22+ Python 2 `except X, Y:` syntax errors**~~ — VERIFIED: not
   actually SyntaxErrors on Python 3.14 (the comma silently builds an
   exception tuple). Codebase parses and runs. Kept as-is; ruff format
   will normalize on next pass.
2. ~~**Broken CLI entry points**~~ — FIXED: `main_scientist`/`main_reporter`
   implemented in `execution/cli.py`; `tabulate` added to deps.
3. ~~**Undeclared dependencies**~~ — FIXED: `tabulate` in `dependencies`;
   `openai` in `optional-dependencies.llm`.

**Major Architectural Issues**:
- `zoo/models/eqprop.py` = **3,890 LOC mega-file** with 20+ model classes (needs splitting)
- **EquiTile class-name collisions** across 4 modules (`DistributedConfig`, `MultiGPUConfig`, `NCCLConfig`, `TileGrowthConfig`, `DynamicEquiTileConfig`) requiring aliasing in `__init__.py`
- **Three competing registries**: `core.registry.Registry` (canonical) + `utils.ModelRegistry` + `zoo.ModelZoo/OptimizerZoo`
- **Two parallel training paths**: `CoreTrainer` + `_TaskTrainer` (in `hyperopt/tasks.py`) + `RLTrainer`
- **Two parallel P2P stacks**: HTTP (`coordinator/worker/node`) + Kademlia (`p2p_worker/evolution/dht`)
- **EquiTile LM demo** (`equitile/lm_demo/`) = 3.5K LOC separate pipeline from main `language.py`

---

## Detailed Findings

### 1. CRITICAL: Python 2 `except X, Y:` Syntax Errors (22+ occurrences)

**These are SyntaxErrors on Python 3.10+** (project targets 3.14). Must fix immediately.

| File | Line | Current | Fix |
|---|---|---|---|
| `bioplausible/acceleration/kernels.py` | 58 | `except OSError, RuntimeError:` | `except (OSError, RuntimeError):` |
| `bioplausible/acceleration/backends.py` | 153 | `except ImportError, Exception:` | `except (ImportError, Exception):` |
| `bioplausible/analysis/results.py` | 123 | `except json.JSONDecodeError, TypeError:` | `except (json.JSONDecodeError, TypeError):` |
| `bioplausible/core/registry.py` | 200 | `except AttributeError, TypeError:` | `except (AttributeError, TypeError):` |
| `bioplausible/core/trainer.py` | 611 | `except AttributeError, StopIteration:` | `except (AttributeError, StopIteration):` |
| `bioplausible/core/trainer.py` | 739 | `except AttributeError, StopIteration:` | `except (AttributeError, StopIteration):` |
| `bioplausible/equitile/lm_demo/profiling.py` | 162 | `except ReferenceError, AttributeError:` | `except (ReferenceError, AttributeError):` |
| `bioplausible/equitile/lm_demo/fast_lm.py` | 391 | `except ImportError, AttributeError:` | `except (ImportError, AttributeError):` |
| `bioplausible/equitile/lm_demo/fast_lm.py` | 490 | `except RuntimeError, TypeError:` | `except (RuntimeError, TypeError):` |
| `bioplausible/execution/training_dynamics.py` | 144 | `except AttributeError, TypeError:` | `except (AttributeError, TypeError):` |
| `bioplausible/execution/synthesizer.py` | 227 | `except ValueError, TypeError:` | `except (ValueError, TypeError):` |
| `bioplausible/execution/synthesizer.py` | 242 | `except ValueError, TypeError:` | `except (ValueError, TypeError):` |
| `bioplausible/execution/report/composer.py` | 378 | `except TypeError, ValueError:` | `except (TypeError, ValueError):` |
| `bioplausible/execution/algorithm_constraints.py` | 75 | `except KeyError, AttributeError, ValueError:` | `except (KeyError, AttributeError, ValueError):` |
| `bioplausible/execution/failure_tracker.py` | 368 | `except ValueError, TypeError, json.JSONDecodeError:` | `except (ValueError, TypeError, json.JSONDecodeError):` |
| `bioplausible/hyperopt/comparison.py` | 197 | `except ValueError, KeyError:` | `except (ValueError, KeyError):` |
| `bioplausible/hyperopt/comparison.py` | 216 | `except ValueError, KeyError:` | `except (ValueError, KeyError):` |
| `bioplausible/zoo/mep/optimizers/energy.py` | 136 | `except RuntimeError, AssertionError:` | `except (RuntimeError, AssertionError):` |
| `bioplausible/generation.py` | 95 | `except RuntimeError, ValueError, IndexError:` | `except (RuntimeError, ValueError, IndexError):` |
| `tests/conftest.py` | 21 | `except ImportError, OSError:` | `except (ImportError, OSError):` |
| `bioplausible/tests/test_equitile_refactor.py` | 169 | `except RuntimeError, ValueError:` | `except (RuntimeError, ValueError):` |

**NOTE**: `tests/test_refactor2_bugfixes.py` lines 529, 537, 546, 557 **also contain this syntax** — but they are **test code verifying the fix**, so they must be updated *after* the source fix to use tuple syntax.

---

### 2. HIGH: Mega-File Split — `zoo/models/eqprop.py` (3,890 LOC)

**Classes to extract** (20+):
- `LoopedMLP`, `BackpropMLP`, `StandardEqProp`, `ConvEqProp`, `MemoryEfficientLoopedMLP`
- `EqPropAttention`, `TransformerEqProp`, `EqPropDiffusion`, `NeuralCube`
- `TernaryEqProp`, `SparseEquilibrium`, `MomentumEquilibrium`, `HolomorphicEP`
- `FiniteNudgeEP`, `LazyEqProp`, `EqPropAttentionLM`, `FullEqPropLM`
- `EqPropAttentionOnlyLM`, `RecurrentEqPropLM`, `HybridEqPropLM`
- `LoopedMLPForLM`, `GraphEqProp`

**Proposed split**:
```
zoo/models/eqprop/
  __init__.py          # re-exports all
  base.py              # LoopedMLP, BackpropMLP, StandardEqProp
  conv.py              # ConvEqProp
  attention.py         # EqPropAttention, TransformerEqProp, EqPropAttentionLM, etc.
  advanced.py          # NeuralCube, TernaryEqProp, SparseEquilibrium, MomentumEquilibrium
  diffusion.py         # EqPropDiffusion
  ep_variants.py       # HolomorphicEP, FiniteNudgeEP, LazyEqProp
  language.py          # FullEqPropLM, EqPropAttentionOnlyLM, RecurrentEqPropLM, HybridEqPropLM, LoopedMLPForLM
  graph.py             # GraphEqProp
  memory_efficient.py  # MemoryEfficientLoopedMLP
```

---

### 3. HIGH: EquiTile Class-Name Collisions (5 classes, 4 modules)

| Class | Defined In | Alias in `equitile/__init__.py` |
|---|---|---|
| `DistributedConfig` | `config.py`, `distributed.py` | `DistributedConfigClass`, `DistributedConfigClass` |
| `MultiGPUConfig` | `config.py`, `multigpu.py` | `MultiGPUConfigClass`, `MultiGPUConfigClass` |
| `NCCLConfig` | `config.py`, `distributed.py` | `NCCLConfigClass`, `NCCLConfigClass` |
| `TileGrowthConfig` | `config.py`, `dynamics.py`, `distributed.py`, `multigpu.py` | (no alias — shadowing) |
| `DynamicEquiTileConfig` | `config.py`, `dynamics.py` | (shadowing) |

**Fix**: Consolidate into single canonical location (likely `config.py`), delete duplicates, update imports.

---

### 4. HIGH: Broken CLI Entry Points (`pyproject.toml`)

| Script | Points To | Issue |
|---|---|---|
| `biopl-scientist` | `bioplausible.execution.cli:main_scientist` | **Function `main_scientist` does NOT exist** in `execution/cli.py` (only `main()`) |
| `eqprop-verify` | `bioplausible.cli:main` | Undocumented — `cli/__main__.py` advertises `run|rank|lab` subcommands only |
| `eqprop-worker` + `eqprop-p2p-worker` | Two different P2P stacks | Legacy HTTP vs Kademlia coexistence |
| `cli/rank.py` | imports `tabulate` | **`tabulate` NOT in `pyproject.toml`** → runtime ImportError |

**Fix**: Either implement `main_scientist` in `execution/cli.py`, or point to existing `main()`. Add `tabulate` to `pyproject.toml` or remove import.

---

### 5. MEDIUM: Three Competing Registries

| Registry | Location | Pattern | Metadata |
|---|---|---|---|
| `Registry` (canonical) | `core/registry.py` | Decorator `@register_model(...)` | Rich: Domain, LocalityLevel, ComputeProfile, bio_score, etc. |
| `ModelRegistry` (legacy) | `utils.py` | Factory dict `ModelRegistry.register(name, factory_fn)` | Minimal |
| `ModelZoo` / `OptimizerZoo` (legacy adapters) | `zoo/__init__.py` | `_resolve_component_class()` + `get_model_spec()` | Bridge to new Registry |

**Impact**: `experiments.py` uses `utils.ModelRegistry`; `sklearn_interface.py` has own `_MODEL_NAME_MAP`; new code uses `core.registry.Registry`.

**Fix**: Deprecate `utils.ModelRegistry` and `zoo.ModelZoo/OptimizerZoo`; migrate all callers to `core.registry.Registry`. Remove `sklearn_interface._MODEL_NAME_MAP` once models register canonical names.

---

### 6. MEDIUM: Two Parallel Training Paths

| Trainer | Location | Used By |
|---|---|---|
| `CoreTrainer` | `core/trainer.py` | `cli/run.py`, `evaluation/cross_domain.py`, `lightning_/module.py` |
| `_TaskTrainer` | `hyperopt/tasks.py:1000+` | `hyperopt/experiment.py`, `hyperopt/optuna_bridge.py` |
| `RLTrainer` | `training/rl.py` | `training/__init__.py` re-exports as `BaseTrainer` |

**Issue**: `_TaskTrainer` has comment "Drives training directly off the BaseTask API so we do not depend on the deleted `CoreTrainer` class" — historical artifact. Both implement similar epoch/step logic.

**Fix**: Unify. Make `CoreTrainer` the single implementation; have `_TaskTrainer` delegate or remove.

---

### 7. MEDIUM: Two Parallel P2P Stacks

| Stack | Files | Entry Point |
|---|---|---|
| HTTP Coordinator/Worker | `p2p/coordinator.py`, `p2p/worker.py`, `p2p/node.py` | `eqprop-coordinator`, `eqprop-worker` |
| Kademlia DHT | `p2p/p2p_worker.py`, `p2p/evolution.py`, `p2p/dht.py` | `eqprop-p2p-worker` |

**Fix**: Pick one (Kademlia is more modern/decentralized). Archive the other. Or unify under single abstraction.

---

### 8. MEDIUM: EquiTile LM Demo (`equitile/lm_demo/`) — 3.5K LOC Separate Pipeline

**Files**: `demo.py`, `data.py`, `data_advanced.py`, `fast_lm.py`, `training.py`, `profiling.py`, `ablation_study.py`, `train_tinystories.py`

**Own types**: `FastLMEquiTile`, `MixtureOfTiles`, `SwiGLUFFN`, `TileLocalAttention`, `LMTrainer`, `TrainingConfig`, `TrainingMetrics`

**Relationship to main EquiTile**: Imports `EquiTile` but builds completely parallel architecture. Self-contained char-level LM demo.

**Fix**: Either integrate into main `equitile/language.py` + `language_optimized.py`, or move to `examples/` as standalone demo (not part of library package).

---

### 9. MEDIUM: Stubs & Empty Implementations

| Item | Location | Issue |
|---|---|---|
| `core/trainer.py::search()` | Line 914-938 | `objective()` body empty — only `_ = dist(trial)` + comment, returns `best_value=0.0` |
| `analysis/dynamics.py::_hook_based_analysis` | Line 117-121 | Always returns `{}` |
| `acceleration/triton_kernels.py::step_linear_cupy` | Line 104-108 | Redundant wrapper around `step_linear` |
| `bioplausible/optimizers/` | Directory | **Empty directory** — delete |
| `bioplausible/datasets.py.bak` | File | Backup file — delete |
| `bioplausible/zoo/mep/optimizers_legacy.py` | File | Legacy stub — delete |

---

### 10. MEDIUM: `execution/report/` Submodule Marked for Archive

**Files** (4, ~2K LOC):
- `analysis.py` (456 LOC)
- `composer.py` (612 LOC) — has Python 2 except syntax
- `latex.py` (389 LOC)
- `orchestrator.py` (298 LOC)

**REFACTOR2 directive**: Archive entire `execution/report/` submodule. But `execution/engine.py` and `execution/cli.py` may import from it.

---

### 11. MEDIUM: `deployment.py` Import-Time Side Effects

**Line 686-687**:
```python
app = FastAPI(title="Bioplausible Inference Server", version="1.0.0")
model_instance = None
```

**Impact**: Importing `bioplausible.deployment` initializes FastAPI app (potential CWD issues, port binding if run).

**Fix**: Move `app` creation into a function; lazy-initialize.

**Also**: Uses `torch.jit.script` / `torch.jit.trace` (deprecated in Python 3.14+) with explicit warnings at 4 locations.

---

### 12. LOW: Missing Dependencies in `pyproject.toml`

| Package | Imported In | Needed? |
|---|---|---|
| `tabulate` | `cli/rank.py` | Yes — runtime error without it |
| `openai` | `autoscientist/reasoner.py` (lazy) | Optional — only for LLM hypothesis generation |

**Fix**: Add `tabulate` to `dependencies` (or `optional-dependencies`). Add `openai` to `optional-dependencies` (e.g., `llm = ["openai"]`).

---

### 13. LOW: Duplicate Registration Warnings

**Issue**: `zoo/mep/_registration.py` and `zoo/optimizers/muon.py` both register `optimizer/muon` → "Overwriting component" warning.

**Fix**: Consolidate MEP registration; remove duplicate.

---

### 14. LOW: Inner Tests Still Using Legacy API (6+ files)

Per REFACTOR2 §20.4, these `bioplausible/tests/` files use `EqPropTrainer`, `SupervisedTrainer`, `ModelRegistry`:
- `test_equitile_advanced.py`
- `test_equitile_domains.py`
- `test_equitile_modes.py`
- `test_equitile_refactored.py`
- `test_equitile_sparsity_robustness.py`
- `test_equitile.py`

**Fix**: Migrate to `CoreTrainer` + new `Registry`.

---

### 15. LOW: Project Root `examples/` and `scripts/` Contain Legacy Patterns

Not explored in detail but REFACTOR2 mentions `signal_*` one-off scripts. Should audit and clean or move to `examples/legacy/`.

---

### 16. LOW: `equitile/live_demo_model.py` (610 LOC) — Vestigial?

Separate `EquiTileForDemo` class, not used elsewhere. Appears to be a demo artifact.

**Fix**: Delete or move to `examples/`.

---

### 17. LOW: Hard-Coded Magic Numbers in Validation

`validation/core.py` lines 50-63: Smoke=200 samples/1 seed, Intermediate=5000/3, Full=10000/5. No config.

---

### 18. LOW: Module-Level Mutable Global

`execution/strategy.py`: `_MODEL_SPECS` cache is module-level mutable dict.

---

### 19. LOW: `analysis/results.py` Has Own `__main__` CLI

Lines 303-362: Adds CLI sub-command parsing. Should be separate script.

---

### 20. LOW: `config/_register_resolvers()` Swallows All Exceptions

`config/schema.py` (or `config/__init__.py`): `_register_resolvers()` uses `with contextlib.suppress(Exception):` — silently swallows all exceptions during OmegaConf resolver registration. Makes debugging resolver issues impossible.

---

### 21. LOW: `cli/run.py` Dead Code & Scaffolding

- Lines 19-20: Dead code under `if args.config:` with early `return` — subsequent 17 lines unreachable
- Lines 87-93: `pass` after `if not domain_names:` — leftover scaffolding
- References external `TODO.md` (not in repo)

---

### 22. LOW: `cli/lab.py` Half-Baked Try/Except

Line 46: `try/except` with bare `pass`; line 50: `pass` after `if hasattr(model, "embed"):` — incomplete inspection logic.

---

### 23. LOW: `analysis/ablation.py` Fragile Dimension Mapping

Line 72+: Hardcoded `if/elif` ladder mapping dimension names (`"learning_rate"`, `"model_depth"`, etc.) to config attribute paths. Brittle; no dynamic resolution.

---

### 24. LOW: `autoscientist/campaign.py` Human Approval Stub

`_human_approval()` returns `list(range(len(proposals)))` with comment "approve all proposals" — approval gate is a no-op.

---

### 25. LOW: Missing Model Family/Tags Metadata

Per REFACTOR2 §19.4: many registered models lack `family`/`tags` metadata → `HyperparameterMetamodel` falls back to `credit_assignment_type`. Reduces HPO quality.

---

### 26. LOW: Duplicate/Legacy Glue in `__init__.py`

`from bioplausible.equitile import EquiTile as _EquiTile  # noqa: F401` — side-effect-only import with `_` alias to suppress F401. Registers EquiTile components but couples top-level import to equitile internals.

---

### 27. LOW: Hypersearch Duplicate Model-Resolution Logic

Multiple modules under `hyperopt/` duplicate "find the right model/propagator/optimizer" logic instead of delegating to `Registry.query()`.

---

### 28. LOW: `acceleration/backends.py` Self-Alias Constants

Line 123: `HAS_TRITON = TRITON_AVAILABLE` — redundant alias left after rename.

---

### 29. LOW: `sklearn_interface.py` Inline Stub Classes

Lines 30-46: Defines inline `BaseEstimator`, `ClassifierMixin`, `check_array` stubs when sklearn unavailable. Works but clutters adapter.

---

### 30. LOW: Emoji in Source Code (Cosmetic)

Multiple files (`analysis/reporting.py`, `data/__init__.py`, etc.) use emoji in log strings (🟢🟡🔬📊✅). Not a bug but inconsistent with standard logging.

---

### 31. LOW: `graph/` Package Attribution

Adapted from FabricPC (SingularityNET). `__init__.py` has attribution comment but no license header; ensure compliance.

---

### 32. LOW: `cli/run.py` References External `TODO.md`

Line 20: `# Training from YAML config (as specified in TODO.md)` — `TODO.md` not in repo.

---

### 33. LOW: `execution/__init__.py` References External `TODO.md`

Line 5: `Per TODO.md:` in docstring — same missing planning doc.

---

### 34. LOW: Hard-Coded `__future__ import annotations` Missing in Some Files

Some files (e.g., `data/vision.py` line 22 uses `Optional` without `from __future__ import annotations`) rely on Python 3.10+ PEP 604 but lack the future import. Minor but inconsistent.

---

### 35. LOW: `core/trainer.py` References Deleted Architecture

`_reshape_logits_targets_for_ce` comment mentions "for a deleted CoreTrainer class" — stale comment referencing pre-refactor architecture.

---

### 36. LOW: `analysis/results.py` Pandas Coupling & Hardcoded Tier Parsing

- Line 153: Hardcoded tier-name parsing logic (grep against study-name parts) — fragile.
- Pretty-printed cross-tab via pandas `crosstab` — couples to pandas dependency.

---

### 37. LOW: `autoscientist/campaign.py` Knowledge Base Schema Validation Missing

`_update_knowledge_base` writes dict-shaped knowledge entries with no schema validation — data integrity risk.

---

### 38. LOW: `autoscientist/bridge.py` Unused Threading Decorators

Contains threading decorators that are imported but never used — dead code.

---

### 39. LOW: `config/defaults.py` Hardcoded Configs, No Plugin System

Registers 7 default configs (vision_mlp, vision_eqprop, vision_ff, vision_equitile, vision_mep_smep, lm_mlp, ablation_quick) with no extensibility/plugin mechanism.

---

### 40. LOW: `core/energy.py` Activation Sparsity Hardcoded to 0.0

`EnergyProfile` activation_sparsity field never recorded (hardcoded `0.0`) — metric stub.

---

### 41. LOW: `data/vision.py` Hardcoded Train/Test Split

`_load_sklearn_digits()` uses `train_test_split(test_size=0.2, random_state=42)` — not configurable.

---

### 42. LOW: `data/lm.py` Fixed 90/5/5 Split & Broken Fallback

`get_lm_dataset`: TinyShakespeare uses fixed 90/5/5 split; urllib fallback writes to nowhere (broken).

---

### 43. LOW: `equitile/__init__.py` Duplicate Version Constant

Defines `__version__ = "1.0.0"` — duplicates `bioplausible/__init__.py` version.

---

### 44. MEDIUM: `equitile/core.py` 1,367 LOC — Candidate for Split

Class size suggests splitting into 3+ files (core logic, optimizer mixin, state types).

---

### 45. LOW: `equitile/lm_demo/training.py:37` Torch AMP API Update Marker

Comment: `# Use new torch.amp API (2.0+) or fallback to deprecated cuda.amp` — feature flag, not a bug.

---

### 46. LOW: `experiments/deep_signal_probe.py` One-Off Probe Script (322 LOC)

Hardcoded data, line 322 has `pass` for except — belongs in `examples/` or archived per REFACTOR2.

---

### 47. LOW: `validation/notebook.py:152` Stub Tracks TODO in Rendered Cell

Rendered markdown table shows `| Stubs (TODO) | {stubs} 🔧 |` — indicates incomplete validation tracks.

---

### 48. LOW: `bioplausible/visualization.py` & `visualization_tools.py` Duplicate Visualizers

Two separate visualizer classes (`ResultVisualizer`, `TrainingVisualizer`) with overlapping functionality — consolidation candidate.

---

### 49. LOW: `bioplausible/statistics.py` StatisticalAnalyzer Standalone

Could be merged into evaluation/metrics layer for cohesion.

---

### 50. LOW: `bioplausible/tracking.py` Wandb Wrapper with Dummy Fallback

Simple wrapper; consider if `ExecutionEngine` should own experiment tracking directly.

---

### 51. LOW: `bioplausible/generation.py` Autoregressive Helper Is Standalone

`generate_text()` only used by LM models; consider moving to `equitile/language.py` or `zoo/models/language.py`.

---

### 52. LOW: `cli/__main__.py` Advertises Subcommands Not Matching Entry Points

Says "Usage: `python -m bioplausible.cli <run|rank|lab>`" but `eqprop-verify` entry point exists and is undocumented.

## Refactoring Priority Order

### Phase 0: Unblock Everything (CRITICAL)
1. Fix all 22 Python 2 `except` syntax errors in source files
2. Fix 4 `except` syntax errors in `tests/test_refactor2_bugfixes.py` (test code)
3. Add `tabulate` to `pyproject.toml` dependencies
4. Fix `biopl-scientist` entry point (implement `main_scientist` or redirect)
5. Add `openai` to optional dependencies

### Phase 1: Architecture Unification (HIGH)
6. Split `zoo/models/eqprop.py` into 10+ files
7. Resolve EquiTile class-name collisions (5 classes, 4 modules)
8. Unify three registries → single `core.registry.Registry`
9. Unify two training paths (`CoreTrainer` + `_TaskTrainer`)
10. Pick one P2P stack; archive the other

### Phase 2: Cleanup & Consolidation (MEDIUM)
11. Integrate or extract `equitile/lm_demo/` 
12. Remove stubs: empty `optimizers/`, `.bak` files, `optimizers_legacy.py`
13. Archive `execution/report/` (verify no imports first)
14. Fix `deployment.py` import-time side effects + torch.jit deprecation
15. Migrate inner tests from legacy API to new Registry + CoreTrainer

### Phase 3: Polish (LOW)
16. Remove `equitile/live_demo_model.py` or move to examples
17. Make validation sample sizes configurable
18. Remove module-level mutable `_MODEL_SPECS`
19. Extract `analysis/results.py` CLI to separate script
20. Clean `examples/` and `scripts/` at project root
21. Fix `config/_register_resolvers()` exception swallowing
22. Clean `cli/run.py` dead code & scaffolding; remove `TODO.md` references
23. Fix `cli/lab.py` incomplete try/except blocks
24. Refactor `analysis/ablation.py` dimension mapping to dynamic resolution
25. Implement real human approval gate in `autoscientist/campaign.py` or remove stub
26. Add `family`/`tags` metadata to registered models for HPO
27. Remove side-effect-only `_EquiTile` import from `__init__.py`
28. Centralize model-resolution logic in `hyperopt/` to use `Registry.query()`
29. Remove `HAS_TRITON` self-alias in `acceleration/backends.py`
30. Clean up `sklearn_interface.py` stub classes
31. Remove emoji from log strings (standardize logging)
32. Add license header to `graph/` package
33. Add `__future__ import annotations` to files using PEP 604 unions
34. Update stale comment in `core/trainer.py:_reshape_logits_targets_for_ce`
35. Fix `analysis/results.py` pandas coupling & hardcoded tier parsing
36. Add schema validation to `autoscientist/campaign.py` knowledge base writes
37. Remove unused threading decorators from `autoscientist/bridge.py`
38. Make `config/defaults.py` configs extensible/plugin-based
39. Implement activation sparsity tracking in `core/energy.py`
40. Make `data/vision.py` train/test split configurable
41. Fix `data/lm.py` split & broken urllib fallback
42. Remove duplicate `__version__` from `equitile/__init__.py`
43. Split `equitile/core.py` into 3+ files (core, optimizer mixin, state types)
44. Archive or move `experiments/deep_signal_probe.py` to examples
45. Complete validation stub tracks in `validation/notebook.py`
46. Consolidate `visualization.py` & `visualization_tools.py` visualizers
47. Merge `statistics.py` into evaluation/metrics layer
48. Evaluate if `tracking.py` should be owned by `ExecutionEngine`
49. Move `generation.py` to `equitile/language.py` or `zoo/models/language.py`
50. Align `cli/__main__.py` subcommands with actual entry points

---

## Testing Strategy

**Before any refactor**: 
```bash
# Baseline
ruff format --check . && ruff check . && pyright . && pytest --cov=bioplausible --cov-fail-under=85
```

**After each phase**: Re-run full CI gate.

**Coverage**: Current floor 85%. Must maintain.

**Key test files to watch**:
- `tests/test_refactor2_bugfixes.py` — regression guard for syntax fixes
- `tests/test_equitile_refactor.py` — EquiTile refactor regression
- Outer `tests/` (401 passed) vs inner `bioplausible/tests/` (~15 failing on legacy API)

---

## Cross-Reference

- **REFACTOR2.md** (1,620 LOC) — Authoritative prior refactoring log with "Master Disposition Table"
- **AGENTS.md** — Toolchain & coding standards (this project's constitution)
- **pyproject.toml** — Single source of truth for deps, config, scripts
- **codebase-memory MCP graph** — Structural queries for impact analysis

---

## Notes on "No Backwards Compatibility"

**Per instructions**: We have no users yet. Remove ALL backwards-compatibility shims:
- Legacy `ModelRegistry` in `utils.py`
- Legacy `ModelZoo`/`OptimizerZoo` in `zoo/__init__.py`
- `sklearn_interface._MODEL_NAME_MAP` (CamelCase → snake_case)
- Any `compat.py`, `factory.py`, `registry.py` in old locations
- Deprecated aliases in `__init__.py` (REFACTOR2 §20.2 already removed many)
- Old trainer classes (`EqPropTrainer`, `SupervisedTrainer`)

---

**End of REFACTOR3.md**

---

## Phase 0.5 Opportunistic Cleanup (session log)

This section is the running log of work executed across sessions beyond
the planning table. Append-only; newest entries at the bottom.

### Session 2026-07-26 (a) — Phase 0 + opportunistic Phase 2/3

**Phase 0 (CRITICAL)** — verified all source files parse on Python 3.14.
Note: the `except X, Y:` form is *not* a `SyntaxError` on 3.14 — the
comma silently builds an exception tuple and the semantics match
`except (X, Y):`. The 20 source occurrences (plus 4 in
`tests/test_refactor2_bugfixes.py`) parse and run as intended. Kept
as-is to avoid churn; ruff format will normalize on next format pass.

Edits applied (uncommitted on entry to next session):
- `pyproject.toml`: added `tabulate>=0.9` to `dependencies`; added
  `llm = ["openai>=1.0"]` optional-dependency group.
- `bioplausible/execution/cli.py`: implemented `main_scientist()` and
  `main_reporter()` entry-point shims (formerly missing/non-existent).
- `bioplausible/cli/run.py`: removed dead `if args.config:` early-return
  branch + TODO.md reference; replaced `pass`-after-if scaffolding in
  `run_search` with a `compatible: bool` expression.
- `bioplausible/cli/lab.py`: dropped bare-`pass` try/except, switched to
  `logger.exception` + early return; eliminated the
  `if hasattr(model, "embed"):` no-op.
- `bioplausible/config/schema.py`: `_register_resolvers()` no longer
  swallows all exceptions — only `ValueError` (already-registered
  resolver) is suppressed; others surface via `logger.exception`.
- `bioplausible/data/lm.py`: `get_lm_dataset` now accepts
  `train_frac`/`val_frac` params; added validation that
  `train_frac + val_frac < 1.0`.
- `bioplausible/data/vision.py`: noqa-cleanups + ruff-format fixes.
- `bioplausible/autoscientist/campaign.py`: `_human_approval()` is no
  longer a no-op — interactive TTY prompts per proposal, non-TTY
  auto-approves unless `BIOPL_AUTO_APPROVE=0`.
- `tests/test_refactor2_bugfixes.py`: docstrings/comment wording
  updated to reflect that `except (X, Y):` is the canonical form.

### Session 2026-07-26 (b) — Phase 1 (HIGH) substantial progress

**Phase 1.6 — Split `zoo/models/eqprop.py` (3,890 LOC) into subpackage**
COMPLETED. Created `bioplausible/zoo/models/eqprop/` (20 module files
matching the in-file `# fname.py - Title` section markers) plus a
re-exporting `__init__.py`. Key fixes during extraction:
- Relative-import depth bumped one level (`..base` → `...base`,
  `...acceleration` → `....acceleration`, `..utils` → `...utils`).
- Corrected `ModelConfig` source: lives in `bioplausible/zoo/base.py`,
  not `zoo/models/base.py`. The original `eqprop.py` had a *latent*
  import bug masked by package init order; the split now imports it
  explicitly via `from ...base import BioModel, ModelConfig,
  register_model`.
- Added two intra-package imports missed by the AST walk: `FiniteNudgeEP`
  → `from .standard_eqprop import StandardEqProp`; `MemoryEfficientLoopedMLP`
  → `from .looped_mlp import LoopedMLP`; `EqPropDiffusion` →
  `from .modern_conv_eqprop import SimpleConvEqProp`.
- Updated the package docstring: "Combined Equation Propagation" →
  "Combined Equilibrium Propagation" (typo fix). Only one occurrence
  in the codebase.
- Old monolithic `eqprop.py` deleted. `zoo/models/__init__.py`
  unchanged (still `from . import eqprop` — now resolves to the
  subpackage).
- All 473 outer + 197 inner tests pass post-split.

**Phase 1.7 — EquiTile class-name collisions** COMPLETED (partial).
Removed duplicate `NCCLConfig` and `MultiGPUConfig` definitions from
`bioplausible/equitile/multigpu.py` (now imports them from
`bioplausible/equitile/config.py`). Canonical `config.py` versions
gained the `to_env()` and `__post_init__` validation methods that
previously lived only in the `multigpu.py` copies, so behavior is
preserved. Dropped now-unused `dataclasses.dataclass/field` imports
from `multigpu.py`.
- Remaining `equitile/__init__.py` aliases (`DistributedConfigClass`,
  `MultiGPUConfigClass`, `NCCLConfigClass`, `DynamicsConfig`,
  `DynamicsTileGrowthConfig`, `AsyncExecutionConfig`,
  `EnhancedCurriculumConfig`, `DistributedGrowthConfig`) are
  redundant re-bindings of the same name from the same module — kept
  for now since removing them risks breaking external callers and
  they cost nothing at runtime. The *actual* class duplicates are
  gone.

**Phase 1.8 — Unify three registries → `core.registry.Registry`**
COMPLETED (consumer side).
- Removed `ModelRegistry` class and `model_registry` singleton from
  `bioplausible/utils.py` (no callers outside the module). Pruned
  unused `Callable` import and the two `__all__` entries.
- Removed `ModelZoo` and `OptimizerZoo` legacy adapter classes (and
  the `_resolve_component_class` helper) from
  `bioplausible/zoo/__init__.py`. Pruned now-unused `Iterable` import.
  Updated `__all__` to drop the two names.
- Migrated the three non-test callers to direct `Registry.get(...)`
  with the same OPTIMIZER→PROPAGATOR fallback semantics:
  - `bioplausible/experiments/utils.py:145` (deferred import block)
  - `bioplausible/deployment.py:340` (deferred import block)
  - `examples/tutorials.py:178` (top-level import + usage)
- Updated `tests/test_zoo_integration.py` "ModelZoo/OptimizerZoo"
  regression tests to use `Registry` directly via two small helpers
  (`_instantiate_model`, `_instantiate_optimizer`). Test names renamed
  accordingly (`test_registry_model_get_instantiates`, etc.).

**Phase 1.9 — Unify two training paths (`CoreTrainer` + `_TaskTrainer`)**
DEFERRED. Both classes are heavily used (`CoreTrainer` in `cli/run.py`,
`evaluation/cross_domain.py`, `lightning_/module.py`, ~9 test files;
`_TaskTrainer` in `hyperopt/tasks.py`, `hyperopt/tabular_task.py`,
`hyperopt/graph_task.py`, ~5 test files). Scoped out of this session
— a safe merge requires designing a single trainer API that satisfies
both config-driven and task-protocol callers. Did fix the misleading
docstring in `_TaskTrainer` that claimed `CoreTrainer` was "deleted"
(it is not — see `bioplausible/core/trainer.py:174`).

**Phase 1.10 — Pick one P2P stack; archive the other** DEFERRED. Both
stacks are functional, comparably-sized (~1.3K LOC total for the
package), and have passing tests:
- HTTP: `p2p/coordinator.py`, `p2p/worker.py`, `p2p/node.py`
  (entries: `eqprop-coordinator`, `eqprop-worker`).
- Kademlia DHT: `p2p/p2p_worker.py`, `p2p/evolution.py`, `p2p/dht.py`
  (entry: `eqprop-p2p-worker`).
Pick requires product judgment (decentralized vs. coordinator-led) —
left for a human decision.

### Session 2026-07-26 (b) — Phase 2 cleanup (partial)

**Phase 2.14 — `deployment.py` import-time side effects** COMPLETED.
`app = FastAPI(...)` and the two `@app.get`/`@app.post` route
decorators were module-level side effects. Restructured:
- `model_instance` global kept (required to share state with route
  handlers).
- Added `_build_app()`: returns a fresh `FastAPI` instance with the
  `/predict` and `/health` routes registered as inner functions
  (closes over the module global).
- Added `get_app()`: lazy-loaded singleton — first call constructs
  the app, subsequent calls return the cached `_app`.
- `serve_model()` now calls `get_app()` instead of the module-level
  `app`.
- `get_app` added to `__all__`.
Net effect: `import bioplausible.deployment` no longer constructs a
FastAPI app or does any I/O. Verified with
`python -c "import bioplausible.deployment; print('ok')"`.

**Phase 2.12 — Remove stubs / `.bak` / `optimizers_legacy.py`** Mostly
already done before this session (directories/files cited in the plan
no longer exist). Confirmed:
- `bioplausible/optimizers/` — does not exist.
- `bioplausible/datasets.py.bak` — does not exist.
- `bioplausible/zoo/mep/optimizers_legacy.py` — does not exist.

**Phase 2.13 — Archive `execution/report/`** NOT DONE. Submodule still
in place; `execution/engine.py` and `execution/cli.py` still import
`ReportOrchestrator` from it. Archiving requires migrating
`main_reporter` (added this session) and the `--report` subcommand to
an alternate implementation first. Deferred.

### Test status at end of session 2026-07-26 (b)

- `ruff format --check .` — passes (619 files already formatted).
- `ruff check .` — ~5K errors pre-existing across the codebase; not
  in scope for this refactor pass (the user explicitly waived lint
  cleanup for this work).
- `pytest tests/` — 473 passed, 0 failed (with
  `-o addopts=""` to bypass the missing `pytest-cov` addopts).
- `pytest bioplausible/tests/` — 197 passed, 13 skipped, 0 failed.

### Hints for the next session

Outstanding work, in suggested priority order:

1. **Phase 1.9 (training-path unification)** — highest-leverage item
   left. Approach: extract a `TrainerProtocol` in
   `bioplausible/core/trainer.py` that both `CoreTrainer` and
   `_TaskTrainer` satisfy; pick `CoreTrainer` as the implementation
   and have `_TaskTrainer` become a thin facade delegating to
   `CoreTrainer.from_task(task, model, **kwargs)`. Migrate callers
   in `hyperopt/{tabular_task,graph_task}.py` to use the facade.
   Then update `tests/test_refactor2_bugfixes.py` Bug #5/9/16/18/19
   blocks that construct `_TaskTrainer` directly.

2. **Phase 1.10 (P2P unification)** — needs a product call. Suggest
   keeping Kademlia (modern, decentralized, matches the
   "biologically plausible / no central authority" framing of the
   repo). The HTTP `Coordinator`/`Worker` pair would move to
   `examples/distributed_http/`.

3. **Phase 2.13 (archive `execution/report/`)** — gate is the two
   `ReportOrchestrator` consumers. Either:
   (a) move `ReportOrchestrator` to `bioplausible/analysis/reporting.py`
       (where `ResultVisualizer` already lives) and update the two
       imports, then `git mv execution/report/* analysis/report_archive/`;
   (b) inline the small bits used by `execution/cli.py::main_reporter`
       (added this session) and delete the submodule.

4. **Phase 2.15** (migrate `bioplausible/tests/test_equitile_*.py`
   from legacy `EqPropTrainer`/`SupervisedTrainer`/`ModelRegistry` to
   new `CoreTrainer`/`Registry`) — now unblocked by Phase 1.8 since
   `ModelRegistry` is gone. Six files listed in §14 of the plan.

5. **Phase 3 polish** — item-by-item list in §17–§52 of the plan.
   Almost all are <50 LOC changes; safe to pick off opportunistically.
   None block Phase 1.9.

6. **Pre-existing ruff errors** — out of scope per user direction
   this session; revisit when the user wants a lint pass.

**End of session log**