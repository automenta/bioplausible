# TODO7.md — Post-Cleanup Roadmap

> **Scope:** Remaining modularization and infrastructure work after Phase 0–3 completion. All critical path items from TODO6.md are ✅ COMPLETE.

---

## Completed (Session Summary)

### ✅ Ontology Decomposition (Phase 2.2)
- Removed `computronium/core/ontology.py` (5,680 lines)
- Created `computronium/ontology/` with per-axis modules:
  - `substrate.py` — SubstrateConfig, Digital/Analog/Memristive/Neuromorphic/Optical/Quantum/Sparse/Ternary/Complex/Noisy/QuantizedSubstrate
  - `geometry.py` — GeometryConfig, Feedforward/Recurrent/TileGeometry
  - `dynamics.py` — StateDynamicsConfig, EnergyMinimization/PredictiveSettling/SpikeIntegration/Instantaneous/Diffusion/LazyStateDynamics
  - `credit.py` — CreditAssignmentConfig, ThermodynamicContrast/RandomProjections/LocalGoodness/TemporalTrace/TargetInversion/Homeostatic/BackpropCredit
  - `update.py` — ParameterUpdateConfig, Euclidean/RiemannianOrthogonal/SpectralConstrained/NaturalGradient/ElasticConsolidationUpdate
  - `system.py` — SystemConfig, System, SystemState, ModelAdapter, FAMILY_TOLERANCES
  - `plasticity.py` — NEW: Re-exports M-axis plasticity primitives (FastWeight/Routing/RuleState/SubstrateCoupled/NullPlasticity)
  - `__init__.py` — Single import surface: `from computronium.ontology import *`

### ✅ Lazy Import Updates
- `computronium/__init__.py` — All ontology symbols now point to `computronium.ontology.*`
- `computronium/core/__init__.py` — All ontology/state symbols updated to new locations
- Verified: `from computronium import System, DigitalSubstrate, ...` works

### ✅ Build Artifacts Removed
- Deleted `build/`, `computronium.egg-info/`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.coverage`

### ✅ CLI Decomposition (2,011 lines → submodules)
- `computronium/cli/shared.py` — Shared constants, FAMILY_MAP, target resolution, trial context
- `computronium/cli/commands/train.py` — train, core-train, from-config
- `computronium/cli/commands/search.py` — search (HPO)
- `computronium/cli/commands/compare.py` — compare (study ranking)
- `computronium/cli/commands/verify.py` — verify (top-k re-runs)
- Original `run.py` preserved for backward compatibility

### ✅ SystemTrainer Decomposition (1,566 lines → package)
- `computronium/core/system_trainer/config.py` — JointSystem protocol, SystemTrainerConfig, TypeVars
- `computronium/core/system_trainer/trainer.py` — SystemTrainer class (training loop)
- `computronium/core/system_trainer/factory.py` — 5-D composition: compose_system, create_eqprop/backprop/fa_system
- `computronium/core/system_trainer/joint.py` — 6-D composition: compose_joint_system, create_routing_eqprop/fast_weight_eqprop_system
- `computronium/core/system_trainer/__init__.py` — Unified exports + continual learning re-exports
- Original `system_trainer.py` removed

### ✅ Tests Passing
- `tests/unit/nn/` — 26 passed
- `tests/unit/stability/` — 55 passed
- `tests/unit/core/test_registry.py` — 21 passed
- `tests/property/joint/test_composability.py` — 17 passed
- `tests/unit/core/test_system_spec.py` — 13 passed
- `tests/unit/validation/` — 22 passed

---

## Remaining Work (Prioritized)

### P0 — High Impact / Blocking
| Item | File | Lines | Notes |
|------|------|-------|-------|
| **knowledge/kb.py** | `computronium/knowledge/kb.py` | 1,642 | Decompose into: `entries.py`, `vector_store.py`, `query.py`, `surrogate.py`, `causal.py` |
| **deployment.py** | `computronium/deployment.py` | 1,635 | Decompose into: `exporter.py`, `onnx_export.py`, `pt2_export.py`, `quantization.py`, `serialization.py` |
| **core/local_learning/algorithm.py** | `computronium/core/local_learning/algorithm.py` | 1,446 | Decompose into: `protocols.py`, `feedback.py`, `activity.py`, `weight_update.py`, `builder.py` |
| **execution/strategy.py** | `computronium/execution/strategy.py` | 1,079 | Decompose into: `criteria.py`, `task_weights.py`, `candidate_gen.py`, `lifecycle.py` |

### P1 — Medium Impact
| Item | File | Lines | Notes |
|------|------|-------|-------|
| **cli/run.py** | Remove legacy file after verifying all commands work via new submodules | — | `python -m computronium.cli.run` still works; can deprecate |
| **Stability standalone tests** | `tests/unit/core/test_stability_standalone.py` | — | Requires `pip install -e .[stability]` + wheel build; CI integration |
| **State module imports** | `computronium/state/__init__.py` | — | LSP shows import errors; verify `computronium.state.composite` etc. exist |

### P2 — Nice to Have
| Item | Description |
|------|-------------|
| Pyright strict mode | 4,315 errors (mostly `reportUnknownMemberType` in tests); suppress or fix incrementally |
| Ruff line-length | Some files exceed 88 chars; run `ruff format` |
| Property test failures | Several `test_ontology_locks.py` failures (pre-existing, not regressions) |
| Documentation | Auto-generate API docs from docstrings (`pdoc`/`mkdocstrings`) |

---

## Execution Order

```
Session A (Week 1):  knowledge/kb.py + deployment.py
Session B (Week 1):  core/local_learning/algorithm.py + execution/strategy.py  
Session C (Week 2):  Verify all tests + CI gates (ruff, pyright, pytest)
Session D (Week 2):  Clean up legacy cli/run.py + update imports
Session E (Week 2):  Document API + migration guide
```

---

## Acceptance Criteria

- [ ] All files >1000 lines decomposed
- [ ] `ruff format --check . && ruff check . && pyright . && pytest --cov` green
- [ ] Coverage ≥85% for `computronium/ontology/`, `computronium/stability/`, `computronium/nn/`
- [ ] No import cycles (`pyright --verifytypes computronium`)
- [ ] Standalone wheel test passes (`tests/unit/core/test_stability_standalone.py`)
- [ ] Migration guide written for external consumers