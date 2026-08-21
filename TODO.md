# Codebase Cleanup Opportunities

Collected during domain registration removal. **Do not start** — just a plan.

---

## 1. Legacy Model Aliases & Duplicate Implementations

| Issue | Location | Action |
|-------|----------|--------|
| `BackpropMLP` lives in `eqprop/looped_mlp.py` but re-exported from `backprop.py` | `zoo/models/eqprop/looped_mlp.py:22`, `zoo/models/backprop.py:22` | Move `BackpropMLP` to `backprop.py`; remove re-export |
| `EquilibriumMLP` + `LoopedMLP` (facade) duplication | `zoo/models/eqprop/_energy.py`, `zoo/models/eqprop/looped_mlp.py` | Collapse: `LoopedMLP` is just a registration facade |
| `TileAlgorithm` + `TileAlgorithmConfig` + algorithm-specific variants | `core/local_learning/`, `zoo/models/deployments/*.py` | Consolidate: variants are just config presets, not classes |
| `*_legacy` modules still imported in places | `zoo/models/eqprop/_legacy/`, `docs/archive/` | Audit imports; remove if unused |

---

## 2. Registry Category Consolidation

| Category | Status | Note |
|----------|--------|------|
| `PROPAGATOR` vs `MODEL` | Overlapping | Many "propagators" are model-side learners (FF, TP, PCN) registered as models with aliases |
| `OPTIMIZER` vs `UPDATE_STRATEGY` | Split | `UPDATE_STRATEGY` = gradient transforms (Muon, Spectral); `OPTIMIZER` = torch.optim wrappers. Could unify with a `is_standalone` flag |
| `CONSTRAINT` | Underused | Only Spectral/Elastic registered. Could merge into `UPDATE_STRATEGY` with `when: "post_step"` |
| `CONTROLLER` | Minimal | Only `DynamicTileAlgorithm`. Consider if separate category justified |
| `TRACK`, `METRIC`, `KERNEL_BACKEND` | Sparse | Evaluate if registry overhead worth it for <3 entries each |

**Proposal**: Reduce to 4 core categories:
1. `MODEL` (includes model-side learners: FF, TP, PCN, Hebbian)
2. `CREDIT_ASSIGNMENT` (propagators: Backprop, FA, EP, TP, etc.)
3. `PARAM_UPDATE` (optimizers + update strategies + constraints)
4. `HARDWARE` (substrates, kernel backends, sparsity)

---

## 3. Configuration System Unification

| Config | Location | Overlap |
|--------|----------|---------|
| `ModelConfig` | `config/unified.py` | Base for all models |
| `TrainerConfig` | `core/trainer.py` | Training hyperparams |
| `*DeploymentConfig` | `zoo/models/deployments/base.py` | Vision/Graph/RL/TS-specific |
| `TileAlgorithmConfig` | `core/local_learning/algorithm.py` | TileNet-specific |
| `DataConfig` | `config/unified.py` | Dataset loading |
| `BenchmarkSuiteConfig` | `evaluation/cross_domain.py` | Benchmark params |

**Issue**: Same fields (`learning_rate`, `batch_size`, `epochs`) redefined in 5+ places with different defaults.

**Proposal**: Single `ExperimentConfig` with composition:
```python
@dataclass
class ExperimentConfig:
    model: ModelConfig      # architecture
    training: TrainingConfig  # lr, epochs, batch, optimizer
    data: DataConfig        # dataset, splits
    hardware: HardwareConfig  # device, precision, distributed
    # Domain-specific via inheritance or extra dict
```

---

## 4. CLI Entry Point Consolidation (Partially Done)

| Command | Status | Notes |
|---------|--------|-------|
| `eqprop-verify` | ✅ Removed | Replaced by `biopl parity` |
| `eqprop-p2p-worker` | ✅ Renamed | → `biopl-p2p-worker` |
| `biopl-run` / `biopl-report` / etc. | ✅ Subcommands | Now under `biopl` dispatcher |
| `biopl-scientist` | Keep standalone | Long-running autonomous loop |
| `biopl-failure-manifesto` | Keep standalone | Specialized report generator |
| `biopl-export-kernel*` | Keep standalone | Specialized export |

**Remaining**: `biopl-hpo`, `biopl-frontier`, `biopl-rank`, `biopl-audit`, `biopl-repro-check`, `biopl-parity` — evaluate if these should be `biopl` subcommands too.

---

## 5. Validation Tracks — Consolidate, Don't Delete

**Correction**: Validation tracks are **not** replaced by property tests. They serve different purposes:

| System | Purpose | Output |
|--------|---------|--------|
| **Property/Integration Tests** | CI gates, formal correctness | Pass/fail, coverage |
| **Validation Tracks** | Research evidence documentation | Human-readable markdown reports with evidence tables |

The `Verifier` class runs tracks at 3 evidence levels (smoke/intermediate/full) and produces `VerificationNotebook` markdown — this is **research documentation infrastructure**.

### Actual Cleanup Opportunities in Validation:

| Track Module | Status | Action |
|--------------|--------|--------|
| `core_tracks.py` (tracks 1-3) | **Keep** — Core claims (SN stability, EP-BP parity, self-healing) | Consolidate with biology axioms tests |
| `scaling_tracks.py` (tracks 12, 23-26, 35) | **Keep** — Scaling laws, deep scaling, O(1) memory | Move scaling law tests to `tests/property/` |
| `hardware_tracks.py` (tracks 16-18) | **Keep** — FPGA/INT8, analog noise, thermodynamic | Substrate property tests already cover S-axis |
| `application_tracks.py` (tracks 19-22) | **Evaluate** — Transfer, continual, golden ref | Cross-domain benchmarks cover some |
| `nebc_tracks.py` (tracks 50-54) | **Keep** — NEBC extension experiments | Could be property tests |
| `signal_tracks.py` + `tradeoff_tracks.py` | **Evaluate** — Signal propagation, tradeoff analysis | Research-specific; may not need automation |
| `research_tracks.py` | **Evaluate** — Ad-hoc research experiments | Likely one-off; document or remove |
| `negative_results.py` | **Keep** — Structured negative results | Valuable for AutoScientist |
| `architecture_comparison.py` | **Evaluate** — Architecture diffs | Could be `biopl lab` command |

**Goal**: 
- Keep tracks that produce **reusable evidence** for research claims
- Move **automatable invariants** (Lipschitz, energy descent, gradient equivalence) to property tests
- Remove **one-off research scripts** masquerading as tracks
- Unify `Verifier` output with `biopl report` / `biopl failure-manifesto`

---

## 6. Deprecated / Dead Code

| Path | Reason | Status |
|------|--------|--------|
| `bioplausible/validation/tracks/` (one-off tracks) | Not reusable evidence; research scripts | **Evaluate per track** (see §5) |
| `bioplausible/validation/tracks/advanced_tracks.py` | Deleted in Phase 4 (comment in track_registry) | **Already gone** |
| `bioplausible/validation/tracks/analysis_tracks.py` | Deleted in Phase 4 | **Already gone** |
| `bioplausible/validation/tracks/engine_validation_tracks.py` | Deleted in Phase 4 | **Already gone** |
| `bioplausible/validation/tracks/enhanced_validation_tracks.py` | Deleted in Phase 4 | **Already gone** |
| `bioplausible/validation/tracks/honest_tradeoff.py` | Deleted in Phase 4 | **Already gone** |
| `bioplausible/validation/tracks/new_tracks.py` | Deleted in Phase 4 | **Already gone** |
| `bioplausible/validation/tracks/rapid_validation.py` | Deleted in Phase 4 | **Already gone** |
| `bioplausible/validation/tracks/special_tracks.py` | Deleted in Phase 4 | **Already gone** |
| `bioplausible/validation/tracks/framework_validation.py` | Deleted in Phase 4 | **Already gone** |
| `docs/archive/` | Historical, not maintained | **Delete** |
| `examples/` | Tutorial notebooks; migrate to `demo/` or delete | **Evaluate** |
| `tools/benchmark_*.py` | One-off scripts; integrate into `biopl lab benchmark` | **Consolidate** |
| `tools/check_*.py` | CI checks; move to pre-commit hooks | **Move** |
| `run_experiment.py` | Legacy scientist runner; replaced by `biopl-scientist` | **Delete** |
| `run_scientist.sh` / `generate_report.sh` | Shell wrappers; replace with `uv run` commands | **Delete** |

---

## 7. Import Hygiene & Circular Dependency Risks

| Module | Imports | Risk |
|--------|---------|------|
| `core/registry.py` | `core/ontology.py` (for `to_system`) | Ontology imports registry → potential cycle |
| `core/trainer.py` | `core/ontology.py`, `zoo/` | Trainer shouldn't know about zoo |
| `execution/engine.py` | `hyperopt/`, `autoscientist/`, `zoo/` | Heavy import chain |
| `autoscientist/dashboard.py` | `nicegui`, `execution/`, `hyperopt/` | UI pulls entire stack |

**Fix**: Dependency injection / lazy imports / protocol-based interfaces.

---

## 8. Test Infrastructure Consolidation

| Issue | Detail |
|-------|--------|
| `tests/property/` + `tests/integration/` + `tests/unit/` | Three parallel hierarchies; property tests are the "real" CI gate |
| `tests/conftest.py` | 200+ lines of fixtures; split by domain |
| Coverage floor 55% but actual ~16% | Most code untested; property tests only cover ontology core |
| Hypothesis tests slow | Some take 30s+; consider marking `@pytest.mark.slow` |

---

## 9. Documentation Debt

| File | Issue |
|------|-------|
| `README.md` | Now has evaluation domains but still references old "Track 37" etc. |
| `AGENTS.md` | Mentions `Domain` enum (removed) |
| `CLAUDE.md` | If exists, likely outdated |
| `pyproject.toml` classifiers | Still says "Development Status :: 3 - Alpha" |

---

## 10. Type System Cleanup

| Pattern | Count | Fix |
|---------|-------|-----|
| `object` as type hint | ~50 | Replace with `Protocol` or `Any` with comment |
| `list[str] \| None` with `None` default | ~30 | Use `list[str] = field(default_factory=list)` |
| `cast()` in registry | ~20 | Improve generic signatures |
| `TYPE_CHECKING` imports for runtime-used types | ~10 | Move out of TYPE_CHECKING |

---

## Priority Order (if we were to execute)

1. **Config unification** — highest impact, touches everything
2. **Registry category reduction** — simplifies AutoScientist composition
3. **Validation tracks deletion** — removes ~2000 lines of dead code
4. **Model alias collapse** — reduces confusion in zoo
5. **CLI subcommand completion** — consistent UX
6. **Test infrastructure** — enables reliable CI
7. **Dead code removal** — reduces cognitive load
8. **Documentation sync** — prevents misinformation
9. **Type cleanup** — improves IDE support
10. **Import hygiene** — prevents circular deps

---

## Notes

- **No users** = no backward compatibility needed
- **Property tests are the spec** — if it passes L1-L7, it's valid
- **Ontology is the source of truth** — everything should compose via 5-D axes
- **AutoScientist drives requirements** — if it doesn't need it, delete it