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
- `computtonium/cli/commands/compare.py` — compare (study ranking)
- `computronium/cli/commands/verify.py` — verify (top-k re-runs)
- Original `run.py` preserved as CLI aggregator for `comp run` / `comp hpo`

### ✅ SystemTrainer Decomposition (1,566 lines → package)
- `computronium/core/system_trainer/config.py` — JointSystem protocol, SystemTrainerConfig, TypeVars
- `computronium/core/system_trainer/trainer.py` — SystemTrainer class (training loop)
- `computronium/core/system_trainer/factory.py` — 5-D composition: compose_system, create_eqprop/backprop/fa_system
- `computronium/core/system_trainer/joint.py` — 6-D composition: compose_joint_system, create_routing_eqprop/fast_weight_eqprop_system
- `computronium/core/system_trainer/__init__.py` — Unified exports + continual learning re-exports
- Original `system_trainer.py` removed

### ✅ Knowledge Base Decomposition (1,642 lines → 6 modules)
- `computronium/knowledge/entries.py` — KnowledgeEntry, ConditionalQuery, ConditionalResult, FlagshipCandidate, FlagshipDecision, helper functions
- `computronium/knowledge/vector_store.py` — VectorStore, VectorStoreConfig, FAISS integration, embedding generation
- `computronium/knowledge/query.py` — QueryEngine, QueryConfig, structured queries, conditional queries, flagship selection
- `computronium/knowledge/surrogate.py` — SurrogateManager, SurrogateConfig, surrogate training/prediction/registration
- `computronium/knowledge/causal.py` — CausalAnalyzer, CausalConfig, causal analysis, scaling laws, fingerprints, failure manifold, phylogeny
- `computronium/knowledge/kb.py` — KnowledgeBase (main facade), KnowledgeBaseConfig, composition of all submodules
- `computronium/knowledge/__init__.py` — Updated exports for all new modules

### ✅ Deployment Decomposition (1,635 lines → 6 modules)
- `computronium/deployment/exporter.py` — ExportConfig, export_model, load_model, ModelExporter facade
- `computtonium/deployment/onnx_export.py` — export_to_onnx with opset 17+, dynamic axes
- `computronium/deployment/pt2_export.py` — export_to_pt2 (torch.export), replaces deprecated torch.jit
- `computronium/deployment/quantization.py` — INT8 PTQ/QAT/dynamic, ternary quantization (TernaryLinear, TernaryQuantize)
- `computronium/deployment/serialization.py` — ModelExporter, ModelLoader, InferenceEngine, InferenceServer, FastAPI serving
- `computronium/deployment/__init__.py` — Unified exports for all deployment modules

### ✅ Local Learning Algorithm Decomposition (1,446 lines → 6 modules)
- `computronium/core/local_learning/protocols.py` — FeedbackFn, ActivityUpdateFn, WeightUpdateFn, WeightLookup protocols
- `computronium/core/local_learning/feedback.py` — symmetric_feedback, no_feedback implementations
- `computronium/core/local_learning/activity.py` — ep_activity_update, hebbian_activity_update, spiking_activity_update
- `computronium/core/local_learning/weight_update.py` — contrastive_weight_update, hebbian_weight_update
- `computronium/core/local_learning/builder.py` — TileAlgorithmConfig, TileAlgorithm (main class with all factory methods)
- `computronium/core/local_learning/__init__.py` — Updated exports for all new modules

### ✅ Execution Strategy Decomposition (1,079 lines → 4 modules)
- `computronium/execution/criteria.py` — CRITERIA dict, check_criterion with task-specific overrides
- `computronium/execution/task_weights.py` — TASK_WEIGHTS, TASK_GROUPS, TIER_ORDER, calculate_future_boost, calculate_complexity_penalty
- `computronium/execution/candidate_gen.py` — CandidateGenerator, ExecutionStrategyConfig, full candidate generation logic
- `computronium/execution/lifecycle.py` — ExecutionStrategy, plan_next, plan_batch
- `computronium/execution/__init__.py` — Updated lazy exports for all new modules

### ✅ Phase 0: Register Native Models + Ontology Axes Metadata
- Added `ontology_substrate`, `ontology_geometry`, `ontology_dynamics`, `ontology_credit`, `ontology_update` fields to `ComponentMetadata`
- Registered 15 native models with explicit 5-D ontology axis assignments:
  - `native_eqprop_mlp`, `native_diffusion_eqprop`, `native_momentum_eqprop`, `native_sparse_eqprop`, `native_ternary_eqprop`
  - `native_fa_mlp`, `native_backprop_mlp`, `native_pepita_mlp`
  - `native_tile_ep`, `native_tile_fa`, `native_tile_tp`, `native_tile_snn`
  - `native_holomorphic_ep`, `native_directed_ep`, `native_finite_nudge_ep`
- `Registry.to_system()` now returns native `_ComposedSystem` directly for native models (bypasses ModelAdapter)
- `Registry.query_ontology()` supports axis-aware queries with explicit ontology layers

### ✅ Phase 0b: Ontology Internal Deduplication (6 utility modules)
- `computronium/ontology/utils/params.py` — `_learnable_weight_names`, `apply_pseudo_gradients`, `_set_param_name`
- `computronium/ontology/utils/geometry.py` — `_layer_stack`, `_recurrent_weight`
- `computtonium/ontology/utils/state.py` — 12 state accessor functions + `StateProtocol`
- `computronium/ontology/utils/config.py` — `ConfigFactory` protocol
- `computronium/ontology/substrate/factory.py` — `substrate_from_config`
- `computronium/ontology/dynamics/primitives.py` — `_settle_step`, `_compute_hopfield_energy`
- Updated `credit.py`, `update.py`, `system.py` to import from utils
- ~200 lines deduplicated across 5 axis modules

### ✅ Tests Passing
- `tests/unit/nn/` — 26 passed
- `tests/unit/stability/` — 55 passed
- `tests/unit/core/test_registry.py` — 21 passed
- `tests/property/joint/test_composability.py` — 17 passed
- `tests/unit/core/test_system_spec.py` — 13 passed
- `tests/unit/validation/` — 22 passed
- All imports verified working

---

## Remaining Work (Prioritized)

### P0 — High Impact / Blocking
| Item | File | Lines | Status |
|------|------|-------|--------|
| **knowledge/kb.py** | `computronium/knowledge/kb.py` | 1,642 | ✅ COMPLETE |
| **deployment.py** | `computronium/deployment.py` | 1,635 | ✅ COMPLETE |
| **core/local_learning/algorithm.py** | `computronium/core/local_learning/algorithm.py` | 1,446 | ✅ COMPLETE |
| **execution/strategy.py** | `computronium/execution/strategy.py` | 1,079 | ✅ COMPLETE |

### P1 — Medium Impact
| Item | File | Lines | Notes |
|------|------|-------|-------|
| **cli/run.py** | Keep as CLI aggregator for `comp run` / `comp hpo` | — | Not legacy - used by `comp` dispatcher |
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

## 🎯 Strategic Ontology Empowerment & Zoo Legacy Deprecation

### Phase 0: Foundation — Ontology Axis Completeness
| Item | Description | Impact | Status |
|------|-------------|--------|--------|
| **Register Native Models** | Register 15 `computronium/models/native/*.py` factories in Registry with explicit ontology axes | Enables native 5-D discovery, removes ModelAdapter dependency for new models | ✅ COMPLETE |
| **Add `ontology_axes` to ComponentMetadata** | Added explicit axis fields to `ComponentMetadata` | Eliminates inference errors, enables cross-axis ablation via Registry queries | ✅ COMPLETE |
| **Complete TileAlgorithm Factory Registry** | Add `@tile_algorithm` decorator registering each factory method (`from_ep`, `from_fa`, etc.) with algorithm metadata; enables `TileAlgorithm.from_config(config)` single entry point | Removes string matching in `_resolve_*`, config-driven composition | 🔄 PENDING |

### Phase 0b: Ontology Internal Deduplication (NEW — Immediate ROI)
| Item | Description | Files Affected | Effort | Status |
|------|-------------|----------------|--------|--------|
| **Shared Parameter Helpers** | Extract `_learnable_weight_names`, `apply_pseudo_gradients`, `_set_param_name` to `ontology/utils/params.py` | `credit.py`, `update.py`, `system.py`, `geometry.py` | 30 min | ✅ COMPLETE |
| **Shared Geometry Introspection** | Extract `_layer_stack`, `_recurrent_weight` to `ontology/utils/geometry.py` | `geometry.py`, `system.py` | 20 min | ✅ COMPLETE |
| **Shared State Accessors** | Extract 12 getter/setter functions to `ontology/utils/state.py` with `StateProtocol` | `dynamics.py` | 30 min | ✅ COMPLETE |
| **Config Factory Base** | `ConfigFactory` protocol in `ontology/utils/config.py` for unified `to_spec`/`from_spec`/`validate` | All axis configs | 40 min | ✅ COMPLETE |
| **Substrate Factory** | Move `substrate_from_config` to `ontology/substrate/factory.py` | `substrate.py`, `system.py` | 20 min | ✅ COMPLETE |
| **Dynamics Primitives** | Extract `_settle_step`, `_compute_hopfield_energy` to `ontology/dynamics/primitives.py` | `dynamics.py` | 30 min | ✅ COMPLETE |
| **Total** | **~3 hours** for ~200 lines deduplicated across 5 axis modules | | | ✅ COMPLETE |

### Phase 1: Deployment Models Unification (Eliminates ~3000 lines)
| Item | Description | Impact |
|------|-------------|--------|
| **Consolidate Vision/RL/Timeseries/Graph** | Single `computronium/zoo/models/deployments/deployment.py` with `FeatureExtractor` protocol + `DeploymentConfig` subclasses; factory `create_deployment_model(config, extractor)` | Removes 4 duplicate modules (~3000 lines), new deployment types in 50 lines |
| **Extract FeatureExtractors Protocol** | `FeatureExtractor` protocol with `output_dim` property + registry for CNN/LSTM/MLP/GraphConv; used by deployment factory | Enables mixing/matching extractors with TileAlgorithm heads |
| **Deprecate per-deployment modules** | Mark `vision.py`, `rl.py`, `timeseries.py`, `graph.py` deprecated; imports redirect to unified factory | Clear migration path, eliminates duplicate `DeploymentModel` boilerplate |

### Phase 2: ModelAdapter Decomposition (Empowers Strangler Fig)
| Item | Description | Impact |
|------|-------------|--------|
| **Split inference logic** | `ontology/adapter/inference.py` — `SubstrateInferer`, `GeometryInferer`, `DynamicsInferer`, `CreditInferer`, `UpdateInferer` protocols + implementations | Testable inference, extensible for new axes |
| **Extract registry metadata** | `ontology/adapter/registry.py` — metadata extraction from `ComponentMetadata` (uses new `ontology_axes` fields) | Single source of truth for axis projection |
| **Heuristics fallback** | `ontology/adapter/heuristics.py` — family/name-based fallbacks when metadata missing | Backward compatibility for legacy models |
| **Main Adapter** | `ontology/adapter/adapter.py` — coordinates inferrers, builds System | Clean facade, ~100 lines |

### Phase 3: Native Model Promotion (Replaces Legacy Zoo)
| Legacy Family | Count | Native Replacement Strategy |
|---------------|-------|------------------------------|
| **eqprop** (17) | 17 | Use `create_native_eqprop_mlp` + variants; register as `native_eqprop_*` |
| **fa** (12) | 12 | Create `create_native_fa_mlp` with configurable feedback (fixed/adaptive/stochastic/contrastive); register `native_fa_*` |
| **hebbian** (4) | 4 | Use `TileAlgorithm.from_hebbian()` (already native); register tile variants |
| **backprop** (3) | 3 | Use `create_native_backprop_mlp` with `BackpropCredit` + `EuclideanUpdate`; register `native_backprop_*` |
| **forward_only** (2) | 2 | Use `LocalGoodness` credit + `TileAlgorithm` mode="pc"; register `native_ff_*` |
| **predictive_coding** (2) | 2 | Use `TileAlgorithm.from_pc()` (already native); register tile variants |
| **spiking/target_prop** (2) | 2 | Use `TileAlgorithm.from_snn()` / `from_tp()` (already native); register tile variants |

**Goal**: Replace 38 legacy models with <10 native compositions, each configurable across axes.

### Phase 4: SystemConfig/JointSystem Split (Follows Ontology Pattern)
| Item | Description | Impact |
|------|-------------|--------|
| **Split `system_trainer/config.py`** | `protocol.py` (JointSystem), `config.py` (SystemTrainerConfig), `spec.py` (to_spec/from_spec) | Follows ontology modularization, cleaner imports |
| **Consolidate factories** | Single `compose.py` with `compose_system` (5-D), `compose_joint_system` (6-D), convenience `create_*` | Single composition entry point |

### Phase 5: Registry Enhancement for Ontology Discovery
| Item | Description | Impact |
|------|-------------|--------|
| **Axis-aware queries** | Add `Registry.query(axis=..., substrate=..., credit=...)` for AutoScientist cross-axis search | Enables "find all models with ThermodynamicContrast + RecurrentGeometry" |
| **Native model registration** | Register 15 `models/native/*.py` factories with full `ontology_axes` metadata | AutoScientist discovers native compositions directly | ✅ COMPLETE |
| **Deprecation warnings** | Add `@deprecated` to legacy model registrations pointing to native replacements | Clear migration path for consumers |

---

## Execution Order (Optimized)

```
Week 1 (Current):  ✅ P0 Decompositions COMPLETE
                   ✅ Phase 0: Register native models + ontology_axes metadata
                   ✅ Phase 0b: Ontology Internal Deduplication (3 hrs, high ROI)
Week 2:             Phase 1: Deployment Models Unification
                    Phase 2: ModelAdapter Decomposition  
Week 3:             Phase 3: Native Model Promotion (eqprop/fa/backprop)
                    Phase 4: SystemConfig/JointSystem Split
Week 4:             Phase 5: Registry Enhancement + Deprecation
                    Documentation + Migration Guide
```

---

## Acceptance Criteria (Updated)

- [x] All files >1000 lines decomposed
- [ ] `ruff format --check . && ruff check . && pyright . && pytest --cov` green
- [ ] Coverage ≥85% for `computronium/ontology/`, `computronium/stability/`, `computronium/nn/`
- [ ] No import cycles (`pyright --verifytypes computronium`)
- [ ] Standalone wheel test passes (`tests/unit/core/test_stability_standalone.py`)
- [ ] Migration guide written for external consumers
- [x] **All 15 native models registered in Registry with `ontology_axes`**
- [ ] **Deployment modules unified into single factory (<1000 lines)**
- [ ] **ModelAdapter decomposed into 4 adapter modules**
- [ ] **38 legacy Zoo models replaced by <10 native compositions**
- [ ] **Registry supports axis-aware queries for AutoScientist**
- [x] **Ontology internal deduplication complete (6 utility modules)**

---

## New Improvement Opportunities (Discovered During Decomposition)

1. **Type Hint Improvements**: Several modules have `object` type annotations that could be made more specific using `TypedDict` or `Protocol`.

2. **Config Classes**: Each decomposed module now has its own config class (e.g., `VectorStoreConfig`, `SurrogateConfig`, `CausalConfig`). Consider creating a unified config hierarchy.

3. **Error Handling**: Some modules catch broad `Exception` - could be more specific.

4. **Async Support**: The `InferenceServer` has async batching but `KnowledgeBase` methods are synchronous. Consider async variants for I/O-bound operations.

5. **Caching**: The `_model_specs` cache in execution could use `functools.lru_cache` instead of function attribute.

6. **Test Coverage**: Add unit tests for new modules: `entries.py`, `vector_store.py`, `query.py`, `surrogate.py`, `causal.py`, `protocols.py`, `feedback.py`, `activity.py`, `weight_update.py`, `builder.py`, `criteria.py`, `task_weights.py`, `candidate_gen.py`, `lifecycle.py`, `exporter.py`, `onnx_export.py`, `pt2_export.py`, `quantization.py`, `serialization.py`, `params.py`, `geometry.py`, `state.py`, `config.py`, `factory.py`, `primitives.py`.

7. **Performance**: The `CandidateGenerator.generate_candidates` method recomputes saturation/failure analysis each call - could memoize or compute incrementally.

8. **Backward Compatibility**: The original `algorithm.py` and `strategy.py` files were removed. Ensure any external consumers are updated via migration guide.

9. **CLI Refactoring**: The `cli/run.py` serves as the main CLI aggregator for `comp run` / `comp hpo`. Consider moving its command definitions to `cli/commands/` for consistency.

---

## Zoo Legacy Deprecation Status

| File | Lines | Status | Replacement |
|------|-------|--------|-------------|
| `fa.py` | 41,257 | **LEGACY — DEPRECATE** | Native FA compositions + TileFA |
| `eqprop/` (8 files) | ~60K | **LEGACY — DEPRECATE** | Native EqProp + TilePC |
| `backprop.py` | 14,272 | **LEGACY — DEPRECATE** | Native backprop + TileAlgorithm |
| `hebbian.py` | 15,828 | **LEGACY — DEPRECATE** | TileAlgorithm.from_hebbian() |
| `predictive_coding.py` | 15,996 | **LEGACY — DEPRECATE** | TileAlgorithm.from_pc() |
| `mep.py` | 18,300 | **LEGACY — DEPRECATE** | M-axis plasticity primitives |
| `o1memory.py` | 11,277 | **LEGACY — DEPRECATE** | Native compositions |
| `spiking.py` | 6,219 | **LEGACY — DEPRECATE** | TileAlgorithm.from_snn() |
| `target_prop.py` | 5,665 | **LEGACY — DEPRECATE** | TileAlgorithm.from_tp() |
| `forward_only.py` | 9,247 | **LEGACY — DEPRECATE** | TileAlgorithm + LocalGoodness |
| `wrappers.py` | 11,277 | **LEGACY — DEPRECATE** | Composition pattern |
| `base.py` | 15,631 | **LEGACY — DEPRECATE** | Protocol-based System |
| `transitions.py` | 3,323 | **LEGACY — DEPRECATE** | TransitionGraphMixin → Geometry |
| `tile_models.py` | 16,029 | **ONTOLOGY-NATIVE — KEEP** | Thin TileAlgorithm wrappers |
| `tile_fa.py` | 5,425 | **ONTOLOGY-NATIVE — KEEP** | Thin TileAlgorithm wrapper |
| `tile_lm.py` | 13,550 | **ONTOLOGY-NATIVE — KEEP** | Thin TileAlgorithm wrapper |
| `deployments/` | ~8,000 | **REFactor — UNIFY** | Single deployment factory |

**Total legacy to deprecate: ~200,000 lines** → **Target: <20,000 lines of native compositions**

---

## Ontology Internal Deduplication Details (Phase 0b)

| Utility Module | Functions/Classes | Source Files | Consumers |
|----------------|-------------------|--------------|-----------|
| `ontology/utils/params.py` | `_learnable_weight_names`, `apply_pseudo_gradients`, `_set_param_name` | `credit.py:266`, `update.py:201`, `system.py:44`, `geometry.py:116` | CreditAssignment, ParameterUpdate, System, Geometry |
| `ontology/utils/geometry.py` | `_layer_stack`, `_recurrent_weight` | `geometry.py:121`, `system.py:77` | Geometry, System, Dynamics |
| `ontology/utils/state.py` | `_is_composite_state`, `_get_state_*`, `_set_state_*` (12 functions) + `StateProtocol` | `dynamics.py:25-132` | All StateDynamics implementations |
| `ontology/utils/config.py` | `ConfigFactory` protocol (`to_spec`, `from_spec`, `validate`) | All 5 axis config classes | Registry, AutoScientist, serialization |
| `ontology/substrate/factory.py` | `substrate_from_config` | `substrate.py:658`, `system.py:1067` | System, deployment factories |
| `ontology/dynamics/primitives.py` | `_settle_step`, `_compute_hopfield_energy` | `dynamics.py:418,510` | EnergyMinimizationDynamics, PredictiveSettlingDynamics |

**Total: ~200 lines deduplicated, 6 new utility modules, single source of truth for cross-axis logic**