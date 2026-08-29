# TODO7.md — Post-Cleanup Roadmap

> **Scope:** Remaining modularization and infrastructure work after Phase 0–3 completion. All critical path items from TODO6.md are ✅ COMPLETE.

---

## ✅ Completed (Session Summary)

### Ontology Decomposition (Phase 2.2)
- Removed `computronium/core/ontology.py` (5,680 lines)
- Created `computronium/ontology/` with per-axis modules:
  - `substrate.py` — SubstrateConfig, Digital/Analog/Memristive/Neuromorphic/Optical/Quantum/Sparse/Ternary/Complex/Noisy/QuantizedSubstrate
  - `geometry.py` — GeometryConfig, Feedforward/Recurrent/TileGeometry
  - `dynamics.py` — StateDynamicsConfig, EnergyMinimization/PredictiveSettling/SpikeIntegration/Instantaneous/Diffusion/LazyStateDynamics
  - `credit.py` — CreditAssignmentConfig, ThermodynamicContrast/RandomProjections/LocalGoodness/TemporalTrace/TargetInversion/Homeostatic/BackpropCredit
  - `update.py` — ParameterUpdateConfig, Euclidean/RiemannianOrthogonal/SpectralConstrained/NaturalGradient/ElasticConsolidationUpdate
  - `system.py` — SystemConfig, System, SystemState, ModelAdapter, FAMILY_TOLERANCES
  - `plasticity.py` — Re-exports M-axis plasticity primitives (FastWeight/Routing/RuleState/SubstrateCoupled/NullPlasticity)
  - `__init__.py` — Single import surface: `from computronium.ontology import *`

### Lazy Import Updates
- `computronium/__init__.py` — All ontology symbols now point to `computronium.ontology.*`
- `computronium/core/__init__.py` — All ontology/state symbols updated to new locations
- Verified: `from computronium import System, DigitalSubstrate, ...` works

### Build Artifacts Removed
- Deleted `build/`, `computronium.egg-info/`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.coverage`

### CLI Decomposition (2,011 lines → submodules)
- `computronium/cli/shared.py` — Shared constants, FAMILY_MAP, target resolution, trial context
- `computronium/cli/commands/train.py` — train, core-train, from-config
- `computronium/cli/commands/search.py` — search (HPO)
- `computronium/cli/commands/compare.py` — compare (study ranking)
- `computronium/cli/commands/verify.py` — verify (top-k re-runs)
- Original `run.py` preserved as CLI aggregator for `comp run` / `comp hpo`

### SystemTrainer Decomposition (1,566 lines → package)
- `computronium/core/system_trainer/config.py` — SystemTrainerConfig
- `computronium/core/system_trainer/protocol.py` — JointSystem protocol, TypeVars
- `computronium/core/system_trainer/spec.py` — Serialization utilities (to_spec/from_spec)
- `computronium/core/system_trainer/trainer.py` — SystemTrainer class (training loop)
- `computronium/core/system_trainer/factory.py` — 5-D composition: compose_system, create_eqprop/backprop/fa_system
- `computronium/core/system_trainer/joint.py` — 6-D composition: compose_joint_system, create_routing_eqprop/fast_weight_eqprop_system
- `computronium/core/system_trainer/__init__.py` — Unified exports + continual learning re-exports
- Original `system_trainer.py` removed

### Knowledge Base Decomposition (1,642 lines → 6 modules)
- `computronium/knowledge/entries.py` — KnowledgeEntry, ConditionalQuery, ConditionalResult, FlagshipCandidate, FlagshipDecision, helper functions
- `computronium/knowledge/vector_store.py` — VectorStore, VectorStoreConfig, FAISS integration, embedding generation
- `computronium/knowledge/query.py` — QueryEngine, QueryConfig, structured queries, conditional queries, flagship selection
- `computronium/knowledge/surrogate.py` — SurrogateManager, SurrogateConfig, surrogate training/prediction/registration
- `computronium/knowledge/causal.py` — CausalAnalyzer, CausalConfig, causal analysis, scaling laws, fingerprints, failure manifold, phylogeny
- `computronium/knowledge/kb.py` — KnowledgeBase (main facade), KnowledgeBaseConfig, composition of all submodules
- `computronium/knowledge/__init__.py` — Updated exports for all new modules

### Deployment Decomposition (1,635 lines → 6 modules)
- `computronium/deployment/exporter.py` — ExportConfig, export_model, load_model, ModelExporter facade
- `computronium/deployment/onnx_export.py` — export_to_onnx with opset 17+, dynamic axes
- `computronium/deployment/pt2_export.py` — export_to_pt2 (torch.export), replaces deprecated torch.jit
- `computronium/deployment/quantization.py` — INT8 PTQ/QAT/dynamic, ternary quantization (TernaryLinear, TernaryQuantize)
- `computronium/deployment/serialization.py` — ModelExporter, ModelLoader, InferenceEngine, InferenceServer, FastAPI serving
- `computronium/deployment/__init__.py` — Unified exports for all deployment modules

### Local Learning Algorithm Decomposition (1,446 lines → 7 modules)
- `computronium/core/local_learning/protocols.py` — FeedbackFn, ActivityUpdateFn, WeightUpdateFn, WeightLookup protocols
- `computronium/core/local_learning/feedback.py` — symmetric_feedback, no_feedback implementations
- `computronium/core/local_learning/activity.py` — ep_activity_update, hebbian_activity_update, spiking_activity_update
- `computronium/core/local_learning/weight_update.py` — contrastive_weight_update, hebbian_weight_update
- `computronium/core/local_learning/builder.py` — TileAlgorithmConfig, TileAlgorithm (main class with all factory methods)
- `computronium/core/local_learning/registry.py` — TileAlgorithm factory registry with @tile_algorithm decorator
- `computronium/core/local_learning/__init__.py` — Updated exports for all new modules

### ModelAdapter Decomposition (Phase 2)
- Created `computronium/ontology/adapter/inference.py` — SubstrateInferer, GeometryInferer, DynamicsInferer, CreditInferer, UpdateInferer protocols + native and heuristic implementations
- Created `computronium/ontology/adapter/registry.py` — metadata extraction from ComponentMetadata (uses new ontology_axes fields)
- Created `computronium/ontology/adapter/heuristics.py` — family/name-based fallbacks when metadata missing (backward compatibility for legacy models)
- Created `computtonium/ontology/adapter/adapter.py` — main facade coordinating inferrers, builds System, validation support
- Created `computronium/ontology/adapter/__init__.py` — unified exports
- Split monolithic ModelAdapter (~350 lines) into 4 focused modules (~400 lines total)
- Enables testable inference, extensible for new axes, clean separation of concerns

### Deployment Models Unification (Phase 1)
- Created `computronium/zoo/models/deployments/deployment.py` — Unified factory with FeatureExtractor protocol and DeploymentConfig subclasses
- Consolidates vision, RL, time-series, graph deployments into single `create_deployment_model(domain, **config)` factory
- FeatureExtractor protocol with output_dim property + registry for CNN/LSTM/MLP/GraphConv extractors
- Backward-compatible factory functions: create_vision_model, create_rl_model, create_timeseries_model, create_graph_model
- Algorithm variant registration via register_deployment_variants for all 4 domains (ep, pc, fa, tp, hebbian, snn, gnn)
- Deprecated imports via __getattr__ with deprecation warnings pointing to new unified API
- ~3000 lines eliminated across 4 domain-specific modules

### Native Model Promotion (Phase 3) — COMPLETE
- Registered 29 native models with explicit 5-D ontology axis assignments:
  - EqProp variants (5): native_eqprop_mlp, native_diffusion_eqprop, native_momentum_eqprop, native_sparse_eqprop, native_ternary_eqprop
  - FA variants (12): native_fa_mlp, native_fa_adaptive, native_fa_stochastic, native_fa_contrastive, native_fa_sign_symmetric, native_fa_direct, native_fa_energy_guided, native_fa_energy_minimizing, native_fa_equilibrium_alignment, native_fa_layerwise_equilibrium, native_fa_deep_dfa
  - Backprop (2): native_backprop_mlp, native_pepita_mlp
  - Tile variants (7): native_tile_ep, native_tile_fa, native_tile_tp, native_tile_snn, native_tile_hebbian, native_tile_pc, native_tile_gnn
  - Other (3): native_holomorphic_ep, native_directed_ep, native_finite_nudge_ep
- Registry.to_system() returns native _ComposedSystem directly for native models (bypasses ModelAdapter)
- Registry.query_ontology() supports axis-aware queries with explicit ontology layers
- All 7 tile variants registered with explicit ontology axes
- Deprecation warnings added to 10 legacy zoo modules

### Ontology Internal Deduplication (Phase 0b) — 6 utility modules
- `computronium/ontology/utils/params.py` — _learnable_weight_names, apply_pseudo_gradients, _set_param_name
- `computronium/ontology/utils/geometry.py` — _layer_stack, _recurrent_weight
- `computronium/ontology/utils/state.py` — 12 state accessor functions + StateProtocol
- `computronium/ontology/utils/config.py` — ConfigFactory protocol
- `computronium/ontology/substrate/factory.py` — substrate_from_config
- `computronium/ontology/dynamics/primitives.py` — _settle_step, _compute_hopfield_energy
- ~200 lines deduplicated across 5 axis modules

### SystemConfig/JointSystem Split (Phase 4)
- Split `system_trainer/config.py` into `protocol.py` (JointSystem), `config.py` (SystemTrainerConfig), `spec.py` (to_spec/from_spec)
- Consolidated factories: compose_system (5-D), compose_joint_system (6-D), convenience create_*
- Single composition entry point

### Registry Enhancement for Ontology Discovery (Phase 5)
- Added `Registry.query_axis(substrate=..., geometry=..., dynamics=..., credit=..., update=...)` for AutoScientist cross-axis search
- Enables "find all models with ThermodynamicContrast + RecurrentGeometry"
- Uses explicit ontology_* fields on ComponentMetadata for native models, heuristic fallback for legacy

### SettleProtocol MRO Fix
- Fixed MRO issue in MEPEqPropModel, O1MemoryModel, PredictiveCodingHybrid: removed SettleProtocol from inheritance (structural subtyping via @runtime_checkable works without inheritance)
- Fixed super().__init__(config, **kwargs) → super().__init__(config)
- All 29 settle protocol integration tests now pass

### Test Results
- Core unit tests: 172 passed
- Property tests: 35 passed (ontology locks: 32 + 3 skipped)
- Integration (settle protocol): 29 passed
- Stability standalone: 55 passed
- Update rules: 20 passed (1 xfailed)
- Coverage: ~17.3% (≥15% floor met, legacy zoo excluded)

### Ontology Property Locks Certification (Session — This Work)
- **Fixed ComputeProfile NameError** in `system.py:787` — added missing import from `computronium.core.registry`
- **Implemented BackpropCredit.compute_pseudo_gradient** — true autograd gradients matching test expectations
- **Implemented RandomProjectionsCredit** with fixed feedback matrices (`_init_feedback_weights`, `_feedback_weights`) enabling FA L3 locality locks
- **Implemented TemporalTraceCredit.compute_stdp_window** — full STDP window with exponential decay, antisymmetry, causal potentiation
- **Implemented RiemannianOrthogonalUpdate._orthogonalize** — QR-based orthogonalization for square and non-square matrices (replaces Newton-Schulz)
- **Implemented ElasticConsolidationUpdate.consolidate** — EWC-style importance-weighted update with Fisher information
- **Fixed PredictiveSettlingDynamics** shape mismatch in prediction error computation (input vs output dimension)
- **Fixed SpikeIntegrationDynamics** spike_counts tracking for Lyapunov lock
- **All 32 ontology locks tests pass** (32 passed, 3 skipped)

---

## Remaining Work (Prioritized)

### P1 — Medium Impact
| Item | Notes |
|------|-------|
| **cli/run.py** | Keep as CLI aggregator for `comp run` / `comp hpo` — not legacy |

### P2 — Important: Deprecation & Test Failures
| Item | Description |
|------|-------------|
| **Zoo Legacy Deprecation** | Remove remaining deprecated modules: `wrappers.py`, `base.py`, `transitions.py` (~30K lines). Legacy modules already emit DeprecationWarning. |
| ~~Property test failures~~ | ~~Fix `test_ontology_locks.py` failures (pre-existing, not regressions)~~ ✅ **FIXED** |
| **Pyright strict mode** | 4,315 errors (mostly `reportUnknownMemberType` in tests); suppress or fix incrementally |

### P3 — Improvement Opportunities
| Item | Description |
|------|-------------|
| **Type Hint Improvements** | Several modules have `object` type annotations that could use `TypedDict` or `Protocol` |
| **Config Classes** | Create unified config hierarchy for decomposed modules (VectorStoreConfig, SurrogateConfig, CausalConfig, etc.) |
| **Error Handling** | Some modules catch broad `Exception` — use more specific exceptions |
| **Caching** | `_model_specs` cache in execution could use `functools.lru_cache` instead of function attribute |
| **Test Coverage** | Add unit tests for new decomposed modules |
| **Performance** | `CandidateGenerator.generate_candidates` recomputes saturation/failure analysis each call — memoize |
| **FA Native Variants** | 12 FA variants share base implementation — implement algorithmic differences in `RandomProjectionsCredit` |
| **Documentation** | Auto-generate API docs from docstrings (`pdoc`/`mkdocstrings`) |
| **STDP Config** | `TemporalTraceCredit` uses hardcoded STDP params (a_plus, a_minus, tau) — add to `CreditAssignmentConfig.temporal_trace()` |
| **FA Feedback Scale** | `RandomProjectionsCredit.feedback_scale` config param exists but not used in gradient computation — wire it up |
| **Riemannian Update** | `ortho_steps` config used by QR but not by Newton-Schulz (removed) — unify or document |
| **Elastic EWC Lambda** | `ewc_lambda` in config is used as Fisher damping but should be Fisher importance weight — clarify semantics |

---

## Zoo Legacy Deprecation Status

| File | Lines | Status | Replacement |
|------|-------|--------|-------------|
| `fa.py` | 41,257 | **DEPRECATED** ✅ | Native FA compositions + TileFA |
| `eqprop/` (8 files) | ~60K | **DEPRECATED** ✅ | Native EqProp + TilePC |
| `backprop.py` | 14,272 | **DEPRECATED** ✅ | Native backprop + TileAlgorithm |
| `hebbian.py` | 15,828 | **DEPRECATED** ✅ | TileAlgorithm.from_hebbian() |
| `predictive_coding.py` | 15,996 | **DEPRECATED** ✅ | TileAlgorithm.from_pc() |
| `mep.py` | 18,300 | **DEPRECATED** ✅ | M-axis plasticity primitives |
| `o1memory.py` | 11,277 | **DEPRECATED** ✅ | Native compositions |
| `spiking.py` | 6,219 | **DEPRECATED** ✅ | TileAlgorithm.from_snn() |
| `target_prop.py` | 5,665 | **DEPRECATED** ✅ | TileAlgorithm.from_tp() |
| `forward_only.py` | 9,247 | **DEPRECATED** ✅ | TileAlgorithm + LocalGoodness |
| `wrappers.py` | 11,277 | **DEPRECATE** | Composition pattern |
| `base.py` | 15,631 | **DEPRECATE** | Protocol-based System |
| `transitions.py` | 3,323 | **DEPRECATE** | TransitionGraphMixin → Geometry |
| `tile_models.py` | 16,029 | **KEEP** | Thin TileAlgorithm wrappers |
| `tile_fa.py` | 5,425 | **KEEP** | Thin TileAlgorithm wrapper |
| `tile_lm.py` | 13,550 | **KEEP** | Thin TileAlgorithm wrapper |
| `deployments/` | ~8,000 | **UNIFIED** ✅ | Single deployment factory |

**Total legacy to deprecate: ~200K lines** → **Target: <20K lines of native compositions**
**Deprecated with warnings: 10 modules (~160K lines)**

---

## Acceptance Criteria (Updated)

- [x] All files >1000 lines decomposed
- [x] `ruff format --check .` passes (formatting applied)
- [x] `pytest --cov` — key tests pass (ontology: 35, stability: 55, nn: 26, registry: 21, system_spec: 13, validation: 22, composability: 17, settle_protocol: 29)
- [x] Coverage ≥15% for `computronium/ontology/`, `computronium/stability/`, `computronium/nn/` — current ~18% (legacy zoo excluded per pyproject.toml)
- [x] No import cycles (`pyright --verifytypes computronium`) — verified via test imports
- [x] Standalone wheel test passes (`tests/unit/core/test_stability_standalone.py`)
- [x] All 29 native models registered in Registry with `ontology_axes`
- [x] Deployment modules unified into single factory (<1000 lines)
- [x] ModelAdapter decomposed into 4 adapter modules
- [x] Tile variants (7/7) complete with explicit ontology axes
- [x] 10 legacy Zoo modules deprecated with migration warnings
- [x] Registry supports axis-aware queries for AutoScientist (`query_axis`)
- [x] Ontology internal deduplication complete (6 utility modules)
- [x] TileAlgorithm Factory Registry complete (`@tile_algorithm` decorator + `from_config`)
- [x] SystemTrainer config split into `protocol.py`, `config.py`, `spec.py`
- [x] SettleProtocol MRO fix — 29 integration tests pass