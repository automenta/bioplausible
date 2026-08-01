# Bioplausible Development Plan (Revised)

**Goal**: Build a credible, GPU-accelerated bio-plausible learning framework with an interactive demo that proves biology — not just plumbing. The demo + passing test suite = viability proof for researchers and contributors.

**Principle**: No cosmetic/lint work until functional milestones land. GPU for heavy tests only. All Tier 1 architecture from RESEARCH.pre.md folded in. RESEARCH.md stays as long-term agenda; this TODO is the only short-term plan.

---

## Session Log

*(New sessions append here)*

---

## Sprint 0: Architecture Foundations (Weeks 1–2)

*Folds RESEARCH.pre.md Tier 1 (1.1–1.6) — high-leverage refactors that unblock everything downstream.*

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| **0.1** | **Domain Exception Hierarchy** (`core/exceptions.py`) — base `BioplausibleError` + `ConfigError`, `RegistryError`, `IncompatibilityError`, `CheckpointError`, `LoadStateError`, `KnowledgeBaseError`, `TrialExecutionError`, `PropagatorError`, `TileGraphError`. Replace 127 bare `except Exception` with narrow+chain. | | ☐ | `pyright` 0 errors; `grep -r "except Exception" bioplausible/ | wc -l` → 0 in lib code |
| **0.2** | **`_QueryFilter` Predicate Dispatch** (`core/registry.py:120-165`) — convert boolean mega-expression to frozen predicate dataclasses + protocol; `matches()` = `all(p(meta) for p in predicates)`. Enables hypothesis tests + AutoScientist capability matching. | | ☐ | Property tests for each predicate axis; registry audit passes |
| **0.3** | **Cyclomatic Complexity Extraction** — hot paths only: `engine.py:_run_discovery_loop` (cc=17), `engine.py:_process_with_retry` (cc=12), `equitile/model.py:_relax` (cc=16), `equitile/model.py:_apply_hebbian_updates` (cc=13). Extract `_`-prefixed helpers with guard clauses. | | ☐ | `ruff check --select C901` = 0 on these files; snapshot tests for `_relax`/`_apply_hebbian_updates` |
| **0.4** | **`match`/`case` Conversion** — closed-enum chains: `equitile/model.py:_get_activation` (5-way), `equitile/model.py:train_step` (3-way mode), `engine.py:_log_task_start` (after dataclass extraction), `engine.py:_prepare_fixed_config` (after dataclass extraction). | | ☐ | Exhaustiveness checking catches new variants; no regressions |
| **0.5** | **Module Boundary Hardening** — `bioplausible/__init__.py`: split heavy registration into `_register_all.py`; `equitile/utils/` → `_utils/` or `_internal/`; verify no external imports of `_internal/`. | | ☐ | `import bioplausible.types` doesn't trigger model registration; `ruff` TID252 clean |
| **0.6** | **SQLite Resource Standardization** — `execution/_state.py`: replace 12+ manual `try/finally` with `@contextmanager _connect(db_path)` helper matching `kb.py` pattern. | | ☐ | No resource leaks under stress; KB meta-analysis (RESEARCH.md 4.2) unblocked |

**Gate**: `uv run pytest tests/unit/ tests/property/ -q --no-cov` < 60s, 0 failures; `pyright` 0 errors.

---

## Sprint 1: GPU-Accelerated Test Infrastructure (Week 2–3)

*Selective GPU: unit/property stay CPU (fast, deterministic); integration/large-model/benchmark tests run on GPU.*

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| **1.1** | **GPU Test Fixtures** (`tests/conftest.py`) — `device` fixture: `cuda` if available else `cpu`; `gpu_only` marker skips on CPU; `synthetic_batch_gpu`, `synthetic_vision_task_gpu`, `synthetic_lm_task_gpu` session-scoped on CUDA. | | ☐ | `pytest -m gpu_only` runs on RTX 3080; CPU suite unchanged |
| **1.2** | **Migrate Heavy Tests to GPU** — move `tests/integration/test_equitile_sparsity_robustness.py`, `test_lm_demo.py`, `test_triton_*.py`, `test_deq.py` (memory tests) to `@pytest.mark.gpu` + GPU fixtures. | | ☐ | GPU suite ~2-3x faster than CPU; memory tests use `torch.cuda.max_memory_allocated()` |
| **1.3** | **Benchmark Harness** (`tests/unit/validation/benchmark_harness.py`) — parametrized `@pytest.mark.benchmark` tests: FLOPs, peak memory, wall-time per model family (EqProp, FA, MEP, EquiTile, FF/PEPITA, Spiking). Uses `torch.profiler` + `torch.cuda.memory`. | | ☐ | `pytest tests/unit/validation/benchmark_harness.py -m benchmark` produces JSONL for Pareto plots |
| **1.4** | **Deterministic GPU Seeding** — extend `utils/reproducibility.py`: `set_global_seed(seed, device="cuda")` covers torch/numpy/random/CUDA/cuDNN; env capture (git commit, torch/cuda versions, deps hash). | | ☐ | `biopl-repro-check` (CLI) runs 1-epoch parity on all models, same seed → bitwise identical |

**Gate**: GPU integration tests < 30s total; benchmark harness produces comparable numbers across runs.

---

## Sprint 2: Biology Validation Expansion (Week 3–4)

*Beyond the 8 axioms: add gradient equivalence (finite-diff), energy landscape visualization, and negative-result documentation.*

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| **2.1** | **Finite-Difference Gradient Equivalence** (`tests/integration/test_gradient_equivalence.py`) — for every propagator: compute `grad_fd = (loss(w+ε) - loss(w-ε)) / 2ε`; assert `cosine(grad_fd, grad_local) ≥ threshold` per family (EqProp 0.7, FA 0.5, MEP 0.6, EquiTile 0.6, FF/PEPITA N/A). | | ☐ | CI gate: all registered propagators pass; thresholds documented in registry metadata |
| **2.2** | **Energy Landscape Visualization** (`analysis/energy_landscape.py`) — 2D slices of `E(w)` around trained weights; contour plots + gradient flow arrows. Integrate with `visualization.py`. | | ☐ | Generates `energy_landscape_{model}_{task}.png` for EqProp/EquiTile |
| **2.3** | **Contraction Mapping Verification** — extend `test_biology_axioms.py`: verify `||Δx_{t+1}|| / ||Δx_t|| < 1` for EquiTile/EP settling dynamics across β, depth, spectral norm. | | ☐ | Property test with hypothesis strategies for config space |
| **2.4** | **Failure Manifesto** (`analysis/failure_manifesto.py`) — structured negative results: what was tried, search space, why it failed, partial successes, hypotheses. Auto-populated from KB failed trials. | | ☐ | `biopl-failure-manifesto --model eqprop_mlp` → markdown report |
| **2.5** | **Biology Metadata Calibration** — extend registry `ComponentMetadata`: `bio_plausibility_score` (0-1, calibrated), `locality_level` (GLOBAL/LAYERWISE/LOCAL/EQUILIBRIUM/FORWARD_ONLY), `memory_complexity`, `requires_backward`, `credit_assignment_type`, `family` tag. Audit all 80+ components. | | ☐ | `biopl-registry-audit --metadata` → calibrated CSV; CI gate on completeness |

**Gate**: All biology property tests + gradient equivalence pass; failure manifesto generates for at least 3 model families.

---

## Sprint 3: Interactive Demo UI — NiceGUI (Weeks 4–6)

*Side-by-side comparison of any 2 configurations (incl. backprop): live charts, animated weight matrices, hyperparameter widgets. Trivial + real tasks.*

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| **3.1** | **NiceGUI Project Setup** (`demo/`) — `pyproject.toml` extra `demo = ["nicegui", "plotly", "torchvision", "datasets"]`; `demo/main.py` entry; Quasar dark theme; asyncio event bus from `execution/engine.py` plugs directly. | | ☐ | `uv run demo/main.py` → browser opens at `localhost:8080` |
| **3.2** | **Config-Driven Widget Generation** — `demo/widgets.py`: inspect Pydantic/dataclass config (e.g., `EquiTileConfig`, `ModelConfig`) → auto-generate sliders, dropdowns, number inputs with tooltips from docstrings. Two panels: **Config A** vs **Config B** (backprop baseline pre-filled). | | ☐ | Changing any widget updates live preview instantly |
| **3.3** | **Task Selector** — tabs: **Toy** (XOR, spiral, concentric circles), **Digits** (sklearn), **MNIST**, **CIFAR-10**, **Tiny Shakespeare**. Each loads synthetic or real data via `tests/conftest.py` fixtures (GPU-accelerated). | | ☐ | All 5 tasks load < 2s; MNIST/CIFAR stream from torchvision cache |
| **3.4** | **Live Training Charts** (`demo/charts.py`) — Plotly `FigureWidget` streaming: loss/accuracy (dual Y), Lipschitz constant, gradient alignment, tile activity heatmap (EquiTile), energy trajectory (EP). WebSocket push from engine callback. | | ☐ | 100-step training animates smoothly at 10 FPS; no UI freeze |
| **3.5** | **Animated Weight Matrices** (`demo/weight_viz.py`) — canvas/Vue component: color-coded `W_t` per layer/tile; play/pause/scrub slider; hover shows value + gradient magnitude; side-by-side diff view (Config A - Config B). | | ☐ | 64×64 matrix @ 30 FPS; diff view highlights divergent weights |
| **3.6** | **Experiment Persistence** — "Save Config" / "Load Config" (JSON); "Export Run" (CSV + charts PNG + weight MP4); shareable URL with encoded config. | | ☐ | Exported config reloads identically; MP4 playable |
| **3.7** | **Backprop Baseline Parity** — pre-built `backprop_mlp`, `backprop_cnn`, `backprop_transformer` configs; one-click "Run Parity" trains both configs, overlays curves, prints final gap %. | | ☐ | Parity gap matches CLI `biopl-parity` within 1% |

**Gate**: Demo runs end-to-end on RTX 3080; researcher can reproduce parity claim in < 5 min.

---

## Sprint 4: Ecosystem Positioning & Recruitment (Week 6–7)

*Articulate Bioplausible's unique value in modern ML; produce recruitment artifacts.*

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| **4.1** | **Positioning Doc** (`docs/positioning.md`) — where Bioplausible fits: (a) **Local learning research** — only framework with EqProp/FA/MEP/EquiTile/FF/Spiking unified; (b) **Neuromorphic bridge** — same code runs GPU + Loihi/SpiNNaker via deployment; (c) **AutoScientist substrate** — registry + KB + campaign = autonomous hypothesis engine; (d) **Memory-efficient training** — O(1) memory claim verified on 1000-layer EquiTile. | | ☐ | Doc reviewed by 2 external researchers; feedback incorporated |
| **4.2** | **5-Minute Colab Notebook** (`examples/colab/bioplausible_demo.ipynb`) — `pip install bioplausible[demo]` → runs EquiTile on CIFAR-10 in browser; links to live demo UI. | | ☐ | Executes in Colab free tier (T4) < 5 min; no auth needed |
| **4.3** | **Leaderboard Automation** (`leaderboard/generator.py` + GitHub Action) — nightly parity benchmarks → markdown table in README; Pareto frontier plots as artifacts. | | ☐ | `README.md` updates automatically; plots viewable in Actions |
| **4.4** | **Good First Issues** — tag 10 issues: test gaps, docstrings, benchmark configs, demo widgets, registry metadata. `CONTRIBUTING.md` with component registration walkthrough. | | ☐ | Issues labeled `good first issue`; PR template enforces registry metadata |

**Gate**: Colab notebook runs green; leaderboard updates nightly; 2+ external PRs merged.

---

## Sprint 5: RESEARCH.pre.md Tier 2–3 (CI Correctness + Types) (Week 7–8)

*Finish Tier 2 (CI gates) and Tier 3 (type system) from RESEARCH.pre.md — now unblocked by Sprint 0.*

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| **5.1** | **`print()` → `logging`** — 4 benchmark files (52+38+26+4 prints) → module-level logger + lazy `%s` interpolation. | | ☐ | `grep -r "print(" bioplausible/ --include="*.py" | grep -v "__main__" | wc -l` = 0 |
| **5.2** | **Narrow `except Exception`** — 5 KB sites + 2 EquiTile scheduler sites → specific exceptions + `logger.exception` + chained domain errors (uses Sprint 0.1 hierarchy). | | ☐ | No bare `except Exception` in lib code; tracebacks preserved |
| **5.3** | **Bare-Except Parens** — 17 sites across 12 files → `except (X, Y):` (mechanical, one pass). | | ☐ | `ruff check --select E722` = 0 |
| **5.4** | **Eliminate `Any`** — 6 sites (trainer, config, equitile/config) → `object` or `Protocol`; `Literal` for `credit_assignment_type`; frozen dataclass audit (3 stragglers). | | ☐ | `pyright --strict` 0 errors (warnings may remain) |
| **5.5** | **CI Pipeline Config** (`.github/workflows/ci.yml`) — `ruff format --check` → `ruff check` → `pyright` → `pytest --cov --maxfail=5` (unit+property+biology); coverage floor 50% → 85% over time. | | ☐ | CI green on main; badge in README |

**Gate**: Full CI pipeline passes; `pyright` 0 errors; coverage ≥ 50%.

---

## Sprint 6: AutoScientist v1 Foundations (Week 8–10)

*Minimal viable autonomous discovery: campaign persistence + structured reasoning + KB synthesis.*

| # | Task | Owner | Status | Validation |
|---|------|-------|--------|------------|
| **6.1** | **Campaign Persistence** (`autoscientist/campaign_v1.py`) — YAML + SQLite state; resume from arbitrary checkpoint; git-like branches for exploration. | | ☐ | `biopl-scientist resume campaign.yaml --from trial_42` works |
| **6.2** | **Chain-of-Thought Templates** (`autoscientist/reasoner.py`) — failure analysis, transfer reasoning, composition reasoning, scaling prediction; structured JSON output matching `Hypothesis` dataclass. | | ☐ | LLM generates valid hypothesis JSON for 5/5 test prompts |
| **6.3** | **KB Meta-Analysis** (`knowledge/kb.py:run_meta_analysis()`) — scaling law fits (power law), algorithm fingerprinting (PCA on hyperparam sensitivity), failure manifold, cross-domain transfer matrix. | | ☐ | `kb.run_meta_analysis()` → report with fitted α,β,γ + confidence intervals |
| **6.4** | **Surrogate-Guided Proposal** — `kb.suggest_next_experiment()` uses GPyTorch/BoTorch (optional dep) over algorithm space; falls back to random if unavailable. | | ☐ | Generates non-trivial config suggestions; logs to KB |

**Gate**: AutoScientist runs overnight → 50 tested hypotheses in KB; meta-analysis report readable.

---

## Deferred / Not In This Plan

| Item | Reason |
|------|--------|
| Ruff style violations (2472 remaining) | Cosmetic; re-scope config or fix opportunistically during real work |
| Full neuromorphic deployment (Loihi, SpiNNaker, BrainScaleS) | Trigger: GPU parity published + hardware partner interest |
| Optical/analog/memristor simulation | Post-GPU-validation; collaboration-dependent |
| Phase 2–10 of RESEARCH.md | Long-term agenda; this plan covers Phase 0 + Demo + Recruitment |
| CLI unification (`bioplausible` single entry) | NiceGUI demo replaces CLI for researchers; CLI for automation only |
| Colab notebooks per domain | One flagship notebook sufficient for recruitment |

---

## Success Metrics (End of Sprint 6)

| Metric | Target |
|--------|--------|
| **Demo viability** | Researcher reproduces EqProp/EquiTile parity on CIFAR-10 in < 5 min via NiceGUI |
| **Test suite** | Unit+property+biology < 60s CPU; GPU integration < 30s; 0 flakes in 5 runs |
| **Biology proof** | 8 axioms + gradient equivalence + energy landscapes + failure manifesto for 3+ families |
| **Registry** | 100% components instantiated, metadata calibrated, audit CI gate green |
| **AutoScientist** | 50 hypotheses/week; meta-analysis extracts scaling laws from KB |
| **Recruitment** | Colab runs green; leaderboard updates nightly; 2+ external PRs |
| **Type safety** | `pyright` 0 errors (strict); `ruff` 0 correctness violations (style ignored) |

---

## Architecture Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-01 | NiceGUI for demo UI | Asyncio-native; headless engine event bus plugs directly; Python-only authoring; Quasar theme; canvas escape hatch for weight matrices |
| 2026-08-01 | Selective GPU testing | Unit/property tests stay CPU (deterministic, fast); integration/benchmarks use GPU for 5-10x speedup on large models |
| 2026-08-01 | Fold RESEARCH.pre.md Tier 1 into Sprint 0 | High-leverage architecture unblocks AutoScientist, registry, KB, scaling sweeps; defer Tier 2-3 to Sprint 5 |
| 2026-08-01 | Defer all lint style work | 2472 violations are ~100% style (N803, PLR09xx, TRY002, E402); config re-scope or opportunistic fixes only |

---

## Quick Reference: Commands

```bash
# Fast gate (CPU only)
uv run pytest tests/unit/ tests/property/ -q --no-cov

# GPU integration gate
uv run pytest tests/integration/ -m gpu -q --no-cov

# Biology property tests
uv run pytest tests/property/biology/ -v --no-cov

# Benchmark harness
uv run pytest tests/unit/validation/benchmark_harness.py -m benchmark -v --no-cov

# Demo UI
uv run demo/main.py

# Registry audit + metadata
uv run biopl-registry-audit --metadata

# Gradient equivalence
uv run pytest tests/integration/test_gradient_equivalence.py -v --no-cov

# AutoScientist overnight
uv run biopl-scientist --campaign config/campaign.yaml --max-trials 50

# Full CI simulation
uv run ruff format --check . && uv run ruff check . && uv run pyright . && uv run pytest --cov --maxfail=5
```

---

## File/Module Map for New Work

```
bioplausible/
├── core/
│   ├── exceptions.py          # NEW Sprint 0.1
│   ├── registry.py            # REFACTOR Sprint 0.2 (_QueryFilter predicates)
│   ├── model.py               # REFACTOR Sprint 0.3, 0.4
│   └── trainer.py             # REFACTOR Sprint 0.3
├── execution/
│   ├── engine.py              # REFACTOR Sprint 0.3, 0.4
│   ├── _state.py              # REFACTOR Sprint 0.6 (SQLite context manager)
│   └── dashboard.py           # INTEGRATES with NiceGUI event bus
├── equitile/
│   ├── core/model.py          # REFACTOR Sprint 0.3, 0.4
│   └── utils/ → _utils/       # Sprint 0.5 (module boundary)
├── knowledge/
│   └── kb.py                  # ENHANCE Sprint 2.4, 6.3 (meta-analysis)
├── analysis/
│   ├── energy_landscape.py    # NEW Sprint 2.2
│   ├── failure_manifesto.py   # NEW Sprint 2.4
│   └── scaling.py             # NEW Sprint 6.3
├── autoscientist/
│   ├── campaign_v1.py         # NEW Sprint 6.1
│   ├── reasoner.py            # ENHANCE Sprint 6.2 (CoT templates)
│   └── proposer.py            # ENHANCE Sprint 6.4 (surrogate-guided)
├── deployment.py              # EXISTING (ONNX/FastAPI)
└── visualization.py           # EXISTING (matplotlib → Plotly for demo)

demo/                          # NEW Sprint 3
├── main.py                    # NiceGUI entry
├── widgets.py                 # Config-driven auto-widgets
├── charts.py                  # Plotly FigureWidget streaming
├── weight_viz.py              # Canvas/Vue weight matrix animation
├── tasks.py                   # Toy/Digits/MNIST/CIFAR/LM loaders
└── demo_config.py             # Pre-built backprop baselines

tests/
├── conftest.py                # ENHANCE Sprint 1.1 (GPU fixtures)
├── integration/
│   ├── test_gradient_equivalence.py  # NEW Sprint 2.1
│   └── ... (migrated to @pytest.mark.gpu)
└── unit/validation/
    └── benchmark_harness.py   # NEW Sprint 1.3

.github/workflows/ci.yml       # NEW Sprint 5.5
docs/positioning.md            # NEW Sprint 4.1
examples/colab/bioplausible_demo.ipynb  # NEW Sprint 4.2
```

---

*This plan replaces the previous TODO.md. RESEARCH.md remains the long-term research agenda. RESEARCH.pre.md is now fully absorbed — its Tier 1 items are Sprint 0, Tier 2-3 are Sprint 5, Appendix items are referenced in relevant sprints.*
