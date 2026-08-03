# VALIDATE2.md — Continuous Validation Pipeline (Phases 2–4+)

**Continuation of**: `VALIDATE.md` (Phase 0 + Phase 1 complete)
**Prerequisite**: Phase 1 gate passed — portfolio ranked, survivors identified, measured memory track working.

---

## Design Philosophy: Continuous Open-Ended Improvement

This is a **continuous validation loop** that runs perpetually, produces improving artifacts, and auto-escalates shareability.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS VALIDATION LOOP                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────┐  │
│   │ PORTFOLIO│───▶│ SCALING     │───▶│ REGIME DEMO  │───▶│ REAL- │  │
│   │ REVELATION│    │ VALIDATION  │    │ (structural) │    │IZATION│  │
│   └──────────┘    └─────────────┘    └──────────────┘    └───────┘  │
│        │               │                   │                  │       │
│        ▼               ▼                   ▼                  ▼       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              KNOWLEDGE BASE (continuous)                    │   │
│   │  • Scaling law fits  • Pareto frontiers  • Failure manifold │   │
│   │  • Algorithm fingerprints  • Cross-domain transfer matrix  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              AUTOSCIENTIST (hypothesis → experiment)        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                           │                                         │
│                           └──────────────────────┐                  │
│                                                  ▼                  │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              SHAREABILITY ESCALATION                        │   │
│   │  Level 1: Internal → Level 2: Preprint → Level 3: Pub      │   │
│   │  Level 4: Industry (structural demo)                        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Scaling Validation (Continuous)

**Objective**: Prove surviving algorithms scale. Runs continuously as new survivors emerge.

### 2.1 Scaling Law Characterization (Automated)

**Existing Infrastructure**: 
- `bioplausible/analysis/scaling.py` — `fit_power_law`, `plot_scaling_curves`, `compute_compute_optimal`
- `validation/tracks/scaling_tracks.py` — Tracks 10 (now measured), 11, 12
- `hyperopt/comparison.py` — `AlgorithmRanking`, `compute_algorithm_rankings`

**Scales to Test** (configurable, extensible):

| Scale | Task | Model Config | Compute Budget |
|-------|------|--------------|----------------|
| S | Digits | MLP 500K | 1 GPU-hr |
| M | CIFAR-10 | CNN 2M | 4 GPU-hr |
| L | CIFAR-10 / TinyImageNet | CNN/Transformer 10M | 16 GPU-hr |
| XL | ImageNet subset | 50M | 64 GPU-hr |

**New Orchestrator**: `bioplausible/validation/continuous_scaling.py`

```python
# Runs when new survivor detected or weekly
async def continuous_scaling_loop():
    while True:
        survivors = kb.get_survivors(status="Scale")
        for algo in survivors:
            if not kb.has_recent_scaling(algo, max_age_days=7):
                await run_scaling_sweep(algo)
                kb.store_scaling_results(algo, results)
                check_scaling_gate(algo)
        await asyncio.sleep(3600)
```

**`run_scaling_sweep(algo)` Implementation**:
```python
async def run_scaling_sweep(algo: str):
    """Run HPO at S, M, L scales for one algorithm."""
    for scale_name, scale_config in SCALES.items():
        study_name = f"{algo}_{scale_name}_{scale_config.task}"
        # Reuse biopl-hpo search logic programmatically
        await run_hpo_study(
            study_name=study_name,
            model=algo,
            task=scale_config.task,
            model_kwargs=scale_config.model_kwargs,
            trials=scale_config.trials,
            seeds=5,
        )
    
    # Collect results, compute scaling laws
    results = collect_scaling_results(algo)
    
    # Power-law fit on gap vs scale
    gaps = [r.parity_gap_pp for r in results.scales]
    scales = [0, 1, 2, 3]  # S, M, L, XL
    coeffs = np.polyfit(scales, gaps, 1)
    trend_pp_per_decade = coeffs[0] * 10  # slope per log10 scale
    
    return ScalingResults(
        algo=algo,
        scales=results,
        gap_trend_pp_per_decade=trend_pp_per_decade,
    )
```

**Metrics per Scale** (extend `TrackResult`):
- Final accuracy (val)
- Training wall-time (s)
- **Peak memory (MB) — measured** (Track 10 now does this)
- FLOPs/epoch (via `torch.profiler.profile`)
- Settling cost (equilibrium): steps × forward pass time
- Energy estimate: GPU TDP × utilization × wall-time

### 2.2 Scaling Gate Check (Automated, Soft Gate)

```python
def check_scaling_gate(algo: str, results: ScalingResults) -> GateDecision:
    """Returns CONTINUE / PAUSE / PIVOT based on gap trajectory."""
    trend = results.gap_trend_pp_per_decade
    
    if trend > 2.0:      
        return GateDecision.PIVOT   # Gap growing >2pp per 10x scale → ceiling
    elif trend > -0.5:   
        return GateDecision.CONTINUE  # Gap stable or shrinking slowly
    else:                 
        return GateDecision.ACCELERATE  # Gap shrinking → invest more
```

**Effect on Algorithm Status** (stored in KB):
- `PIVOT` → status = "Hold", reduce compute allocation
- `CONTINUE` → status = "Scale", normal compute
- `ACCELERATE` → status = "Scale", increase compute allocation

### 2.3 Compute-Matched Pareto Frontiers (Continuous)

**Existing**: `analysis/scaling.compute_compute_optimal`, `hyperopt/comparison.AlgorithmRanking.pareto_count`

**New Continuous Artifact**: `results/pareto/<algo>_<task>_<timestamp>.json`

```json
{
  "algo": "eqprop_mlp",
  "task": "cifar10",
  "timestamp": "2026-08-03T14:22:00Z",
  "frontier": [
    {"accuracy": 0.82, "peak_mem_mb": 1200, "wall_time_s": 3400, "flops": 1.2e12, "scale": "M"},
    {"accuracy": 0.79, "peak_mem_mb": 600, "wall_time_s": 1800, "flops": 6e11, "scale": "S"},
    ...
  ],
  "backprop_frontier": [...],
  "pareto_dominance": {
    "memory": "eqprop dominates at all scales",
    "accuracy": "backprop dominates at all scales",
    "time": "backprop 1.5-2x faster"
  }
}
```

**Visualization**: `results/pareto/plots/<algo>_<task>.html` (Plotly) via `analysis.scaling.plot_scaling_curves`.

**CLI** (new `biopl-hpo pareto` from VALIDATE.md Stage A.5):
```bash
uv run biopl-hpo pareto --study eqprop_cifar10 --output-dir results/pareto
```

### 2.4 Memory Advantage Demonstration (Continuous, Flagship)

**Now uses measured memory from fixed Track 10.**

**Protocol** (extends `track_10_memory_scaling`):
```python
def run_memory_demo(algo: str, depths: list[int] = [10, 25, 50, 100]):
    """Measure actual peak memory at depth for bio vs backprop."""
    results = {}
    for depth in depths:
        bio_model = create_model(algo, depth=depth).to(device)
        bp_model = create_backprop_baseline(depth=depth).to(device)
        
        bio_mem = measure_peak_memory(bio_model, test_loader, device)
        bp_mem = measure_peak_memory(bp_model, test_loader, device)
        
        results[depth] = {
            "bio_mb": bio_mem,
            "backprop_mb": bp_mem,
            "ratio": bp_mem / bio_mem,
            "bio_acc": evaluate(bio_model),
            "bp_acc": evaluate(bp_model),
        }
    return results
```

**Success Criterion** (continuous check after each sweep):
- At depth ≥ 50: `bio_peak_mem / bp_peak_mem < 0.2` (5× advantage)
- Accuracy within 10 pp of backprop at same depth

**Output**: `results/memory_demo/<algo>_depth<N>_<timestamp>.json`

### 2.5 Continual Learning Demonstration (If Applicable)

**Existing**: `validation/tracks/application_tracks.py::track_21_continual_learning` (EWC on EqProp)

**Extend to**: All survivors with `locality_level` in {LOCAL, EQUILIBRIUM, FORWARD_ONLY}

**Protocol** (from original plan):
- Split CIFAR-10: Task A (classes 0-4) → Task B (5-9) sequential
- Measure: Task A accuracy after Task B (forgetting)
- Compare: Bio-algo (no replay) vs Backprop (no replay) vs Backprop + replay buffer
- **Success**: Bio-algo retains >80% Task A accuracy, Backprop no-replay <30%

**New Runner**: `bioplausible/validation/continual_demo.py` — runs on every new survivor with local updates.

---

## Phase 3: Regime-Specific Demonstration (Continuous)

**Objective**: Produce **one concrete capability backprop fundamentally cannot provide**.

### 3.1 Regime Selection (Automated from Phase 2 Results)

```python
def select_regime(scaling_results: dict) -> RegimeDemo:
    advantages = {}
    
    for algo, results in scaling_results.items():
        # Memory advantage
        mem_ratio = results.memory_ratio_at_depth(100)
        if mem_ratio > 5:
            advantages[f"{algo}_memory"] = ("memory_constrained", mem_ratio)
        
        # Continual learning
        if results.has_continual_demo:
            retention = results.task_a_retention_after_b
            if retention > 0.8:
                advantages[f"{algo}_continual"] = ("on_device_adaptation", retention)
        
        # Asynchronous (EquiTile async, LazyEqProp)
        if results.supports_async:
            speedup = results.async_speedup
            if speedup > 1.5:
                advantages[f"{algo}_async"] = ("async_distributed", speedup)
        
        # Analog/noise tolerance
        if results.noise_tolerance > 0.8:
            advantages[f"{algo}_analog"] = ("analog_hardware", results.noise_tolerance)
    
    return max(advantages.items(), key=lambda x: x[1][1])  # highest score
```

### 3.2 Demonstration Template (Runnable Script)

Each regime gets a **single-file runnable demo** with three criteria:

| Criterion | Implementation |
|-----------|----------------|
| **A. Backprop cannot do it** | Structural limitation, not performance gap. E.g., "Train 200-layer net in 256MB RAM" — backprop physically impossible. |
| **B. Concrete & measurable** | Specific numbers: "Trains to 78% on CIFAR-10 in 180MB peak RAM. Backprop needs 1.2GB (gradient checkpointing) or 3.4GB (standard)." |
| **C. Reproducible on commodity HW** | Single command, CPU or single GPU. No exotic hardware. |

**Template**: `bioplausible/demos/regime_<name>.py`

```python
#!/usr/bin/env python
"""
Regime Demo: Memory-Constrained Deep Learning
Run: uv run python -m bioplausible.demos.regime_memory_constrained
"""
import argparse
from bioplausible.validation.regime_demos import run_memory_demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--memory-limit-mb", type=int, default=256)
    parser.add_argument("--compare-backprop", action="store_true")
    parser.add_argument("--output", type=str, help="JSON output path")
    args = parser.parse_args()
    
    results = run_memory_demo(
        depth=args.depth,
        memory_limit_mb=args.memory_limit_mb,
        compare_backprop=args.compare_backprop,
    )
    
    print(f"Bio-algo: {results.bio.accuracy:.1%} acc, {results.bio.peak_mem_mb:.0f}MB peak")
    if results.backprop:
        print(f"Backprop: {results.backprop.accuracy:.1%} acc, {results.backprop.peak_mem_mb:.0f}MB peak")
        print(f"Advantage: {results.backprop.peak_mem_mb / results.bio.peak_mem_mb:.1f}× less memory")
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
    
    exit(0 if results.success else 1)
```

### 3.3 Backprop + All Optimizations Comparison (Required for Fairness)

**New Module**: `bioplausible/validation/backprop_optimized.py`

Runs backprop with every known memory optimization under the same constraint:
- Gradient checkpointing (`torch.utils.checkpoint`)
- Mixed precision (FP16/BF16 via `torch.autocast`)
- Activation offloading (CPU offload)
- Model parallelism (FSDP/ZeRO if multi-GPU)

Finds best achievable backprop accuracy under the memory limit.

### 3.4 Shareability Artifacts (Auto-generated)

| Artifact | Location | Purpose |
|----------|----------|---------|
| Runnable script | `bioplausible/demos/regime_<name>.py` | Single-command reproduction |
| Comparison table | `results/regime_demos/<name>_comparison.csv` | Bio vs BP vs BP+optimizations |
| Hardware projection | `results/regime_demos/<name>_hardware_projection.md` | **Labeled projections** for Loihi, analog, etc. |
| Video walkthrough | `results/regime_demos/<name>_walkthrough.mp4` | 5-min demo for presentations (manual) |

---

## Phase 4: Realization Path (Continuous, Simulation-First)

**Contingent on Phase 3 demo passing Level 4 gate.**

### 4.1 Hardware Target Selection (Automated)

| Regime Demo | Primary Target | Secondary Target |
|-------------|----------------|------------------|
| Memory-constrained | Loihi 2 / SpiNNaker | Edge TPU (Coral) |
| On-device adaptation | Jetson Orin / Coral | Snapdragon NPU |
| Analog/noisy | Memristor crossbar sim | Analog AI simulator |
| Async distributed | Multi-chip neuromorphic | Multi-GPU (DDP) |

### 4.2 Simulation-First Validation (Required Before Hardware)

**Infrastructure**: `bioplausible/acceleration/kernels.py` (NumPy ref), `surrogate.py` (spiking)

**Constraints to Simulate** (configurable matrix):

```python
CONSTRAINT_MATRIX = {
    "quantization": [8, 4, 2],           # bits
    "weight_noise": [0.0, 0.01, 0.05, 0.1],  # Gaussian σ
    "activation_noise": [0.0, 0.01, 0.05],
    "asynchrony": [False, True],         # random update order
    "sparsity": [0.0, 0.5, 0.75, 0.9],   # weight pruning
    "device_variation": [0.0, 0.05, 0.1], # per-device param variation
}
```

**Validation Protocol** (continuous for each survivor):

```python
def simulate_hardware_constraints(algo: str, base_config: dict) -> HardwareViability:
    results = []
    for constraints in product(*CONSTRAINT_MATRIX.values()):
        config = apply_constraints(base_config, constraints)
        acc = run_training(config, epochs=10, seeds=3)
        retention = acc / base_accuracy
        results.append((constraints, retention))
    
    # Gate: Must retain >80% accuracy under AT LEAST ONE constraint combo
    viable = any(r > 0.8 for _, r in results)
    best_combo = max(results, key=lambda x: x[1])
    
    return HardwareViability(
        viable=viable,
        best_retention=best_combo[1],
        best_constraints=best_combo[0],
        all_results=results,
    )
```

### 4.3 Deployment Prototype (After Simulation Gate)

**Deliverables**:
- Trained model running inference on target (or closest sim)
- **Learning rule executing on-device** (not just inference — actual weight updates)
- Measured: inference latency, update latency, energy/update, memory footprint

**NiceGUI Integration**: Live deployment status panel (manual addition to `demo/`).

---

## Continuous Infrastructure (Shared Across Phases)

### C1. Knowledge Base Integration (Extend Existing)

**Existing**: `bioplausible/knowledge/kb.py` — extend with continuous methods:

```python
def run_meta_analysis(self) -> MetaAnalysisReport:
    return MetaAnalysisReport(
        scaling_laws=self.extract_all_scaling_laws(),
        algorithm_fingerprints=self.compute_hyperparam_sensitivity_pca(),
        failure_manifold=self.map_failures_by_model_task(),
        transfer_matrix=self.compute_cross_domain_transfer(),
    )

def suggest_next_experiment(self) -> ExperimentProposal:
    """Bayesian optimization over algorithm space."""
    # Uses Pareto frontiers, scaling law uncertainty, failure density
    pass
```

### C2. Automated Shareability Escalation

**New**: `bioplausible/validation/shareability_watcher.py`

```python
async def watch_and_escalate():
    while True:
        for result in kb.get_new_results():
            level = assess_shareability_level(result)
            if level > result.current_level:
                publish_to_level(result, level)
                result.current_level = level
        await asyncio.sleep(300)
```

**Level Criteria** (automated assessment):

| Level | Name | Auto-Criteria |
|-------|------|---------------|
| 1 | Internal | Portfolio ranked, reproducibility confirmed |
| 2 | Preprint | 2 families <5pp gap digits (n≥10), 1 <8pp CIFAR (n≥5), effect sizes, negatives documented |
| 3 | Publication | Scaling gap stable/shrinking, memory advantage measured ≥50 layers, Pareto distinct, regime advantage demonstrated |
| 4 | Industry | Structural limitation exploited, single-command reproduction, BP+optimizations compared, independent reproduction |

### C3. Continuous CI/CD Integration

**Extend `.github/workflows/continuous-validation.yml`**:

```yaml
on:
  schedule: [cron: "0 */6 * * *"]
  workflow_dispatch:

jobs:
  continuous-validation:
    runs-on: gpu-runner
    timeout-minutes: 360
    steps:
      - uses: actions/checkout@v4
      - name: Run continuous validation loop
        run: |
          uv run biopl-validate --continuous --max-hours=5
      - name: Update knowledge base
        run: uv run biopl-kb-sync
      - name: Check shareability gates
        run: uv run biopl-shareability-check
      - name: Publish artifacts
        if: success()
        run: uv run biopl-publish-artifacts
```

### C4. New CLI Commands Needed

| Command | Purpose | Implementation |
|---------|---------|----------------|
| `biopl-scaling sweep` | Run multi-scale HPO for one algo | `validation/continuous_scaling.py` |
| `biopl-validate` | Orchestrator for continuous loop | `validation/validate_orchestrator.py` |
| `biopl-kb-sync` | Sync results to KB, run meta-analysis | `knowledge/kb.py` + CLI |
| `biopl-shareability-check` | Assess and escalate shareability levels | `validation/shareability_watcher.py` |

---

## Decision Gates (Continuous, Soft)

| Check | Frequency | Action if Failed |
|-------|-----------|------------------|
| Scaling gap trend > +2pp/10x | After each sweep | Algorithm → **Hold**, reduce compute |
| Memory advantage < 5× at depth 50 | After each sweep | Algorithm → **Hold**, investigate |
| No regime advantage after 3 sweeps | Monthly | Algorithm → **Eliminate**, document in manifesto |
| Shareability Level 3 achieved | Continuous | Auto-prepare preprint draft |
| Shareability Level 4 achieved | Continuous | Auto-prepare industry demo package |

---

## Success Metrics (Tracked Continuously in KB)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Best parity gap (CIFAR-10) | < 5 pp | HPO best trial (n≥5 seeds) |
| Memory advantage (depth 100) | > 5× | Measured peak RAM (Track 10) |
| Continual retention (no replay) | > 80% | Track 21 / continual_demo |
| Scaling law exponent (gap vs scale) | ≤ 0 | Power-law fit on sweep data |
| Algorithms at Level 3+ | ≥ 2 | Shareability watcher |
| Independent reproductions | ≥ 1 per Level 4 | Community tracker |

---

## Implementation Checklist (What to Actually Build)

| Task | File(s) | Effort | Depends On |
|------|---------|--------|------------|
| `continuous_scaling.py` orchestrator | `validation/continuous_scaling.py` | 4 hrs | Stage A (CLI) |
| `regime_demos/memory_constrained.py` | `demos/regime_memory_constrained.py` | 3 hrs | Phase 2 memory demo |
| `backprop_optimized.py` | `validation/backprop_optimized.py` | 3 hrs | — |
| `hardware_sim.py` | `validation/hardware_sim.py` | 4 hrs | Phase 3 demo |
| `shareability_watcher.py` | `validation/shareability_watcher.py` | 2 hrs | KB methods |
| `validate_orchestrator.py` + `biopl-validate` CLI | `validation/validate_orchestrator.py`, `cli/validate.py` | 3 hrs | All above |
| `biopl-scaling` CLI | `cli/scaling.py` | 1 hr | `continuous_scaling.py` |
| `biopl-kb-sync` CLI | `cli/kb_sync.py` | 1 hr | `knowledge/kb.py` |
| **Total** | | **~21 hrs** | Stage A complete |

---

## Summary: Continuous Path to Industry-Relevant Result

```
Week 1:     Stage A (wire CLI) + Stage B (fix Track 10 measured memory)
Week 2:     Phase 0/1 HPO sweeps on digits → portfolio ranking → survivors
Week 3:     Phase 2 continuous scaling sweeps on survivors (S, M, L)
Week 4:     Memory advantage demo at depth 50, 100 (measured)
Week 5:     Regime demo (memory-constrained) → Level 4 artifact
Week 6+:    Hardware simulation → deployment prototype
Continuous: KB meta-analysis → AutoScientist hypotheses → new algorithms → loop
```

**Key Insight**: The memory advantage demo (Phase 2.3 / Phase 3) is the **highest-leverage single result**. It requires no hardware, no accuracy parity, just: *train 100-layer net with BP (measure RAM) vs EqProp (measure RAM)*. If ratio > 5×, result is undeniable and industry-relevant immediately.

All infrastructure exists except the orchestration layer. The work is **wiring, measurement, and continuous execution** — not new algorithm implementation.