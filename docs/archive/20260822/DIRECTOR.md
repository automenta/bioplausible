# The Director — Autonomous Research Agent Specification

> **A fiduciary for research compute. Converts compute into verified, compounding knowledge with zero disappointment.**

---

## 1. Purpose

The Director is a **continuous autonomous research process** that owns the entire lifecycle: hypothesis → execution → verification → learning → resource request. It operates under a written Constitution that prevents overpromise, unverified publication, wasted failure, and credit inflation.

**It is not a scheduler. It is a fiduciary.**

---

## 2. The Constitution (Invariants)

The Director **cannot** violate these. Violation = immediate halt + human intervention.

| Article | Principle | Enforcement |
|---------|-----------|-------------|
| **I. Credit Conservation** | `compute_spent ≤ credit_earned + endowment` | Pre-execution gate: `estimated_cost ≤ available_credit` |
| **II. Verification First** | No result enters KB without passing ALL gates | Gates: gradient_equiv, reproducibility, parity, registry_audit |
| **III. Minimal Disappointment** | `promise ≤ delivery` | `proposal.budget ≤ available_credit` |
| **IV. Failure as Asset** | Every failure yields structured FailureManifesto entry | Required fields: root_cause, search_space, partial_successes, next_implication |
| **V. External Validity** | Every result independently reproducible | Seed, git_commit, env_hash, config_hash recorded |
| **VI. Compounding Knowledge** | Each cycle builds on prior verified results | Proposal must cite KB lineage |
| **VII. Transparent Accounting** | Ledger is public, append-only, cryptographically auditable | Merkle tree over verified results + credit deltas |

---

## 3. Core Data Structures

### 3.1 VerifiedResult (Atomic Unit of Credit)

```python
@dataclass(frozen=True, slots=True)
class VerifiedResult:
    experiment_id: str
    hypothesis: str
    config_hash: str
    seed: int
    git_commit: str
    environment_hash: str

    # ALL must be True
    gradient_equivalence: bool
    reproducibility: bool
    parity_benchmark: bool
    registry_audit: bool

    metrics: dict[str, float]  # accuracy, flops, memory, wall_time, energy
    artifacts: list[str]  # plots, data, models
    failure_manifesto_id: str | None

    novelty_score: float  # 0-1: fills KB gaps
    rigor_score: float  # 0-1: gate pass quality
    reproducibility_score: float  # 1.0=bitwise, 0.5=statistical, 0=failed

    credit_earned: float  # = novelty × rigor × reproducibility × base_value
    timestamp: datetime
```

### 3.2 CreditLedger (Append-Only, Immutable)

```python
@dataclass(frozen=True, slots=True)
class CreditLedger:
    entries: tuple[VerifiedResult, ...] = ()
    initial_endowment: float = 10.0  # Infrastructure completion credit

    @property
    def total_earned(self) -> float:
        return sum(r.credit_earned for r in self.entries)

    @property
    def available_credit(self) -> float:
        days_idle = (now() - self.entries[-1].timestamp).days if self.entries else 0
        decay = 0.5 ** (days_idle / 30)  # Half-life: 30 days
        return (self.initial_endowment + self.total_earned) * decay

    def append(self, result: VerifiedResult) -> "CreditLedger":
        return CreditLedger(
            entries=self.entries + (result,), initial_endowment=self.initial_endowment
        )
```

### 3.3 Proposal (Investment Request)

```python
@dataclass(frozen=True, slots=True)
class Proposal:
    hypothesis: str
    experiment_file: str  # e.g., "experiments/tile_scaling.py"
    config: dict
    estimated_credit_cost: float
    estimated_duration_hours: float
    required_gpus: int
    kb_lineage: list[str]  # KB entry IDs this extends/contradicts
    gates_required: list[str]
    expected_deliverables: list[str]
    risk_assessment: dict[str, float]  # failure_mode → probability
```

### 3.4 DirectorState (Complete Serializable State)

```python
@dataclass(frozen=True, slots=True)
class DirectorState:
    ledger: CreditLedger
    knowledge_base_snapshot: str  # KB content hash
    failure_manifesto_snapshot: str  # Manifesto content hash
    current_proposal: Proposal | None = None
    cycle_count: int = 0
    last_cycle_timestamp: datetime | None = None
    status: Literal[
        "idle", "proposing", "executing", "verifying", "learning", "reporting", "halted"
    ] = "idle"
    halt_reason: str | None = None
```

---

## 4. Decision Algorithm (Pure Function: State → Action)

```python
@dataclass(frozen=True, slots=True)
class Action:
    type: Literal["propose", "execute", "verify", "learn", "report", "request", "halt"]
    payload: dict
    reasoning: str  # Human-readable
    constitutional_basis: list[str]  # Which articles apply


def decide(state: DirectorState) -> Action:
    """Pure, deterministic, auditable. No side effects. No randomness."""

    # HALT: Constitution violations
    if state.status == "halted":
        return halt("already_halted")

    if state.ledger.available_credit < MIN_CREDIT_FOR_ANY_EXPERIMENT:
        return halt("insufficient_credit", "Credit exhausted", ["I"])

    # STATE MACHINE
    match state.status:
        case "idle":
            return _propose(state)
        case "proposing":
            return _execute(state)
        case "executing":
            return _verify(state)
        case "verifying":
            return _learn(state)
        case "learning":
            return _report(state)
        case "reporting":
            return _request(state)
        case _:
            return halt("invalid_state", "Unknown state", ["all"])


def _propose(state) -> Action:
    candidates = kb_query(
        max_cost=state.ledger.available_credit, exclude_tried=True, require_lineage=True
    )
    if not candidates:
        return halt("no_candidates", "No viable experiments within credit", ["I", "VI"])

    # Maximize: (expected_info_gain × success_prob) / cost
    best = max(candidates, key=lambda c: c.expected_value / c.estimated_credit_cost)
    return propose(best, "Highest value/cost ratio", ["III", "VI"])


def _execute(state) -> Action:
    return execute(state.current_proposal, "Launching experiment", ["II"])


def _verify(state) -> Action:
    return verify(
        state.current_proposal.experiment_id, "Running all verification gates", ["II"]
    )


def _learn(state) -> Action:
    result = load_verified_result(state.current_proposal.experiment_id)
    if not all([
        result.gradient_equivalence,
        result.reproducibility,
        result.parity_benchmark,
        result.registry_audit,
    ]):
        return halt("gates_failed", "Verification failed", ["II"])
    if result.failure_manifesto_id:
        generate_failure_manifesto(result)  # Article IV
    return learn(result, f"Credit earned: {result.credit_earned:.2f}", ["IV", "VI"])


def _report(state) -> Action:
    return report(state.cycle_count, "Emitting cycle report", ["VII"])


def _request(state) -> Action:
    credit = state.ledger.available_credit
    if credit > RESOURCE_REQUEST_THRESHOLD:
        return request(
            credit,
            credit_to_gpus(credit),
            f"Credit {credit:.1f} → requesting GPUs",
            ["I"],
        )
    return idle("Insufficient credit for request", ["I"])
```

---

## 5. Credit Mechanics

### 5.1 Credit Calculation

```python
def calculate_credit(result: VerifiedResult, base_value: float) -> float:
    """Credit = base × novelty × rigor × reproducibility"""
    if not result.all_gates_passed:
        return 0.0  # Article II: no partial credit

    return (
        base_value
        * result.novelty_score
        * result.rigor_score
        * result.reproducibility_score
    )
```

### 5.2 Base Values (Per Experiment Type)

| Experiment | Base Credit | Rationale |
|------------|-------------|-----------|
| TileNet Scaling Sweep | 5.0 | High novelty, multiple algorithms, Pareto frontiers |
| EqProp Vision Parity | 4.0 | Multiple variants, statistical rigor |
| FA Depth Scaling | 4.0 | Extreme depth, novel regime |
| MEP Preset Tournament | 3.0 | Factorized ablation, Sobol indices |
| MoT Ablation | 3.0 | Routing comparison, novel architecture |
| Cross-Domain Transfer | 4.0 | Multi-domain, transfer efficiency |
| Tile Algorithm Comparison | 5.0 | Fair comparison isolating credit assignment |

### 5.3 Credit Decay

- **Half-life**: 30 days without new verified results
- **Formula**: `available = (endowment + earned) × 0.5^(days_idle/30)`
- **Purpose**: Prevents credit hoarding; forces continuous delivery

### 5.4 External Replication Bonus

- Each independent replication: `+20%` credit on original result
- Verified via submitted reproduction report with matching seed/config

---

## 6. Verification Gates (Article II)

All gates **must pass** for credit. No exceptions.

| Gate | Command | Purpose |
|------|---------|---------|
| **Gradient Equivalence** | `biopl-verify-gradients --experiment ID` | Finite-difference verification of local gradients vs true gradients |
| **Reproducibility** | `biopl-repro-check --experiment ID --seed N` | Bitwise (or statistical) reproduction from recorded seed |
| **Parity Benchmark** | `biopl-parity --experiment ID` | Compute-matched comparison vs backprop baseline |
| **Registry Audit** | `biopl-registry-audit` | All components have complete, accurate metadata |

**Gate Results → VerifiedResult fields:**
- `gradient_equivalence`: gate 1 passed
- `reproducibility`: gate 2 passed
- `parity_benchmark`: gate 3 passed
- `registry_audit`: gate 4 passed

---

## 7. Failure Manifesto (Article IV)

Every failed experiment **must** produce:

```python
@dataclass(frozen=True, slots=True)
class FailureManifestoEntry:
    experiment_id: str
    hypothesis: str
    root_cause_hypothesis: str  # Why it failed (testable)
    search_space_explored: dict  # Hyperparameters, seeds, architectures tried
    partial_successes: list[str]  # What DID work (e.g., "converges at depth 10")
    next_experiment_implication: str  # What to try next (e.g., "adaptive β needed")
    kb_entries_created: list[str]  # Negative knowledge entered into KB
```

**Credit for failure**: `base_value × 0.3 × novelty` (if manifesto complete)
**Credit penalty**: `-base_value × 0.5` (if manifesto incomplete — wasted compute)

---

## 8. Knowledge Base Integration (Article VI)

### 8.1 KB Query for Proposals

```python
def kb_query(
    max_cost: float, exclude_tried: bool, require_lineage: bool
) -> list[Proposal]:
    """Returns candidate proposals ranked by expected_value / cost."""
    # 1. Identify KB gaps (missing scaling laws, untested algorithm/task combos)
    # 2. Generate proposals to fill gaps
    # 3. Filter by max_cost (Article III)
    # 4. Require KB lineage (Article VI)
    # 5. Rank by (info_gain × success_prob) / cost
```

### 8.2 KB Absorption (After Verification)

```python
def absorb_result(kb: KnowledgeBase, result: VerifiedResult):
    kb.add_entry(
        experiment_id=result.experiment_id,
        hypothesis=result.hypothesis,
        metrics=result.metrics,
        config_hash=result.config_hash,
        seed=result.seed,
        git_commit=result.git_commit,
        artifacts=result.artifacts,
        credit_earned=result.credit_earned,
        failure_manifesto_id=result.failure_manifesto_id,
    )
    # Trigger meta-analysis if enough new data
    if kb.entries_since_last_meta > META_ANALYSIS_THRESHOLD:
        kb.run_meta_analysis()
```

### 8.3 Meta-Analysis (Automatic)

- Scaling law fits across all runs
- Algorithm fingerprinting (hyperparameter sensitivity → embeddings)
- Failure manifold clustering (DBSCAN on error modes)
- Algorithm phylogeny (hierarchical clustering on fingerprints)

---

## 9. Resource Request Protocol (Article I, VII)

```python
def emit_resource_request(credit: float, gpus: int, justification: str):
    request = {
        "director_version": "1.0",
        "cycle": current_cycle,
        "available_credit": credit,
        "requested_gpus": gpus,
        "justification": justification,
        "verification_url": f"https://domain/director/cycle_{cycle}/report.md",
        "ledger_merkle_root": merkle_root(ledger),
        "ledger_proof": merkle_proof(ledger),
        "constitution_hash": hash(CONSTITUTION),
    }
    write_json("director_resource_request.json", request)
    # Also POST to resource coordinator if available
```

**Credit → GPU Mapping** (configurable):
- 0-5 credit: 1 GPU
- 5-15 credit: 2 GPUs
- 15-30 credit: 4 GPUs
- 30-60 credit: 8 GPUs
- 60+ credit: 16+ GPUs (multi-node)

---

## 10. Cycle Artifacts (Article VII)

Each cycle produces **one directory** with verifiable artifacts:

```
director_cycles/cycle_N/
├── report.md              # Human-readable: hypothesis, methods, results, Pareto plots, scaling laws
├── ledger.json            # Full credit ledger with Merkle proof
├── kb_delta.json          # KB entries added/updated this cycle
├── manifesto_delta.json   # Failure manifesto entries added
├── proposal.json          # The proposal that was executed
├── verification.json      # Gate results with timestamps
├── experiment_outputs/    # Raw data, logs, checkpoints
└── merkle_proof.json      # Cryptographic proof of ledger integrity
```

**Merkle Tree Structure:**
```
leaf_i = hash(VerifiedResult_i)
parent = hash(left_child + right_child)
root = merkle_root(ledger)
```
Anyone can verify: `verify_merkle_proof(ledger, proof, root)`

---

## 11. CLI Interface

```bash
# Daemon mode (continuous cycles until halt)
uv run biopl-director --daemon

# Single cycle (for CI, debugging)
uv run biopl-director --cycle

# Status query
uv run biopl-director --status
# → JSON: state, credit, current proposal, last result

# Credit ledger
uv run biopl-director --credit
# → JSON: ledger entries, available credit, decay status

# Constitution check
uv run biopl-director --constitution
# → Prints all articles with current compliance status
```

---

## 12. Initialization & Bootstrap

### 12.1 Initial Endowment

```python
INITIAL_ENDOWMENT = 10.0  # Credit for completing infrastructure
```

**Justification**: The framework has delivered:
- 74 registered models, 20 propagators, 111 components
- 2403 tests passing, all gates green
- Complete TileNet substrate (6 algorithms × 5 domains)
- GPU acceleration (16 Triton kernel families)
- AutoScientist with CoT, KB, dashboard, local LLM
- Deployment pipeline (ONNX, TorchScript, INT8, ternary)
- Analysis toolkit (14 modules)

This infrastructure **is** the initial credit. It enables the first experiments.

### 12.2 Bootstrap Sequence

```bash
# 1. Initialize Director state
uv run biopl-director --init

# 2. Verify constitution compliance
uv run biopl-director --constitution

# 3. Run first cycle (dry-run)
uv run biopl-director --cycle --dry-run

# 4. Start daemon
uv run biopl-director --daemon
```

---

## 13. Implementation Plan

### 13.1 Module Structure

```
bioplausible/director/
├── __init__.py
├── constitution.py      # CONSTITUTION dict, validation functions
├── state.py             # VerifiedResult, CreditLedger, Proposal, DirectorState
├── decide.py            # decide(), _propose(), _execute(), _verify(), _learn(), _report(), _request()
├── credit.py            # calculate_credit(), decay(), external_replication_bonus()
├── gates.py             # run_verification_gates(), gate implementations
├── kb_interface.py      # kb_query(), absorb_result(), meta_analysis_trigger()
├── manifesto.py         # generate_failure_manifesto(), validate_manifesto()
├── artifacts.py         # emit_cycle_report(), merkle_tree(), write_artifacts()
├── runtime.py           # Director class, run_cycle(), side effects
├── cli.py               # main(), argument parsing
└── constants.py         # MIN_CREDIT, BASE_VALUES, THRESHOLDS, HALF_LIFE
```

### 13.2 Dependencies (Existing, Verified)

| Component | Used For |
|-----------|----------|
| `biopl-run` | Experiment execution |
| `biopl-repro-check` | Reproducibility gate |
| `biopl-parity` | Parity benchmark gate |
| `biopl-verify-gradients` | Gradient equivalence gate |
| `biopl-registry-audit` | Registry audit gate |
| `KnowledgeBase` | KB query, absorption, meta-analysis |
| `FailureManifestoGenerator` | Structured failure documentation |
| `ExecutionEngine` | Experiment orchestration |
| `CoreTrainer` | Model training |

### 13.3 New Code Estimate

| Module | Lines | Complexity |
|--------|-------|------------|
| constitution.py | ~50 | Data only |
| state.py | ~80 | Frozen dataclasses |
| decide.py | ~120 | Pure functions |
| credit.py | ~40 | Pure functions |
| gates.py | ~60 | Subprocess wrappers |
| kb_interface.py | ~60 | KB wrappers |
| manifesto.py | ~40 | Validation |
| artifacts.py | ~80 | Merkle, report generation |
| runtime.py | ~150 | Async orchestration |
| cli.py | ~30 | Argument parsing |
| constants.py | ~30 | Configuration |
| **Total** | **~740** | **Pure + thin wrappers** |

---

## 14. First Cycle Projection

### 14.1 Bootstrap State

```json
{
  "ledger": {
    "entries": [],
    "initial_endowment": 10.0
  },
  "available_credit": 10.0,
  "cycle_count": 0,
  "status": "idle"
}
```

### 14.2 Cycle 0: TileNet Scaling Sweep

```json
{
  "proposal": {
    "hypothesis": "TileNet with algorithm=pc matches backprop on MNIST at 10× depth with 1/10 memory",
    "experiment_file": "experiments/tile_scaling.py",
    "config": {
      "models": ["conv_tile_pc", "conv_tile_ep", "conv_tile_fa", "backprop_mlp"],
      "depths": [2, 4, 8, 16, 32],
      "tasks": ["mnist"],
      "seeds": 5
    },
    "estimated_credit_cost": 3.0,
    "estimated_duration_hours": 4.0,
    "required_gpus": 2,
    "kb_lineage": ["tile_substrate_architecture", "eqprop_scaling_laws"],
    "gates_required": ["gradient_equivalence", "reproducibility", "parity", "registry_audit"],
    "expected_deliverables": ["pareto_frontier_plot", "scaling_law_fits", "algorithm_comparison_table"],
    "risk_assessment": {"divergence_at_depth": 0.15, "memory_overflow": 0.05}
  }
}
```

### 14.3 Expected Outcome

```json
{
  "result": {
    "experiment_id": "tile_scaling_20260819_cycle0",
    "credit_earned": 4.2,
    "metrics": {
      "conv_tile_pc_mnist_depth32_acc": 0.97,
      "backprop_mlp_mnist_depth32_acc": 0.98,
      "memory_ratio_pc_vs_bp": 0.12,
      "flops_ratio_pc_vs_bp": 1.8
    },
    "artifacts": ["pareto_frontier.html", "scaling_laws.json", "comparison_table.md"]
  },
  "new_available_credit": 14.2,
  "resource_request": {"gpus": 4, "justification": "Next: FA depth scaling to 1000 layers"}
}
```

---

## 15. Constitutional Compliance Checklist (Per Cycle)

| Check | When | Failure Mode |
|-------|------|--------------|
| `proposal.cost ≤ available_credit` | Propose | Halt: Article III |
| `proposal.kb_lineage ≠ ∅` | Propose | Halt: Article VI |
| `all_gates_passed(result)` | Verify | Halt: Article II |
| `manifesto_complete(result)` | Learn | Penalty: Article IV |
| `merkle_proof_valid(ledger)` | Report | Halt: Article VII |
| `request.gpus ≤ credit_to_gpus(available)` | Request | Halt: Article I |

---

## 16. Halting Conditions

The Director halts (requires human) when:

1. **Credit exhausted**: `available_credit < MIN_CREDIT_FOR_ANY_EXPERIMENT`
2. **No viable candidates**: KB query returns empty within budget
3. **Gate failure**: Any verification gate fails
4. **Constitution violation**: Any article violated
5. **External halt signal**: `SIGTERM` or `--halt` flag
6. **KB stagnation**: No KB growth for N cycles (configurable)

**On halt**: Emits final report, preserves full ledger, prints `halt_reason`.

---

## 17. External Interface (For Funders/Collaborators)

### 17.1 Verification Endpoint

```
GET /director/cycle_N/report.md      # Human-readable
GET /director/cycle_N/ledger.json    # Machine-readable
GET /director/cycle_N/merkle_proof.json  # Cryptographic proof
```

### 17.2 Reproduction Protocol

```bash
# Anyone can reproduce cycle N:
git checkout <result.git_commit>
uv run biopl-repro-check --experiment <result.experiment_id> --seed <result.seed>
```

### 17.3 Credit Audit

```bash
# Verify credit ledger integrity:
uv run biopl-director --audit --cycle N
# → Verifies Merkle proofs, gate results, KB lineage
```

---

## 18. Evolution Mechanisms

The Director **improves itself** through:

1. **Meta-analysis feedback**: KB meta-analysis identifies high-value experiment patterns → improves `kb_query()` ranking
2. **Failure manifold learning**: Clustered failures → better `risk_assessment` in proposals
3. **Credit calibration**: Track actual vs estimated cost → adjust `BASE_VALUES`
4. **Gate refinement**: Failed gates → improve experiment templates

**All evolution is logged** in the ledger as meta-experiments with their own credit accounting.

---

## 19. Open Questions (To Resolve Before Implementation)

| Question | Options | Recommendation |
|----------|---------|----------------|
| **KB query strategy** | Heuristic vs learned surrogate | Start heuristic; add surrogate after 50 cycles |
| **Multi-proposal parallelism** | Sequential only vs parallel within credit | Sequential first; parallel after credit > 20 |
| **Human oversight** | Full autonomy vs approval gate | Approval gate for `estimated_cost > 0.5 × available_credit` |
| **External replication** | Passive (wait) vs active (solicit) | Passive initially; active after 5 external reps |
| **Constitution amendment** | Immutable vs versioned | Versioned with supermajority (human + 3 external auditors) |

---

## 20. Success Criteria (For the Director Itself)

| Metric | 30 Days | 90 Days | 1 Year |
|--------|---------|---------|--------|
| **Verified results produced** | ≥ 10 | ≥ 50 | ≥ 200 |
| **Credit earned** | ≥ 20 | ≥ 100 | ≥ 500 |
| **External replications** | 0 | ≥ 3 | ≥ 20 |
| **KB entries** | +100 | +500 | +2000 |
| **Failure manifesto entries** | ≥ 5 | ≥ 30 | ≥ 100 |
| **Resource requests granted** | 1 | 5 | 20 |
| **Papers enabled** | 0 | 1 | 5 |

---

## Appendix A: Constitution Hash

```
CONSTITUTION_VERSION = "1.0.0"
CONSTITUTION_HASH = "sha256:..."  # Computed at runtime
```

**Any modification requires**: Human approval + 3 external auditor signatures + new hash recorded in ledger.

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Credit** | Verified compute capital. Earned only through passed gates. Decays with inactivity. |
| **VerifiedResult** | Atomic unit of credit. One experiment, all gates passed, fully reproducible. |
| **KB Lineage** | Citation of prior KB entries that the proposal extends or contradicts. |
| **Failure Manifesto** | Structured negative result documentation. Required for every failed experiment. |
| **Merkle Proof** | Cryptographic proof that ledger has not been tampered with. |
| **Minimal Disappointment** | The principle that `promise ≤ delivery` enforced by credit budgeting. |
| **Endowment** | Initial credit granted for infrastructure completion (10.0). |

---

*This specification is the Director's source of truth. The code implements the spec; the spec governs the code. Both evolve together, never diverge.*

**Version**: 1.0.0  
**Status**: Ready for implementation  
**Next**: Implement `bioplausible/director/` per this spec