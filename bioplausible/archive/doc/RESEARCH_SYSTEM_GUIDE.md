# Research System Guide

> **Quick Reference**: How to use the TorEqProp research management system

---

## Quick Start

```bash
# Show research status and publishability
python research.py

# Continue research (run next suggested experiment)
python research.py --action continue

# Validate all claims
python research.py --action validate

# Generate figures
python research.py --action figures

# Generate paper draft
python research.py --action paper

# Prepare arXiv submission
python research.py --action arxiv

# Run complete pipeline
python research.py --action full
```

---

## System Architecture

### Files

| File | Purpose |
|------|---------|
| `research.py` | Main research manager (v2.0) |
| `research_config.yaml` | Research configuration (claims, experiments, papers) |
| `.research_state.json` | Persistent state (auto-saved) |
| `research.log` | Complete activity log |

### Design Principles

1. **Configuration-driven**: Edit `research_config.yaml` to add claims/experiments
2. **State persistence**: Progress automatically saved
3. **Robust error handling**: All errors logged, graceful degradation
4. **Extensible**: Add new claims/actions without code changes

---

## Adding a New Claim

Edit `research_config.yaml`:

```yaml
claims:
  my_new_claim:
    name: "My New Finding"
    description: "Description of the claim"
    priority: 1  # 1=highest priority
    validated: false
    confidence: 0.0
    required_seeds: 3
    current_seeds: 0
    evidence_paths:
      - "docs/MY_EVIDENCE.md"
      - "results/my_results.json"
    status: "pending"
```

Then add corresponding experiment:

```yaml
experiments:
  my_new_claim:
    script: "scripts/my_experiment.py"
    args: ["--seeds", "3"]
    timeout_minutes: 60
    output: "results/my_results.json"
```

Run: `python research.py --action continue --target my_new_claim`

---

## Publishability Score

The system calculates a weighted publishability score based on:

- **Validation status**: Validated claims weighted 1.0, incomplete 0.3
- **Confidence**: Higher confidence = higher contribution
- **Priority**: Priority 1 claims weighted more than priority 3

### Thresholds

| Score | Status | Meaning |
|-------|--------|---------|
| ≥85% | 🟢 READY TO PUBLISH | 4+ validated claims, high confidence |
| ≥70% | 🟡 ALMOST READY | 3+ validated claims, strengthen evidence |
| <70% | 🔴 NEEDS MORE WORK | More validation required |

---

## Action Details

### `continue`
- Runs experiments for lowest-confidence validated claims
- Updates seed counts automatically
- Saves state after completion

### `validate`
- Runs `scripts/generate_paper.py --validate-claims`
- Checks all evidence files exist
- Prints validation report

### `figures`
- Attempts to use matplotlib if available
- Falls back to text-based markdown descriptions
- Saves to `figures/` directory

### `paper`
- Runs `scripts/generate_paper.py --paper <name>`
- Fills in template markers with real data
- Outputs to `papers/<name>_paper_generated.md`

### `arxiv`
- Copies paper and figures to `arxiv_submission/`
- Creates README with conversion instructions
- Ready for arXiv upload

### `full`
- Runs all actions in sequence: validate → figures → paper → arxiv
- Continues on errors (logs failures)
- Complete publication pipeline

---

## Error Handling

All errors are logged to `research.log` with full stack traces.

Common issues:

| Error | Solution |
|-------|----------|
| Config file not found | Create `research_config.yaml` from template |
| Script not found | Check `scripts/` directory |
| Matplotlib import error | Fallback to text descriptions (automatic) |
| Permission denied | Check file permissions |

---

## State Persistence

State is automatically saved to `.research_state.json` after each action.

Contains:
- Current seed counts per claim
- Last update timestamp
- Accumulated results
- Novelty confirmation status

To reset: `rm .research_state.json`

---

## Logging

All activity logged to `research.log`:

```bash
# View logs
tail -f research.log

# View errors only
grep ERROR research.log

# Enable verbose logging
python research.py --verbose
```

---

## Extending the System

### Add New Action

Edit `research.py`, add method to `ResearchManager`:

```python
def my_new_action(self):
    """Description of action."""
    print("Running my action...")
    # Implementation here
```

Add to main():
```python
elif args.action == "my_action":
    manager.my_new_action()
```

### Custom Publishability Calculation

Edit `ResearchState.calculate_publishability()`:

```python
def calculate_publishability(self) -> Tuple[float, str]:
    # Custom scoring logic
    score = custom_calculation()
    status = determine_status(score)
    return score, status
```

---

## Best Practices

1. **Always use config file**: Don't hardcode claims
2. **Save state frequently**: State saved automatically after actions
3. **Check logs**: `research.log` has full details
4. **Version config**: Commit `research_config.yaml` to git
5. **Document evidence**: Update evidence paths as you produce results

---

## Troubleshooting

### Status shows old data
```bash
# Force reload from config
rm .research_state.json
python research.py
```

### Experiments not running
```bash
# Check configuration
python -c "
import yaml
with open('research_config.yaml') as f:
    print(yaml.safe_load(f)['experiments'])
"
```

### Publishability score wrong
```bash
# Check claim weights
python research.py --verbose
grep "calculate_publishability" research.log
```

---

## Complete Workflow Example

```bash
# 1. Check current status
python research.py

# 2. Run next suggested experiment
python research.py --action continue

# 3. Validate all claims (with statistical tests)
python research.py --action validate

# 4. Generate publication materials
python research.py --action full

# 5. Review outputs
cat papers/spectral_normalization_paper_generated.md
ls -l figures/
ls -l arxiv_submission/

# 6. Submit to arXiv
cd arxiv_submission/
pandoc paper.md -o paper.pdf
# Upload to arXiv
```

---

## Support

For issues:
1. Check `research.log` for errors
2. Verify `research_config.yaml` syntax
3. Ensure all scripts in `scripts/` directory exist
4. Test with `--verbose` flag for detailed output
