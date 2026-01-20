# Research System Architecture

> **Technical Documentation**: Implementation details of the research management system

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                       User Interface                     │
│                      research.py CLI                     │
└────────────┬────────────────────────────────────────────┘
             │
             ├── Status Display
             ├── Action Dispatcher
             └── Error Handler
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼────┐  ┌─────▼──────┐  ┌──▼─────────┐
│   Config   │  │   State    │  │   Scripts  │
│   YAML     │  │    JSON    │  │            │
└────────────┘  └────────────┘  └────────────┘
     │               │               │
     │               │               │
     └───────[ResearchManager]───────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼─┐    ┌───▼──┐   ┌───▼────┐
    │Claims│    │ Eval │   │ Actions│
    └──────┘    └──────┘   └────────┘
```

---

## Core Components

### 1. Configuration System (`research_config.yaml`)

**Purpose**: Single source of truth for research state

**Structure**:
```yaml
novelty: {...}          # Novelty verification status
claims: {...}           # All research claims
publishability: {...}   # Scoring thresholds
experiments: {...}      # Experiment configurations
papers: {...}           # Paper templates
```

**Benefits**:
- Add claims without code changes
- Version control friendly
- Human-readable and editable
- Validates on load

### 2. State Persistence (`.research_state.json`)

**Purpose**: Track progress between runs

**Contents**:
- Current seed counts per claim
- Last update timestamp
- Accumulated experimental results
- Novelty confirmation

**Lifecycle**:
1. Loaded on startup
2. Merged with config
3. Updated during actions
4. Saved after each action

### 3. ResearchManager Class

**Responsibilities**:
- Load config and state
- Display status
- Execute actions
- Handle errors
- Save state

**Key Methods**:
```python
print_status()          # Display research state
continue_research()     # Run next experiment
run_validation()        # Validate claims
generate_figures()      # Create figures
generate_paper()        # Generate paper draft
prepare_arxiv()         # Package for submission
run_full_pipeline()     # Complete workflow
```

---

## Data Models

### Claim Dataclass

```python
@dataclass
class Claim:
    name: str
    description: str
    evidence_paths: List[str]
    priority: int = 1
    validated: bool = False
    confidence: float = 0.0
    required_seeds: int = 3
    current_seeds: int = 0
    status: str = "pending"
```

**Design Decisions**:
- Immutable evidence requirements
- Confidence in [0.0, 1.0]
- Priority (1=highest)
- Status is human-readable string

### ResearchState Dataclass

```python
@dataclass
class ResearchState:
    config: ResearchConfig
    last_updated: str
    accumulated_results: Dict
```

**Design Decisions**:
- Config contains claims
- Results stored separately
- Timestamps for auditability

---

## Publishability Calculation

### Algorithm

```python
def calculate_publishability():
    # Weight by validation, confidence, and priority
    score = sum(
        claim.confidence * 
        (1.0 if validated else 0.3) * 
        (2.0 - priority * 0.3)
        for claim in claims
    ) / (total_claims * 1.7)
    
    # Classify
    if score >= 0.85 and validated_count >= 4:
        return score, "READY"
    elif score >= 0.70 and validated_count >= 3:
        return score, "ALMOST READY"
    else:
        return score, "NEEDS WORK"
```

### Weighting Factors

| Factor | Formula | Range |
|--------|---------|-------|
| Validation | 1.0 (valid) or 0.3 (incomplete) | [0.3, 1.0] |
| Confidence | claim.confidence | [0.0, 1.0] |
| Priority | 2.0 - priority * 0.3 | [0.8, 2.0] |

**Example**:
- Priority 1, validated, 95% confidence: 0.95 * 1.0 * 2.0 = 1.90
- Priority 3, incomplete, 30% confidence: 0.30 * 0.3 * 1.1 = 0.099

**Normalization**: Sum / (count * 1.7) brings to [0, 1] range

---

## Error Handling Strategy

### Levels

1. **Try/Except at Action Level**
   - All user-facing actions wrapped
   - Errors logged to file
   - Graceful degradation

2. **Validation at Load**
   - Config file syntax check
   - Path existence validation
   - Fallback to defaults

3. **Subprocess Error Handling**
   - Capture stdout/stderr
   - Log command and output
   - Return error to user

### Example

```python
def continue_research(self, target):
    try:
        self._run_experiment(target)
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed: {e}")
        print(f"Error: {e.stderr}")
    except Exception as e:
        logger.error(traceback.format_exc())
        print(f"Unexpected error: {e}")
    finally:
        self.state.save()  # Always save state
```

---

## Logging System

### Log Levels

| Level | Usage |
|-------|-------|
| DEBUG | Detailed traces (--verbose) |
| INFO | Normal operations |
| WARNING | Recoverable issues |
| ERROR | Action failures |

### Destinations

1. **File**: `research.log` (persistent, all levels)
2. **Console**: stdout (INFO and above)

### Log Format

```
2025-12-31 10:24:23,142 - INFO - ResearchManager initialized successfully
```

---

## Extensibility Points

### 1. Adding New Claim

**Where**: `research_config.yaml`
```yaml
claims:
  new_claim:
    name: "New Finding"
    # ...
```

**No code changes needed**

### 2. Adding New Action

**Where**: `ResearchManager` class
```python
def my_action(self):
    """New action."""
    # Implementation

# In main():
elif args.action == "my_action":
    manager.my_action()
```

### 3. Custom Scoring

**Where**: `ResearchState.calculate_publishability()`
```python
def calculate_publishability(self):
    # Custom algorithm
    score = custom_calculation()
    return score, status
```

### 4. New Output Format

**Where**: `ResearchManager.generate_paper()`
```python
def generate_paper(self, format="markdown"):
    if format == "latex":
        # Generate LaTeX
    elif format == "docx":
        # Generate Word doc
```

---

## Testing Strategy

### Manual Tests

```bash
# 1. Status display
python research.py

# 2. Config validation
python -c "
import yaml
with open('research_config.yaml') as f:
    config = yaml.safe_load(f)
print('Config valid:', config['version'])
"

# 3. State persistence
python research.py  # Creates state
cat .research_state.json  # Verify
python research.py  # Should reload
```

### Integration Tests

```bash
# Full pipeline
python research.py --action full

# Verify outputs
ls -l figures/
ls -l papers/
ls -l arxiv_submission/
```

### Error Tests

```bash
# Missing config
mv research_config.yaml research_config.yaml.bak
python research.py  # Should use defaults

# Invalid action
python research.py --action invalid  # Should show help

# Keyboard interrupt
python research.py --action full
# Press Ctrl+C - should save state
```

---

## Performance Considerations

### Load Time
Optimizations:
- Config loaded once at startup
- State persisted as JSON (fast)
- Lazy loading of experimental results

### Memory Usage
- Dataclasses are efficient
- Results stored as references
- Logs rotated automatically

### Scalability
Current limits:
- ~100 claims: O(n) operations acceptable
- Config file: <1MB typical
- State file: <100KB typical

---

## Security Considerations

### File Access
- Only writes to project directory
- No network access
- Scripts verified before execution

### Input Validation
- Config schema validation
- Path sanitization
- Command injection prevention

### Logging
- No sensitive data logged
- Logs can be audited
- Rotation prevents disk fill

---

## Future Enhancements

### Planned

1. **Web Dashboard**
   - Flask/FastAPI server
   - Real-time status updates
   - Interactive experiment launching

2. **Distributed Experiments**
   - Slurm/PBS integration
   - Parallel seed runs
   - Cloud execution

3. **Advanced Analytics**
   - Trend analysis
   - Confidence intervals
   - Statistical tests

4. **Export Formats**
   - LaTeX generation
   - Word/PDF export
   - Presentation slides

### Possible

1. **Git Integration**
   - Auto-commit results
   - Tag releases
   - Compare branches

2. **Notification System**
   - Email on completion
   - Slack/Discord webhooks
   - Mobile push

3. **Collaborative Features**
   - Multi-user state
   - Claim attribution
   - Review workflow

---

## Dependencies

### Required
- Python 3.8+
- PyYAML (optional, JSON fallback)

### Optional
- Matplotlib (for figures)
- Pandoc (for paper conversion)
- Git (for version control)

### Installation

```bash
# Minimal
python -m pip install pyyaml

# Full
python -m pip install pyyaml matplotlib pandas
sudo apt install pandoc  # or brew install pandoc
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Config not loaded | YAML syntax error | Validate with `python -c "import yaml; yaml.safe_load(open('research_config.yaml'))"` |
| State not saved | Permission denied | Check write permissions |
| Experiment fails | Script not found | Verify path in config |
| Figures fail | Matplotlib error | Falls back automatically to text |

### Debug Mode

```bash
# Enable verbose logging
python research.py --verbose

# Check log
tail -f research.log

# Inspect state
cat .research_state.json | python -m json.tool
```

---

## Maintenance

### Regular Tasks

1. **Review logs**: Check for warnings/errors
2. **Update config**: Add new claims as discovered
3. **Validate state**: Ensure seed counts accurate
4. **Backup**: Commit config and state to git

### Periodic Tasks

1. **Rotate logs**: Archive old `research.log`
2. **Clean cache**: Remove temp files
3. **Update docs**: Keep guide current
4. **Audit claims**: Remove obsolete

---

## Version History

### v2.0 (2025-12-31)
- YAML configuration support
- State persistence
- Comprehensive error handling
- Logging system
- Extensible design

### v1.0 (2025-12-31)
- Initial release
- Basic status display
- Hardcoded claims
- Simple actions

---

## Contributing

To contribute improvements:

1. Test changes thoroughly
2. Update documentation
3. Add examples to guide
4. Commit with clear messages
5. Consider backward compatibility
