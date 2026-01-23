# DEPRECATED: Custom Evolution Code

> **⚠️ This directory is deprecated and will be removed in v0.2.0**

All evolution functionality has been replaced by [Optuna](https://optuna.org/).

## Migration Guide

### Before (Custom Evolution)
```python
from bioplausible.evolution import EvolutionEngine, VariationBreeder

engine = EvolutionEngine(config)
engine.run(n_generations=50)
```

### After (Optuna)
```python
from bioplausible.hyperopt.optuna_bridge import create_study, create_optuna_space

study = create_study(
    model_names=["EqProp MLP"],
    sampler_name="nsga2"  # Multi-objective
)

def objective(trial):
    config = create_optuna_space(trial, "EqProp MLP")
    # ... run training ...
    return accuracy, loss

study.optimize(objective, n_trials=100)
```

## Why Deprecated?

1. **Code Reduction**: Removes ~1,100+ lines of custom code
2. **Better Algorithms**: Optuna's TPE and NSGA-II are well-tested
3. **Automatic Pruning**: Stops bad trials early (30-50% compute savings)
4. **Built-in Viz**: Parameter importance, Pareto fronts, etc.
5. **Persistence**: SQLite/PostgreSQL storage out of the box

## What Was Deleted

- `evolution/breeder.py` (293 lines) - Genetic operations
- `evolution/engine.py` (410 lines) - Evolution loop
- `evolution/evaluator.py` (~350 lines) - Tiered evaluation
- Custom NSGA-II implementation

All replaced by ~200 lines of Optuna bridge code.

## Files Kept

The following files are kept for reference but not imported:

- `algorithm.py` - Algorithm wrapper (may be useful)
- `fitness.py` - Fitness scoring (may be useful)
- `pareto.py` - Pareto utilities (Optuna handles this)
- Other utility files

## Removal Timeline

- **v0.1.0**: Marked as deprecated (this release)
- **v0.2.0**: Directory removed entirely

For issues or questions, see: https://github.com/yourusername/bioplausible/issues
