# Evolution System Architecture

## Overview

The EqProp+SN Evolution System is designed with clean abstractions and modularity.

```
evolution/
├── __init__.py          # Package exports
├── base.py              # Abstract interfaces (ABC)
├── config.py            # Constants and configurations
├── utils.py             # Helper functions
├── models.py            # Model registry (centralized building)
├── fitness.py           # Multi-objective fitness scoring
├── breeder.py           # Genetic operations
├── evaluator.py         # Tiered evaluation pipeline
├── breakthrough.py      # Breakthrough detection
└── engine.py            # Main evolution loop
```

## Key Design Patterns

### 1. Abstract Base Classes (`base.py`)

Defines extensible interfaces:
- `ModelBuilder`: Build models from configs
- `Evaluator`: Evaluate architectures
- `SelectionStrategy`: Parent selection
- `BreedingStrategy`: Offspring generation
- `TerminationCriterion`: Stop conditions

### 2. Model Registry (`models.py`)

Centralized, validated model building:

```python
@ModelRegistry.register('my_model')
def build_my_model(config, input_dim, output_dim):
    return MyModel(...)

# Usage
builder = DefaultModelBuilder()
model = builder.build(config, task='mnist')
```

### 3. Configuration Management (`config.py`)

Centralized constants:
- Tier configurations
- Task definitions
- Model constraints
- Fitness weights

### 4. Utilities (`utils.py`)

Reusable helper functions:
- Logging setup
- Random seeding
- Parameter counting
- Time formatting

## Extensibility Points

### Adding New Model Types

```python
# In models.py
@ModelRegistry.register('new_model')
def build_new_model(config, input_dim, output_dim):
    return NewModel(...)

# In config.py - add constraints
MODEL_CONSTRAINTS['new_model'] = {
    'max_depth': 100,
    'supports_sn': True,
}
```

### Adding New Tasks

```python
# In config.py
TASK_CONFIGS['new_task'] = {
    'input_dim': 1000,
    'output_dim': 20,
    'type': 'classification',
}
```

### Custom Selection Strategies

```python
from evolution.base import SelectionStrategy

class TournamentSelection(SelectionStrategy):
    def select_parents(self, population, n_parents):
        # Custom tournament logic
        return selected
```

## Code Quality Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Lines of code | ~2,000 | ~2,300 (+15% for abstractions) |
| Model building | Duplicated in evaluator (67 lines) | Centralized registry (5 lines) |
| Type hints | Partial | Comprehensive |
| Error handling | Basic | Validated + logged |
| Constants | Inline magic numbers | Centralized config |
| Extensibility | Modify core files | Extend via registry |

## Testing

The refactored system maintains 100% API compatibility:

```bash
# All previous commands still work
python evolve.py --task mnist --dry-run
python verify.py --track 60 --quick
```
