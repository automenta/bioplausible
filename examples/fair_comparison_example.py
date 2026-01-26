"""
Fair Algorithm Comparison Example

Demonstrates patience-based evaluation tiers for fair comparison across algorithms.
"""

from bioplausible.hyperopt import (
    PatientLevel,
    create_optuna_space,
    create_study,
    get_evaluation_config,
    print_evaluation_summary,
)
from bioplausible.models.registry import get_model_spec

print("=" * 70)
print("Fair Algorithm Comparison with Patience Levels")
print("=" * 70)

# Example 1: SMOKE test - Ultra-fast validation
print("\n📊 Example 1: SMOKE Test (Ultra-Fast Validation)")
print("-" * 70)

eval_config = get_evaluation_config(PatientLevel.SMOKE, model_family="eqprop")
print_evaluation_summary(PatientLevel.SMOKE, n_models=3)

print("Config details:")
print(f"  Epochs: {eval_config.epochs}")
print(f"  Max hidden: {eval_config.max_hidden_dim}")
print(f"  Max layers: {eval_config.max_layers}")
print(f"  Train samples: {eval_config.train_samples}")
print(f"  Trials: {eval_config.n_trials}")
print(f"  Pruning: {eval_config.use_pruning}")

# Example 2: SHALLOW - Quick comparison
print("\n\n📊 Example 2: SHALLOW Mode (Quick Exploration)")
print("-" * 70)

print_evaluation_summary(PatientLevel.SHALLOW, n_models=5)

# Demonstrate usage
study = create_study(
    model_names=["EqProp MLP"],
    n_objectives=2,
    sampler_name="tpe",
    evaluation_config=get_evaluation_config(PatientLevel.SHALLOW),
)

print("✅ Created study with SHALLOW evaluation config")
print(f"   Sampler startup trials: {study.sampler._n_startup_trials}")

def shallow_objective(trial):
    """Example objective with shallow config."""
    eval_config = get_evaluation_config(PatientLevel.SHALLOW)
    
    # Auto-constrained by evaluation_config
    config = create_optuna_space(
        trial, 
        "EqProp MLP",
        evaluation_config=eval_config
    )
    
    print(f"\n   Trial {trial.number}:")
    print(f"     Epochs: {config['epochs']}")
    print(f"     Hidden dim: {config['hidden_dim']}")
    print(f"     Layers: {config['num_layers']}")
    
    # Simulate training (in reality, call run_single_trial_task)
    accuracy = 0.7 + (config["lr"] * 5) + (config["beta"] * 0.1)
    loss = 0.5 - (config["lr"] * 2)
    
    return min(1.0, accuracy), max(0.0, loss)

print("\nRunning 3 trials with SHALLOW config...")
study.optimize(shallow_objective, n_trials=3, show_progress_bar=False)

print(f"\n✅ Completed {len(study.trials)} trials")
print(f"   Best accuracy: {study.best_trials[0].values[0]:.4f}")

# Example 3: Comparing STANDARD vs DEEP
print("\n\n📊 Example 3: STANDARD vs DEEP Comparison")
print("-" * 70)

print("\n** STANDARD MODE (Balanced):")
print_evaluation_summary(PatientLevel.STANDARD, n_models=3)

print("\n** DEEP MODE (Overnight Run):")
print_evaluation_summary(PatientLevel.DEEP, n_models=3)

# Example 4: Multi-model comparison with same patience
print("\n\n📊 Example 4: Fair Multi-Model Comparison")
print("-" * 70)

models = ["EqProp MLP", "DFA (Direct Feedback Alignment)", "Backprop Baseline"]
patience = PatientLevel.SHALLOW

print(f"Comparing {len(models)} models with {patience.value.upper()} patience:")
print_evaluation_summary(patience, n_models=len(models))

results = {}

for model in models:
    spec = get_model_spec(model)
    eval_config = get_evaluation_config(patience, model_family=spec.family)
    
    study = create_study(
        model_names=[model],
        n_objectives=1,
        sampler_name="tpe", 
        evaluation_config=eval_config,
    )
    
    def objective(trial):
        config = create_optuna_space(
            trial, model, evaluation_config=eval_config
        )
        # Simulate
        return 0.75 + (hash(str(config)) % 100) / 200
    
    study.optimize(objective, n_trials=eval_config.n_trials, show_progress_bar=False)
    results[model] = study.best_value
    
    print(f"\n  {model}:")
    print(f"    Family: {spec.family}")
    print(f"    Adjusted epochs: {eval_config.epochs}")
    print(f"    Best accuracy: {study.best_value:.4f}")

print("\n✅ Fair Comparison Results (sorted):")
for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"   {acc:.4f} - {model}")

# Example 5: Scaling recommendation
print("\n\n📊 Example 5: Patience Level Recommendations")
print("-" * 70)

print("""
When to use each patience level:

SMOKE (1-2 min/trial):
  • Quick sanity checks
  • Debugging hyperopt setup
  • Initial exploration
  • CI/CD smoke tests
  ✅ Good for: Verifying code works
  ❌ Not for: Final model selection

SHALLOW (5-10 min/trial):
  • Rapid prototyping
  • Comparing many models quickly
  • Finding promising hyperparameter ranges
  • Interactive exploration
  ✅ Good for: Narrowing down options
  ❌ Not for: Publication results

STANDARD (30-60 min/trial):
  • Balanced evaluation
  • Most research use cases
  • Reliable model comparison
  • Production hyperparameter selection
  ✅ Good for: Most use cases
  ❌ Not for: Hasty decisions

DEEP (2-4 hours/trial):
  • Overnight optimization
  • Publication-quality results
  • Final model selection
  • Maximum performance
  ✅ Good for: Best possible results
  ❌ Not for: Quick iterations
""")

print("\n" + "=" * 70)
print("✅ Fair Comparison Examples Complete!")
print("=" * 70)

print("\n💡 Key Benefits:")
print("  • Fair comparison: Same compute budget per model")
print("  • Patience control: Trade speed for thoroughness")
print("  • Smoke tests: Quick validation even with 3 epochs")
print("  • Scalable: From 1-min smoke to overnight deep runs")
print("  • Family-aware: Adjusts for model characteristics (EqProp, etc.)")
