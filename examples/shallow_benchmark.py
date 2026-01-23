"""
Complete SHALLOW Benchmark Evaluation

Runs fair comparison across multiple models with SHALLOW patience level.
Shows top configurations and insights from results.
"""

import time
from pathlib import Path

import optuna

from bioplausible.hyperopt import (
    PatientLevel,
    create_optuna_space,
    create_study,
    get_evaluation_config,
    print_evaluation_summary,
)
from bioplausible.hyperopt.runner import run_single_trial_task
from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec

print("=" * 80)
print("SHALLOW BENCHMARK: Fair Algorithm Comparison")
print("=" * 80)

# Select representative models from different families
models_to_test = [
    "Backprop Baseline",
    "EqProp MLP",
    "DFA (Direct Feedback Alignment)",
]

# Patience level
patience = PatientLevel.SHALLOW
eval_config = get_evaluation_config(patience)

print("\n📋 Evaluation Configuration")
print_evaluation_summary(patience, n_models=len(models_to_test))

# Task
task = "mnist"
print(f"Task: {task}")
print(f"Models: {', '.join(models_to_test)}\n")

# Storage
storage_path = "sqlite:///examples/shallow_benchmark.db"
print(f"Storage: {storage_path}\n")

results = {}
all_trials = {}

for model_name in models_to_test:
    print("=" * 80)
    print(f"Evaluating: {model_name}")
    print("=" * 80)
    
    spec = get_model_spec(model_name)
    print(f"Family: {spec.family}")
    
    # Create study
    study = create_study(
        model_names=[model_name],
        n_objectives=2,  # accuracy, loss
        storage=storage_path,
        study_name=f"shallow_{model_name.replace(' ', '_').lower()}",
        evaluation_config=eval_config,
    )
    
    # Check if we already have trials
    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"\n⚠️  Found {existing_trials} existing trials, resuming...")
        if existing_trials >= eval_config.n_trials:
            print(f"✅ Already completed {existing_trials} trials, skipping")
            all_trials[model_name] = study.trials
            results[model_name] = {
                "study": study,
                "best_trials": study.best_trials[:3],
                "n_trials": len(study.trials),
            }
            continue
    
    start_time = time.time()
    
    def objective(trial):
        print(f"\n  Trial {trial.number + 1}/{eval_config.n_trials}")
        
        # Generate config with constraints
        config = create_optuna_space(
            trial, model_name, evaluation_config=eval_config
        )
        
        print(f"    lr={config['lr']:.6f}, hidden={config['hidden_dim']}, layers={config['num_layers']}", end="")
        if 'beta' in config:
            print(f", beta={config['beta']:.3f}", end="")
        if 'steps' in config:
            print(f", steps={config['steps']}", end="")
        print()
        
        # Run trial
        metrics = run_single_trial_task(
            task=task,
            model_name=model_name,
            config=config,
            quick_mode=True,
            verbose=False,  # Set to True for debugging
        )
        
        if metrics:
            accuracy = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", float("inf"))
            print(f"    → acc={accuracy:.4f}, loss={loss:.4f}")
            return accuracy, loss
        else:
            print(f"    → FAILED")
            raise optuna.TrialPruned()
    
    # Run optimization
    print(f"\nRunning {eval_config.n_trials} trials...")
    study.optimize(
        objective,
        n_trials=eval_config.n_trials - existing_trials,
        show_progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    
    # Store results
    all_trials[model_name] = study.trials
    results[model_name] = {
        "study": study,
        "best_trials": study.best_trials[:3],
        "n_trials": len(study.trials),
        "time": elapsed,
    }
    
    print(f"\n✅ Completed in {elapsed/60:.1f} minutes")
    print(f"   Total trials: {len(study.trials)}")
    print(f"   Pareto frontier: {len(study.best_trials)} trials")

# Analysis
print("\n\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

# 1. Best configurations per model
print("\n📊 Top 3 Configurations per Model")
print("-" * 80)

for model_name, result in results.items():
    print(f"\n**{model_name}**")
    
    for i, trial in enumerate(result["best_trials"][:3], 1):
        acc = trial.values[0] if trial.values else 0.0
        loss = trial.values[1] if len(trial.values) > 1 else 0.0
        
        print(f"\n  #{i}: Trial {trial.number}")
        print(f"     Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        print(f"     Hyperparameters:")
        for param, value in sorted(trial.params.items()):
            if param != "epochs":
                print(f"       {param}: {value}")

# 2. Overall comparison
print("\n\n📊 Model Comparison (Best Pareto Trials)")
print("-" * 80)

comparison = []
for model_name, result in results.items():
    if result["best_trials"]:
        best = result["best_trials"][0]
        comparison.append({
            "model": model_name,
            "accuracy": best.values[0],
            "loss": best.values[1],
            "trial_num": best.number,
        })

comparison.sort(key=lambda x: x["accuracy"], reverse=True)

print(f"\n{'Rank':<5} {'Model':<35} {'Accuracy':<10} {'Loss':<10} {'Trial #'}")
print("-" * 80)
for rank, entry in enumerate(comparison, 1):
    print(f"{rank:<5} {entry['model']:<35} {entry['accuracy']:<10.4f} {entry['loss']:<10.4f} {entry['trial_num']}")

# 3. Hyperparameter insights
print("\n\n📊 Hyperparameter Sensitivity Analysis")
print("-" * 80)

for model_name, result in results.items():
    print(f"\n**{model_name}**")
    
    completed_trials = [t for t in all_trials[model_name] if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) < 5:
        print("  (Insufficient trials for analysis)")
        continue
    
    # Analyze lr
    lrs = [t.params['lr'] for t in completed_trials]
    accs = [t.values[0] for t in completed_trials]
    
    # Correlation analysis (simple)
    mean_lr = sum(lrs) / len(lrs)
    mean_acc = sum(accs) / len(accs)
    
    # Top 25% trials
    sorted_trials = sorted(completed_trials, key=lambda t: t.values[0], reverse=True)
    top_25_pct = sorted_trials[:max(1, len(sorted_trials) // 4)]
    
    top_lr_mean = sum(t.params['lr'] for t in top_25_pct) / len(top_25_pct)
    top_hidden_mean = sum(t.params['hidden_dim'] for t in top_25_pct) / len(top_25_pct)
    
    print(f"  Top 25% trials prefer:")
    print(f"    Learning rate: {top_lr_mean:.6f} (vs overall mean: {mean_lr:.6f})")
    print(f"    Hidden dim: {top_hidden_mean:.0f}")
    
    if 'beta' in top_25_pct[0].params:
        top_beta_mean = sum(t.params['beta'] for t in top_25_pct) / len(top_25_pct)
        all_beta_mean = sum(t.params['beta'] for t in completed_trials) / len(completed_trials)
        print(f"    Beta: {top_beta_mean:.3f} (vs overall mean: {all_beta_mean:.3f})")

# 4. Optimization efficiency
print("\n\n📊 Optimization Efficiency")
print("-" * 80)

for model_name, result in results.items():
    if "time" in result:
        avg_time = result["time"] / result["n_trials"]
        print(f"\n{model_name}:")
        print(f"  Total time: {result['time']/60:.1f} minutes")
        print(f"  Avg time per trial: {avg_time:.1f} seconds")
        print(f"  Trials completed: {result['n_trials']}")
        
        # Pruning effectiveness
        pruned = sum(1 for t in all_trials[model_name] if t.state == optuna.trial.TrialState.PRUNED)
        if pruned > 0:
            print(f"  Pruned trials: {pruned} ({pruned/result['n_trials']*100:.1f}%)")

# Summary
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n✅ Evaluation Complete")
print(f"   Patience level: {patience.value.upper()}")
print(f"   Epochs per trial: {eval_config.epochs}")
print(f"   Trials per model: {eval_config.n_trials}")
print(f"   Models evaluated: {len(models_to_test)}")

winner = comparison[0]
print(f"\n🏆 Best Model: {winner['model']}")
print(f"   Accuracy: {winner['accuracy']:.4f}")
print(f"   Loss: {winner['loss']:.4f}")

print(f"\n💾 Results saved to: {storage_path}")
print("   View with: optuna-dashboard " + storage_path.replace("sqlite:///", ""))

print("\n📊 Key Insights:")
print("   • All models evaluated with same compute budget")
print("   • Pareto frontier shows accuracy/loss tradeoffs")
print("   • Hyperparameter preferences differ by algorithm")
print("   • Results are reproducible from database")
