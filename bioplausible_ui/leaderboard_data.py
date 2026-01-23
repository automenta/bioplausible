"""
Leaderboard Data Processing - Optuna Compatible

Loads trial data from Optuna SQLite database and prepares it for visualization.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from collections import defaultdict


def load_trials(db_path: str) -> List[Dict[str, Any]]:
    """
    Load all trials from Optuna SQLite database.
    
    Args:
        db_path: Path to SQLite database (without sqlite:/// prefix)
    
    Returns:
        List of trial dictionaries with all metrics and parameters
    """
    if not Path(db_path).exists():
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query trials with study names
    trials = []
    cursor.execute("""
        SELECT 
            t.trial_id,
            t.number,
            t.state,
            s.study_name
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE t.state = 'COMPLETE'
        ORDER BY t.trial_id
    """)
    
    for row in cursor.fetchall():
        trial = dict(row)
        trial_id = trial['trial_id']
        
        # Extract model name from study name (e.g., "shallow_eqprop_mlp" -> "EqProp MLP")
        study_name = trial['study_name']
        if study_name.startswith('shallow_'):
            model_name = study_name.replace('shallow_', '').replace('_', ' ').title()
        else:
            model_name = study_name.replace('_', ' ').title()
        trial['model_name'] = model_name
        
        # Load trial values (objectives: accuracy, params, time)
        cursor.execute("""
            SELECT objective, value
            FROM trial_values
            WHERE trial_id = ?
            ORDER BY objective
        """, (trial_id,))
        
        values = cursor.fetchall()
        if len(values) >= 3:
            trial['accuracy'] = values[0]['value']  # Objective 0: accuracy
            trial['param_count'] = values[1]['value']  # Objective 1: param_count
            trial['iteration_time'] = values[2]['value']  # Objective 2: time
        else:
            # Fallback for incomplete data
            trial['accuracy'] = 0.0
            trial['param_count'] = 0.0
            trial['iteration_time'] = 0.0
        
        # Load trial parameters (hyperparameters)
        cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,))
        
        params = cursor.fetchall()
        trial['config'] = {p['param_name']: p['param_value'] for p in params}
        
        # Add placeholder fields for compatibility
        trial['final_loss'] = 0.0
        trial['perplexity'] = 0.0
        trial['status'] = 'completed'
        trial['epochs_completed'] = trial['config'].get('epochs', 5)
        
        trials.append(trial)
    
    conn.close()
    return trials


def load_trials_timeseries(db_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load trials ordered by time/ID for progress visualization.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        Dictionary mapping model_name to list of trials (time-ordered)
    """
    trials = load_trials(db_path)
    
    # Group by model and sort by trial_id (proxy for time)
    model_series = defaultdict(list)
    for trial in sorted(trials, key=lambda t: t['trial_id']):
        model_series[trial['model_name']].append(trial)
    
    return dict(model_series)


def compute_pareto_frontier(trials: List[Dict[str, Any]]) -> List[int]:
    """
    Compute Pareto frontier trial IDs.
    
    A trial is Pareto-optimal if no other trial is better in all objectives.
    Objectives: maximize accuracy, minimize param_count, minimize iteration_time
    
    Args:
        trials: List of trial dictionaries
    
    Returns:
        List of trial_ids on the Pareto frontier
    """
    if not trials:
        return []
    
    pareto_ids = []
    
    for i, trial_a in enumerate(trials):
        is_dominated = False
        
        for j, trial_b in enumerate(trials):
            if i == j:
                continue
            
            # Check if trial_b dominates trial_a
            better_acc = trial_b['accuracy'] >= trial_a['accuracy']
            better_params = trial_b['param_count'] <= trial_a['param_count']
            better_time = trial_b['iteration_time'] <= trial_a['iteration_time']
            
            strictly_better = (
                trial_b['accuracy'] > trial_a['accuracy'] or
                trial_b['param_count'] < trial_a['param_count'] or
                trial_b['iteration_time'] < trial_a['iteration_time']
            )
            
            if better_acc and better_params and better_time and strictly_better:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_ids.append(trial_a['trial_id'])
    
    return pareto_ids


def compute_statistics(trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics per model.
    
    Args:
        trials: List of trial dictionaries
    
    Returns:
        Dictionary mapping model_name to statistics
    """
    import numpy as np
    
    stats = defaultdict(lambda: {
        'accuracy': [],
        'param_count': [],
        'iteration_time': [],
    })
    
    for trial in trials:
        model = trial['model_name']
        stats[model]['accuracy'].append(trial['accuracy'])
        stats[model]['param_count'].append(trial['param_count'])
        stats[model]['iteration_time'].append(trial['iteration_time'])
    
    # Compute mean and std
    result = {}
    for model, metrics in stats.items():
        result[model] = {
            'accuracy_mean': float(np.mean(metrics['accuracy'])),
            'accuracy_std': float(np.std(metrics['accuracy'])),
            'param_count_mean': float(np.mean(metrics['param_count'])),
            'param_count_std': float(np.std(metrics['param_count'])),
            'time_mean': float(np.mean(metrics['iteration_time'])),
            'time_std': float(np.std(metrics['iteration_time'])),
            'n_trials': len(metrics['accuracy']),
        }
    
    return result


def format_for_frontend(trials: List[Dict[str, Any]], pareto_ids: List[int]) -> Dict[str, Any]:
    """
    Format trial data for frontend consumption.
    
    Args:
        trials: List of trial dictionaries
        pareto_ids: List of Pareto-optimal trial IDs
    
    Returns:
        Dictionary with all data for frontend
    """
    stats = compute_statistics(trials)
    
    # Mark Pareto trials
    for trial in trials:
        trial['is_pareto'] = trial['trial_id'] in pareto_ids
    
    # Best trials per model
    best_per_model = {}
    model_trials = defaultdict(list)
    
    for trial in trials:
        model_trials[trial['model_name']].append(trial)
    
    for model, model_trial_list in model_trials.items():
        # Best = highest accuracy
        best = max(model_trial_list, key=lambda t: t['accuracy'])
        best_per_model[model] = best
    
    return {
        'trials': trials,
        'pareto_ids': pareto_ids,
        'statistics': stats,
        'best_per_model': best_per_model,
    }
