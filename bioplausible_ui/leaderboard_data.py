"""
Leaderboard Data Processing

Loads trial data from SQLite and prepares it for visualization.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


def load_trials(db_path: str) -> List[Dict[str, Any]]:
    """
    Load all trials from SQLite database.
    
    Args:
        db_path: Path to SQLite database (without sqlite:/// prefix)
    
    Returns:
        List of trial dictionaries with all metrics and parameters
    """
    if not Path(db_path).exists():
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    # Query trials
    trials = []
    cursor.execute("""
        SELECT 
            t.trial_id,
            t.model_name,
            t.config,
            t.status,
            t.epochs_completed,
            t.final_loss,
            t.accuracy,
            t.perplexity,
            t.iteration_time,
            t.param_count
        FROM trials t
        WHERE t.status = 'completed'
        ORDER BY t.trial_id
    """)
    
    for row in cursor.fetchall():
        trial = dict(row)
        # Parse JSON config
        trial['config'] = json.loads(trial['config']) if trial['config'] else {}
        trials.append(trial)
    
    conn.close()
    return trials


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
            # trial_b dominates if it's >= in all objectives and > in at least one
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
    from collections import defaultdict
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
    from collections import defaultdict
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
        'pareto_ids':pareto_ids,
        'statistics': stats,
        'best_per_model': best_per_model,
        'timestamp': str(Path(trials[0]['config']).parent) if trials else None,  # Just for testing
    }
