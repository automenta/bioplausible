"""
Analysis Core Logic

Decoupled from UI to enable headless CLI usage.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

from bioplausible.hyperopt.comparison import (
    group_trials_by_family, 
    compute_algorithm_rankings, 
    ComparisonMetric
)


def load_trials(db_path: str) -> List[Dict[str, Any]]:
    """
    Load all trials from Optuna SQLite database.
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
        
        # Extract model name
        study_name = trial['study_name']
        if study_name.startswith('shallow_'):
            model_name = study_name.replace('shallow_', '').replace('_', ' ').title()
        else:
            model_name = study_name.replace('_', ' ').title()
        trial['model_name'] = model_name
        
        # Load trial values
        cursor.execute("""
            SELECT objective, value
            FROM trial_values
            WHERE trial_id = ?
            ORDER BY objective
        """, (trial_id,))
        
        values = cursor.fetchall()
        if len(values) >= 3:
            trial['accuracy'] = values[0]['value']
            trial['param_count'] = values[1]['value']
            trial['iteration_time'] = values[2]['value']
        else:
            trial['accuracy'] = 0.0
            trial['param_count'] = 0.0
            trial['iteration_time'] = 0.0
        
        # Load params
        cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,))
        
        params = cursor.fetchall()
        params = cursor.fetchall()
        trial['config'] = {p['param_name']: p['param_value'] for p in params}
        
        # Load user attributes (e.g. tier)
        cursor.execute("""
            SELECT key, value
            FROM trial_user_attributes
            WHERE trial_id = ?
        """, (trial_id,))
        attrs = cursor.fetchall()
        trial['user_attrs'] = {a['key']: a['value'] for a in attrs}
        
        # Extract tier specifically for top-level access
        # Default to 'shallow' if missing (legacy compatibility)
        trial['tier'] = trial['user_attrs'].get('tier', 'shallow')
        
        # Placeholders
        trial['final_loss'] = 0.0
        trial['perplexity'] = 0.0
        trial['status'] = 'completed'
        trial['epochs_completed'] = trial['config'].get('epochs', 5)
        
        trials.append(trial)
    
    conn.close()
    return trials


def compute_statistics(trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute statistics per model."""
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


def compute_pareto_frontier(trials: List[Dict[str, Any]]) -> List[int]:
    """Compute Pareto frontier trial IDs."""
    if not trials:
        return []
    
    pareto_ids = []
    
    for i, trial_a in enumerate(trials):
        is_dominated = False
        
        for j, trial_b in enumerate(trials):
            if i == j: continue
            
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


def get_rankings(trials: List[Dict[str, Any]]) -> List[Any]:
    """Compute comprehensive rankings with gap analysis."""
    trials_by_family = group_trials_by_family(trials)
    rankings = compute_algorithm_rankings(trials_by_family, metric=ComparisonMetric.ACCURACY)
    
    baseline = next((r for r in rankings if 'backprop' in r.family.lower() or 'baseline' in r.family.lower()), None)
    if baseline and baseline.best_value > 0:
        for r in rankings:
            gap = (baseline.best_value - r.best_value) / baseline.best_value * 100
            r.gap_to_baseline = gap
            
    return rankings
