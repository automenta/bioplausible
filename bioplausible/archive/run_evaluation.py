#!/usr/bin/env python3
"""
Novel Hybrid Algorithm Evaluation - Production Run

Comprehensive 1-hour evaluation of all 13 learning algorithms.

Usage:
    python run_evaluation.py [--hours HOURS] [--output-dir DIR]
    
Options:
    --hours HOURS       Duration in hours (default: 1.0)
    --output-dir DIR    Output directory (default: results/algorithm_comparison)
    --param-budgets     Comma-separated param budgets (default: 50000,100000,200000)
    --dataset          Dataset to use: mnist, cifar10, fashion (default: mnist)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import torch
from datetime import datetime
from pathlib import Path

from experiments.shallow_search import ShallowSearcher, load_mnist_subset
from algorithms import ALGORITHM_REGISTRY, list_algorithms


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive algorithm evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--hours', type=float, default=1.0, help='Time budget in hours')
    parser.add_argument('--output-dir', type=str, default='results/algorithm_comparison',
                       help='Output directory for results')
    parser.add_argument('--param-budgets', type=str, default='50000,100000,200000',
                       help='Comma-separated parameter budgets')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List all available algorithms and exit')
    
    args = parser.parse_args()
    
    if args.list_algorithms:
        list_algorithms()
        return
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    budgets = [int(b.strip()) for b in args.param_budgets.split(',')]
    algorithms = list(ALGORITHM_REGISTRY.keys())
    
    print("="*70)
    print("NOVEL HYBRID ALGORITHM EVALUATION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {args.hours:.2f} hours")
    print(f"Algorithms: {len(algorithms)}")
    print(f"Parameter Budgets: {budgets}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Training Samples: {args.n_samples:,}")
    print(f"Output Directory: {output_dir}")
    print("="*70)
    print()
    
    # Save configuration
    config = {
        'timestamp': timestamp,
        'duration_hours': args.hours,
        'algorithms': algorithms,
        'param_budgets': budgets,
        'dataset': args.dataset,
        'n_samples': args.n_samples,
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    print(f"Loading {args.dataset.upper()} dataset...")
    if args.dataset == 'mnist':
        train_loader, test_loader = load_mnist_subset(n_samples=args.n_samples)
        input_dim, output_dim = 784, 10
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet implemented")
    
    # Time per algorithm per budget
    time_per_algo = (args.hours * 3600) / (len(algorithms) * len(budgets))
    
    print(f"Time per algorithm per budget: {time_per_algo:.1f}s (~{time_per_algo/60:.1f} min)")
    print()
    
    # Run evaluation for each budget
    all_results = {}
    
    for budget in budgets:
        print(f"\n{'='*70}")
        print(f"PARAMETER BUDGET: {budget:,}")
        print(f"{'='*70}\n")
        
        searcher = ShallowSearcher(
            algorithms=algorithms,
            param_budget=budget,
        )
        
        results = searcher.ultra_shallow_eval(
            train_loader=train_loader,
            test_loader=test_loader,
            input_dim=input_dim,
            output_dim=output_dim,
            time_budget=time_per_algo,
        )
        
        searcher.print_summary()
        
        # Save intermediate results
        result_file = output_dir / f'results_budget_{budget}.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        all_results[f'{args.dataset}_{budget}'] = results
    
    # Generate final report
    _generate_markdown_report(all_results, output_dir, config)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


def _generate_markdown_report(results, output_dir, config):
    """Generate comprehensive markdown report."""
    lines = [
        "# Novel Hybrid Algorithm Evaluation Results",
        "",
        f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Duration**: {config['duration_hours']:.2f} hours",
        f"**Dataset**: {config['dataset'].upper()}",
        f"**Training Samples**: {config['n_samples']:,}",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Find overall winner
    best_overall = None
    best_acc = 0
    
    for task, res in results.items():
        successful = [(name, data['test_acc']) 
                     for name, data in res.items() 
                     if data.get('success', False)]
        if successful:
            best_in_task = max(successful, key=lambda x: x[1])
            if best_in_task[1] > best_acc:
                best_acc = best_in_task[1]
                best_overall = (best_in_task[0], task)
    
    if best_overall:
        lines.extend([
            f"**Best Overall**: `{best_overall[0]}` ({best_acc:.3f} accuracy on {best_overall[1]})",
            "",
        ])
    
    # Detailed results by budget
    lines.extend([
        "## Detailed Results",
        "",
    ])
    
    for task, res in results.items():
        dataset, budget = task.rsplit('_', 1)
        lines.extend([
            f"### {dataset.upper()} - {budget} parameters",
            "",
            "| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) | Params |",
            "|------|-----------|----------|-----------|--------|----------|--------|",
        ])
        
        successful = [(name, data) 
                     for name, data in res.items() 
                     if data.get('success', False)]
        successful.sort(key=lambda x: x[1]['test_acc'], reverse=True)
        
        for i, (name, data) in enumerate(successful, 1):
            lines.append(
                f"| {i} | `{name}` | {data['test_acc']:.3f} | "
                f"{data['train_acc']:.3f} | {data['epochs']} | "
                f"{data['time']:.1f} | {data['params']:,} |"
            )
        
        lines.append("")
    
    # Algorithm characteristics
    lines.extend([
        "## Algorithm Characteristics",
        "",
        "| Algorithm | Type | Key Innovation |",
        "|-----------|------|----------------|",
    ])
    
    algo_notes = {
        'backprop': ('Baseline', 'Standard gradient descent'),
        'eqprop': ('Baseline', 'Contrastive Hebbian learning'),
        'feedback_alignment': ('Baseline', 'Random fixed feedback'),
        'eq_align': ('Hybrid', 'EqProp dynamics + FA training'),
        'ada_fa': ('Hybrid', 'Adaptive feedback evolution'),
        'cf_align': ('Hybrid', 'Contrastive via FA signals'),
        'leq_fa': ('Hybrid', 'Layer-wise local settling'),
        'pc_hybrid': ('Radical', 'Predictive coding + FA'),
        'eg_fa': ('Radical', 'Energy-guided updates'),
        'sparse_eq': ('Radical', 'Top-K sparse dynamics'),
        'mom_eq': ('Radical', 'Momentum-accelerated settling'),
        'sto_fa': ('Radical', 'Stochastic feedback dropout'),
        'em_fa': ('Radical', 'Energy minimization objective'),
    }
    
    for algo in config['algorithms']:
        if algo in algo_notes:
            typ, innov = algo_notes[algo]
            lines.append(f"| `{algo}` | {typ} | {innov} |")
    
    lines.append("")
    
    # Save report
    report_path = output_dir / 'report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nMarkdown report saved to: {report_path}")


if __name__ == '__main__':
    main()
