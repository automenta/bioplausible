#!/usr/bin/env python3
"""
Hyperparameter Search and Comparison Tool - Entry Point

This module serves as the main entry point for the hyperparameter optimization dashboard.
It combines all the decomposed modules into a cohesive system.
"""

import sys
import argparse
from eqprop_trainer.hyperopt_dashboard import HyperoptSearchDashboard
from PyQt6.QtWidgets import QApplication


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search and Comparison Tool")
    parser.add_argument('--task', type=str, required=True,
                        choices=['shakespeare', 'wikitext', 'mnist', 'cifar10'],
                        help='Task to optimize for (MANDATORY)')
    parser.add_argument('--db', type=str, default=None,
                        help='Database path (default: results/hyperopt_search_{task}.db)')
    parser.add_argument('--quick', action='store_true', default=True,
                        help='Quick mode (5 epochs, faster) vs full mode (20 epochs)')
    parser.add_argument('--population', type=int, default=10,
                        help='Population size for evolutionary search (default: 10)')
    parser.add_argument('--generations', type=int, default=5,
                        help='Number of generations (default: 5)')
    parser.add_argument('--max_trial_time', type=float, default=60.0,
                        help='Total time budget per trial in seconds (default: 60.0)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Hyperparameter Search and Comparison Tool")
    print(f"{'='*60}")
    print(f"Task: {args.task.upper()}")
    print(f"Mode: {'Quick (5 epochs)' if args.quick else 'Full (20 epochs)'}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Database: {args.db or f'results/hyperopt_search_{args.task}.db'}")
    print(f"{'='*60}\n")

    # Enable high DPI scaling
    if hasattr(sys.modules.get('PyQt6.QtCore'), 'Qt'):
        from PyQt6.QtCore import Qt
        if hasattr(Qt, 'ApplicationAttribute'):
            if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
                QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
            if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
                QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Hyperparameter Search Dashboard")
    app.setApplicationVersion("1.0.0")

    window = HyperoptSearchDashboard(
        task=args.task,
        db_path=args.db,
        quick_mode=args.quick,
        population_size=args.population,
        n_generations=args.generations,
        max_trial_time=args.max_trial_time
    )
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()