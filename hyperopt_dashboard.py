#!/usr/bin/env python3
"""
Hyperparameter Optimization Dashboard

Live dashboard for multi-objective hyperparameter optimization.
Visualizes Pareto frontiers, tracks experiment progress, and exports best configurations.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QComboBox, QSpinBox, QTextEdit, QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np

from hyperopt.engine import EvolutionaryOptimizer, OptimizationConfig
from hyperopt.experiment import TrialRunner
from hyperopt.storage import HyperoptStorage
from hyperopt.metrics import get_pareto_frontier, rank_trials
from gui.algorithms import MODEL_REGISTRY


class OptimizationWorker(QThread):
    """Background worker that runs trials."""
    
    trial_completed = pyqtSignal(int, bool)  # trial_id, success
    status_update = pyqtSignal(str)
    
    def __init__(self, optimizer, runner):
        super().__init__()
        self.optimizer = optimizer
        self.runner = runner
        self.running = True
        self.paused = True
    
    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            # Get next trial
            trial_id = self.optimizer.get_next_trial()
            
            if trial_id is None:
                # No pending trials, evolve population
                self.status_update.emit("Evolving to next generation...")
                for model_name in self.optimizer.model_names:
                    self.optimizer.evolve_generation(model_name)
                time.sleep(1)
                continue
            
            # Run trial
            trial = self.runner.storage.get_trial(trial_id)
            self.status_update.emit(f"Running trial {trial_id}: {trial.model_name}")
            
            success = self.runner.run_trial(trial_id)
            self.trial_completed.emit(trial_id, success)
            
            # Update Pareto frontiers
            self.optimizer.update_pareto_frontiers()
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def stop(self):
        self.running = False
        self.wait()


class HyperoptDashboard(QMainWindow):
    """Main dashboard window."""
    
    def __init__(self, task='shakespeare', db_path=None, quick_mode=True, population_size=10, n_generations=5):
        super().__init__()
        self.task = task
        self.setWindowTitle(f"Hyperopt Dashboard - Task: {task.upper()}")
        self.setGeometry(100, 100, 1800, 1000)
        
        # Initialize components
        self.storage = HyperoptStorage(db_path or f"results/hyperopt_{task}.db")
        
        # Get model names from registry
        self.model_names = [spec.name for spec in MODEL_REGISTRY]
        
        # Optimizer
        self.optimizer = EvolutionaryOptimizer(
            self.model_names,
            OptimizationConfig(population_size=population_size, n_generations=n_generations),
            self.storage
        )
        
        # Trial runner
        self.runner = TrialRunner(self.storage, task=task, quick_mode=quick_mode)
        
        # Worker thread
        self.worker = OptimizationWorker(self.optimizer, self.runner)
        self.worker.trial_completed.connect(self.on_trial_completed)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.start()
        self.worker.pause()
        
        # UI
        self.init_ui()
        
        # Timer for updating plots
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(2000)  # Update every 2s
    
    def init_ui(self):
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        
        # Top: Controls
        controls = self.create_controls()
        layout.addWidget(controls)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Pareto frontier plots
        pareto_group = self.create_pareto_plots()
        main_splitter.addWidget(pareto_group)
        
        # Bottom splitter
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Experiment status
        status_group = self.create_status_table()
        bottom_splitter.addWidget(status_group)
        
        # Best configs
        configs_group = self.create_best_configs()
        bottom_splitter.addWidget(configs_group)
        
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 2)
        
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        
        layout.addWidget(main_splitter)
    
    def create_controls(self):
        group = QGroupBox("Optimization Controls")
        layout = QHBoxLayout(group)
        
        # Task label
        task_label = QLabel(f"<b>Task:</b> {self.task.upper()}")
        task_label.setStyleSheet("padding: 10px; font-size: 14px; color: #2ecc71;")
        layout.addWidget(task_label)
        
        layout.addWidget(QLabel("|"))
        
        # Start/Pause button
        self.start_btn = QPushButton("▶ Start Optimization")
        self.start_btn.clicked.connect(self.toggle_optimization)
        self.start_btn.setStyleSheet(
            "font-weight: bold; background-color: #27ae60; color: white; padding: 10px; font-size: 14px;"
        )
        layout.addWidget(self.start_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 10px; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Model selection
        layout.addWidget(QLabel("Active Models:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("All Models")
        for name in self.model_names:
            self.model_combo.addItem(name)
        layout.addWidget(self.model_combo)
        
        # Export button
        export_btn = QPushButton("📥 Export Best Configs")
        export_btn.clicked.connect(self.export_configs)
        layout.addWidget(export_btn)
        
        return group
    
    def create_pareto_plots(self):
        group = QGroupBox("Pareto Frontier Visualization")
        layout = QHBoxLayout(group)
        
        # 4 scatter plots showing different trade-offs
        plots_layout = [[None, None], [None, None]]
        
        # Accuracy vs Perplexity
        self.plot_acc_ppl = pg.PlotWidget(title="Accuracy vs Perplexity")
        self.plot_acc_ppl.setLabel('bottom', 'Perplexity (lower is better)')
        self.plot_acc_ppl.setLabel('left', 'Accuracy')
        self.plot_acc_ppl.showGrid(x=True, y=True, alpha=0.3)
        plots_layout[0][0] = self.plot_acc_ppl
        
        # Accuracy vs Speed
        self.plot_acc_speed = pg.PlotWidget(title="Accuracy vs Speed")
        self.plot_acc_speed.setLabel('bottom', 'Iteration Time (ms)')
        self.plot_acc_speed.setLabel('left', 'Accuracy')
        self.plot_acc_speed.showGrid(x=True, y=True, alpha=0.3)
        plots_layout[0][1] = self.plot_acc_speed
        
        # Accuracy vs Params
        self.plot_acc_params = pg.PlotWidget(title="Accuracy vs Parameters")
        self.plot_acc_params.setLabel('bottom', 'Parameters (M)')
        self.plot_acc_params.setLabel('left', 'Accuracy')
        self.plot_acc_params.showGrid(x=True, y=True, alpha=0.3)
        plots_layout[1][0] = self.plot_acc_params
        
        # Speed vs Params
        self.plot_speed_params = pg.PlotWidget(title="Speed vs Parameters")
        self.plot_speed_params.setLabel('bottom', 'Parameters (M)')
        self.plot_speed_params.setLabel('left', 'Iteration Time (ms)')
        self.plot_speed_params.showGrid(x=True, y=True, alpha=0.3)
        plots_layout[1][1] = self.plot_speed_params
        
        # Add to layout in 2x2 grid
        grid_widget = QWidget()
        grid_layout = QVBoxLayout(grid_widget)
        
        for row in plots_layout:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            for plot in row:
                row_layout.addWidget(plot)
            grid_layout.addWidget(row_widget)
        
        layout.addWidget(grid_widget)
        
        # Color map for models
        self.colors = {}
        for i, spec in enumerate(MODEL_REGISTRY):
            self.colors[spec.name] = spec.color
        
        return group
    
    def create_status_table(self):
        group = QGroupBox("Experiment Status")
        layout = QVBoxLayout(group)
        
        self.status_table = QTableWidget()
        self.status_table.setColumnCount(6)
        self.status_table.setHorizontalHeaderLabels([
            'Trial ID', 'Model', 'Epoch', 'Accuracy', 'Perplexity', 'Status'
        ])
        layout.addWidget(self.status_table)
        
        # Statistics
        self.stats_label = QLabel("No trials yet")
        self.stats_label.setStyleSheet("padding: 5px; font-size: 11px;")
        layout.addWidget(self.stats_label)
        
        return group
    
    def create_best_configs(self):
        group = QGroupBox("Best Configurations")
        layout = QVBoxLayout(group)
        
        # Tabs for each model
        self.config_tabs = QTabWidget()
        
        for model_name in self.model_names:
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QFont("Monospace", 9))
            self.config_tabs.addTab(text_edit, model_name.split('(')[0].strip())
        
        layout.addWidget(self.config_tabs)
        
        return group
    
    def toggle_optimization(self):
        if self.worker.paused:
            self.worker.resume()
            self.start_btn.setText("⏸ Pause Optimization")
            self.start_btn.setStyleSheet(
                "font-weight: bold; background-color: #f39c12; color: white; padding: 10px; font-size: 14px;"
            )
        else:
            self.worker.pause()
            self.start_btn.setText("▶ Resume Optimization")
            self.start_btn.setStyleSheet(
                "font-weight: bold; background-color: #27ae60; color: white; padding: 10px; font-size: 14px;"
            )
    
    def on_trial_completed(self, trial_id, success):
        if success:
            self.status_label.setText(f"✅ Trial {trial_id} completed")
        else:
            self.status_label.setText(f"❌ Trial {trial_id} failed")
        self.update_visualizations()
    
    def on_status_update(self, message):
        self.status_label.setText(message)
    
    def update_visualizations(self):
        """Update all plots and tables."""
        # Get all completed trials
        all_trials = self.storage.get_all_trials(status='completed')
        
        if not all_trials:
            return
        
        # Update Pareto plots
        self.plot_acc_ppl.clear()
        self.plot_acc_speed.clear()
        self.plot_acc_params.clear()
        self.plot_speed_params.clear()
        
        # Group by model
        trials_by_model = {}
        for trial in all_trials:
            if trial.model_name not in trials_by_model:
                trials_by_model[trial.model_name] = []
            trials_by_model[trial.model_name].append(trial)
        
        # Plot each model
        for model_name, trials in trials_by_model.items():
            color = self.colors.get(model_name, '#888888')
            
            # Extract data
            accs = [t.accuracy for t in trials]
            ppls = [t.perplexity for t in trials]
            times = [t.iteration_time * 1000 for t in trials]  # ms
            params = [t.param_count for t in trials]  # millions
            
            # Identify Pareto frontier
            frontier_indices = get_pareto_frontier(trials)
            
            # Plot regular points
            self.plot_acc_ppl.plot(
                ppls, accs,
                pen=None, symbol='o', symbolSize=6,
                symbolBrush=color, name=model_name
            )
            self.plot_acc_speed.plot(
                times, accs,
                pen=None, symbol='o', symbolSize=6,
                symbolBrush=color
            )
            self.plot_acc_params.plot(
                params, accs,
                pen=None, symbol='o', symbolSize=6,
                symbolBrush=color
            )
            self.plot_speed_params.plot(
                params, times,
                pen=None, symbol='o', symbolSize=6,
                symbolBrush=color
            )
            
            # Highlight Pareto frontier
            if frontier_indices:
                f_accs = [trials[i].accuracy for i in frontier_indices]
                f_ppls = [trials[i].perplexity for i in frontier_indices]
                f_times = [trials[i].iteration_time * 1000 for i in frontier_indices]
                f_params = [trials[i].param_count for i in frontier_indices]
                
                self.plot_acc_ppl.plot(
                    f_ppls, f_accs,
                    pen=None, symbol='star', symbolSize=12,
                    symbolBrush=color, symbolPen='w'
                )
                self.plot_acc_speed.plot(
                    f_times, f_accs,
                    pen=None, symbol='star', symbolSize=12,
                    symbolBrush=color, symbolPen='w'
                )
                self.plot_acc_params.plot(
                    f_params, f_accs,
                    pen=None, symbol='star', symbolSize=12,
                    symbolBrush=color, symbolPen='w'
                )
                self.plot_speed_params.plot(
                    f_params, f_times,
                    pen=None, symbol='star', symbolSize=12,
                    symbolBrush=color, symbolPen='w'
                )
        
        # Update status table
        self.update_status_table()
        
        # Update best configs
        self.update_best_configs()
        
        # Update statistics
        stats = self.optimizer.get_statistics()
        stats_text = f"Generation {stats['generation']} | "
        for model_name, model_stats in stats['models'].items():
            stats_text += f"{model_name.split('(')[0].strip()}: {model_stats['completed']}/{model_stats['total_trials']} | "
        self.stats_label.setText(stats_text)
    
    def update_status_table(self):
        # Get recent trials
        all_trials = self.storage.get_all_trials()
        recent_trials = sorted(all_trials, key=lambda t: t.trial_id, reverse=True)[:20]
        
        self.status_table.setRowCount(len(recent_trials))
        
        for i, trial in enumerate(recent_trials):
            self.status_table.setItem(i, 0, QTableWidgetItem(str(trial.trial_id)))
            self.status_table.setItem(i, 1, QTableWidgetItem(trial.model_name.split('(')[0].strip()))
            self.status_table.setItem(i, 2, QTableWidgetItem(f"{trial.epochs_completed}"))
            self.status_table.setItem(i, 3, QTableWidgetItem(f"{trial.accuracy:.3f}"))
            self.status_table.setItem(i, 4, QTableWidgetItem(f"{trial.perplexity:.2f}"))
            self.status_table.setItem(i, 5, QTableWidgetItem(trial.status))
    
    def update_best_configs(self):
        for i, model_name in enumerate(self.model_names):
            best_trials = self.optimizer.get_best_configs(model_name, top_k=3)
            
            if not best_trials:
                text = "No completed trials yet"
            else:
                text = f"Top 3 Configurations for {model_name}\n\n"
                for rank, trial in enumerate(best_trials, 1):
                    text += f"{'='*60}\n"
                    text += f"Rank {rank} | Trial {trial.trial_id}\n"
                    text += f"{'='*60}\n"
                    text += f"Accuracy: {trial.accuracy:.4f}\n"
                    text += f"Perplexity: {trial.perplexity:.2f}\n"
                    text += f"Speed: {trial.iteration_time*1000:.1f}ms/iter\n"
                    text += f"Params: {trial.param_count:.2f}M\n"
                    text += f"Composite Score: {trial.composite_score():.4f}\n\n"
                    text += "Config:\n"
                    text += json.dumps(trial.config, indent=2)
                    text += "\n\n"
            
            self.config_tabs.widget(i).setText(text)
    
    def export_configs(self):
        """Export best configurations to JSON files."""
        export_dir = Path("results/best_configs")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name in self.model_names:
            best_trials = self.optimizer.get_best_configs(model_name, top_k=5)
            
            if best_trials:
                export_data = {
                    'model_name': model_name,
                    'top_configs': [
                        {
                            'rank': i + 1,
                            'trial_id': trial.trial_id,
                            'config': trial.config,
                            'metrics': {
                                'accuracy': trial.accuracy,
                                'perplexity': trial.perplexity,
                                'iteration_time': trial.iteration_time,
                                'param_count': trial.param_count,
                                'composite_score': trial.composite_score()
                            }
                        }
                        for i, trial in enumerate(best_trials)
                    ]
                }
                
                filename = export_dir / f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
        
        self.status_label.setText(f"✅ Exported best configs to {export_dir}")
    
    def closeEvent(self, event):
        self.worker.stop()
        self.storage.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization Dashboard")
    parser.add_argument('--task', type=str, default='shakespeare',
                        choices=['shakespeare', 'wikitext', 'mnist', 'cifar10'],
                        help='Task to optimize for (default: shakespeare)')
    parser.add_argument('--db', type=str, default=None,
                        help='Database path (default: results/hyperopt_{task}.db)')
    parser.add_argument('--quick', action='store_true', default=True,
                        help='Quick mode (5 epochs, faster) vs full mode (20 epochs)')
    parser.add_argument('--population', type=int, default=10,
                        help='Population size for evolutionary search (default: 10)')
    parser.add_argument('--generations', type=int, default=5,
                        help='Number of generations (default: 5)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Hyperparameter Optimization Dashboard")
    print(f"{'='*60}")
    print(f"Task: {args.task.upper()}")
    print(f"Mode: {'Quick (5 epochs)' if args.quick else 'Full (20 epochs)'}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Database: {args.db or f'results/hyperopt_{args.task}.db'}")
    print(f"{'='*60}\n")
    
    app = QApplication(sys.argv)
    window = HyperoptDashboard(
        task=args.task,
        db_path=args.db,
        quick_mode=args.quick,
        population_size=args.population,
        n_generations=args.generations
    )
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
