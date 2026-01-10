"""
Hyperopt dashboard module for hyperparameter optimization.

This module contains the main dashboard window for hyperparameter search and comparison.
"""

import sys
import json
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .config import GLOBAL_CONFIG
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QComboBox, QSpinBox, QTextEdit, QCheckBox, QScrollArea, QProgressBar,
    QFrame, QMessageBox, QAbstractItemView
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pyqtgraph as pg

from hyperopt.engine import EvolutionaryOptimizer, OptimizationConfig
from hyperopt.experiment import TrialRunner
from hyperopt.storage import HyperoptStorage
from hyperopt.metrics import get_pareto_frontier, rank_trials
from gui.algorithms import MODEL_REGISTRY
from eqprop_trainer.experiment_scheduler import ExperimentScheduler
from eqprop_trainer.hyperopt_worker import HyperoptSearchWorker


class HyperoptSearchDashboard(QMainWindow):
    """Main dashboard window for hyperparameter search and comparison."""

    def __init__(self, task='shakespeare', db_path=None, quick_mode=True, population_size=10, n_generations=5, max_trial_time=60.0):
        super().__init__()
        # Set global config
        GLOBAL_CONFIG.task = task
        GLOBAL_CONFIG.quick_mode = quick_mode
        GLOBAL_CONFIG.epochs = 3 if quick_mode else 20
        GLOBAL_CONFIG.max_trial_time = max_trial_time
        
        self.task = task
        self.setWindowTitle(f"Hyperparameter Search Dashboard - Task: {task.upper()}")
        self.setGeometry(100, 100, 1800, 1000)

        # Initialize components - but don't start optimization yet
        self.storage = None
        self.optimizer = None
        self.runner = None
        self.scheduler = None
        self.worker = None
        self.baseline_results = {}
        # Initialize baseline table (even though it's created later in create_baseline_section)
        self.baseline_table = None

        # Store initial parameters
        self.initial_task = task
        self.initial_db_path = db_path
        self.initial_quick_mode = quick_mode
        self.initial_population_size = population_size
        self.initial_n_generations = n_generations

        # Get model names from registry
        self.model_names = [spec.name for spec in MODEL_REGISTRY]

        # Initially not running
        self.is_initialized = False

        # Auto-initialize optimization since task is mandatory
        QTimer.singleShot(100, self.initialize_optimization)


        # Timer for updating plots - will be started after initialization
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualizations)

        # Timer for frequent status updates - will be started after initialization
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_info)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        # Top: Controls (always visible)
        controls = self.create_controls()
        layout.addWidget(controls)

        # Progress and status section (always visible)
        progress_group = self.create_progress_section()
        layout.addWidget(progress_group)

        # Main content area with experiment list as primary interface
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Experiment list (primary interface)
        self.experiment_list = self.create_experiment_list()
        main_splitter.addWidget(self.experiment_list)

        # Pareto plots (secondary visualization)
        pareto_group = self.create_pareto_plots()
        main_splitter.addWidget(pareto_group)

        # Set stretch factors to emphasize experiment list
        main_splitter.setStretchFactor(0, 3)  # Experiment list gets more space
        main_splitter.setStretchFactor(1, 2)  # Pareto plots get less space

        layout.addWidget(main_splitter)

    def create_collapsible_section(self, title, content_widget):
        """Create a collapsible section with the given title and content."""
        from PyQt6.QtWidgets import QToolButton, QFrame
        from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve

        # Main group box
        group_box = QGroupBox(title)
        group_layout = QVBoxLayout(group_box)

        # Header with toggle button
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Toggle button
        toggle_button = QToolButton()
        toggle_button.setCheckable(True)
        toggle_button.setChecked(True)  # Start expanded
        toggle_button.setArrowType(Qt.ArrowType.DownArrow)
        toggle_button.setStyleSheet("""
            QToolButton {
                border: none;
                font-weight: bold;
                font-size: 12px;
                padding: 5px;
            }
        """)
        toggle_button.setText(title)

        header_layout.addWidget(toggle_button)
        header_layout.addStretch()
        header_widget.setMaximumHeight(30)

        group_layout.addWidget(header_widget)

        # Content frame
        content_frame = QFrame()
        content_frame.setLayout(QVBoxLayout(content_frame))
        content_frame.layout().addWidget(content_widget)
        content_frame.setMinimumHeight(200)  # Set minimum height for content area

        group_layout.addWidget(content_frame)

        # Animation for smooth collapse/expand
        animation = QPropertyAnimation(content_frame, b"maximumHeight")
        animation.setDuration(200)
        animation.setStartValue(0)
        animation.setEndValue(content_frame.sizeHint().height() + 20)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        # Connect toggle to animation
        def toggle_content():
            checked = toggle_button.isChecked()
            toggle_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

            if checked:
                content_frame.setMaximumHeight(content_frame.sizeHint().height() + 20)
            else:
                content_frame.setMaximumHeight(0)

        toggle_button.toggled.connect(toggle_content)

        return group_box

    def create_progress_section(self):
        """Create a section with overall progress and insights."""
        group = QGroupBox("Overall Progress & Insights")
        layout = QHBoxLayout(group)

        # Overall progress bar
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("Initialization Phase: Baselines Running...")
        self.overall_progress.setToolTip("Overall progress through the hyperparameter optimization pipeline")
        layout.addWidget(self.overall_progress)

        # Insights panel
        insights_layout = QVBoxLayout()

        # Current phase label
        self.phase_label = QLabel("Phase: Running Baselines")
        self.phase_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 12px;")
        insights_layout.addWidget(self.phase_label)

        # Insights text
        self.insights_text = QTextEdit()
        self.insights_text.setMaximumHeight(80)
        self.insights_text.setReadOnly(True)
        self.insights_text.setFont(QFont("Arial", 9))
        self.insights_text.setHtml("<font color='#808090'>Initializing... Baselines will establish performance benchmarks.</font>")
        insights_layout.addWidget(self.insights_text)

        layout.addLayout(insights_layout)

        return group

    def create_experiment_list(self):
        """Create a sortable experiment list as the primary interface."""
        group = QGroupBox("Experiments")
        layout = QVBoxLayout(group)

        # Create a table for experiments with sorting enabled
        self.experiment_table = QTableWidget()
        self.experiment_table.setColumnCount(10)
        self.experiment_table.setHorizontalHeaderLabels([
            'Trial ID', 'Model', 'Status', 'Epoch', 'Accuracy', 'Perplexity', 'Time (ms)', 'Params (M)', 'Acc/Param', 'Acc/Time'
        ])

        # Enable sorting
        self.experiment_table.setSortingEnabled(True)

        # Set selection behavior
        self.experiment_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.experiment_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Enable context menu
        self.experiment_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.experiment_table.customContextMenuRequested.connect(self.spawn_trainer_context_menu)

        layout.addWidget(self.experiment_table)

        return group

    def create_controls(self):
        group = QGroupBox("Optimization Controls")
        layout = QHBoxLayout(group)

        # Start/Pause button (initially auto-started)
        self.start_btn = QPushButton("⏸ Pause Optimization")
        self.start_btn.clicked.connect(self.toggle_optimization)
        self.start_btn.setStyleSheet(
            "font-weight: bold; background-color: #f39c12; color: white; padding: 10px; font-size: 14px;"
        )
        layout.addWidget(self.start_btn)



        # Status label
        self.status_label = QLabel("Initializing...")
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

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.update_visualizations)
        layout.addWidget(refresh_btn)


        # Add a reset button to clear and restart optimization
        reset_btn = QPushButton("🔄 Reset Optimization")
        reset_btn.clicked.connect(self.reset_optimization)
        reset_btn.setStyleSheet(
            "font-weight: bold; background-color: #e74c3c; color: white; padding: 10px; font-size: 14px;"
        )
        layout.addWidget(reset_btn)
        
        # Checkbox to show/hide pruned trials
        self.show_pruned_cb = QCheckBox("Show Pruned")
        self.show_pruned_cb.setChecked(False)  # Hide by default
        self.show_pruned_cb.toggled.connect(self.update_visualizations)
        layout.addWidget(self.show_pruned_cb)

        return group

    # on_dataset_selected removed - optimization starts automatically

    def initialize_optimization(self):
        """Initialize optimization components after dataset selection."""
        if self.is_initialized:
            return  # Already initialized

        # Initialize components
        self.storage = HyperoptStorage(self.initial_db_path or f"results/hyperopt_search_{self.task}.db")

        # Optimizer
        self.optimizer = EvolutionaryOptimizer(
            self.model_names,
            OptimizationConfig(population_size=self.initial_population_size, n_generations=self.initial_n_generations),
            self.storage
        )

        # Trial runner
        self.runner = TrialRunner(self.storage, task=self.task, quick_mode=self.initial_quick_mode)

        # Initialize scheduler
        self.scheduler = ExperimentScheduler(self.baseline_results)

        # Worker thread
        self.worker = HyperoptSearchWorker(self.optimizer, self.runner, self.scheduler)
        self.worker.trial_completed.connect(self.on_trial_completed)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.baseline_completed.connect(self.on_baseline_completed)
        self.worker.experiment_progress.connect(self.on_experiment_progress)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.insight_update.connect(self.on_insight_update)
        self.worker.start()
        self.worker.resume()  # specific fix: ensure worker is not paused

        # Queue baselines if not already run
        if self.scheduler.baseline_results:
            # If baselines were loaded from storage, update the UI
            self.add_log_message("Baselines already exist. Starting optimization...")
            self.on_baseline_completed(self.scheduler.baseline_results) # Pass the full dict
        else:
            self.worker.queue_baseline()
            self.start_btn.setText("⏸ Pause Optimization")
            self.start_btn.setStyleSheet(
                "font-weight: bold; background-color: #f39c12; color: white; padding: 10px; font-size: 14px;"
            )


        # Start update timers after initialization
        self.update_timer.start(2000)  # Update every 2s
        self.status_timer.start(500)  # Update every 500ms

        self.is_initialized = True

    def create_baseline_section(self):
        group = QGroupBox("Baseline Comparison")
        layout = QHBoxLayout(group)

        # Create a table to show baseline results
        self.baseline_table = QTableWidget()
        self.baseline_table.setColumnCount(5)
        self.baseline_table.setHorizontalHeaderLabels([
            'Model', 'Accuracy', 'Perplexity', 'Walltime (s)', 'Parameters (M)'
        ])
        self.baseline_table.setRowCount(len(self.model_names))

        for i, model_name in enumerate(self.model_names):
            self.baseline_table.setItem(i, 0, QTableWidgetItem(model_name))
            self.baseline_table.setItem(i, 1, QTableWidgetItem("--"))
            self.baseline_table.setItem(i, 2, QTableWidgetItem("--"))
            self.baseline_table.setItem(i, 3, QTableWidgetItem("--"))
            self.baseline_table.setItem(i, 4, QTableWidgetItem("--"))

        layout.addWidget(self.baseline_table)

        return group

    def _create_plot_widget(self, title: str, xlabel: str, ylabel: str) -> pg.PlotWidget:
        """Create a standardized plot widget."""
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.setLabel('bottom', xlabel)
        plot_widget.setLabel('left', ylabel)
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        return plot_widget

    def create_pareto_plots(self):
        group = QGroupBox("Pareto Frontier Visualization")
        layout = QHBoxLayout(group)

        # Create standardized plots
        self.plot_acc_ppl = self._create_plot_widget(
            "Accuracy vs Perplexity",
            "Perplexity (lower is better)",
            "Accuracy"
        )

        self.plot_acc_speed = self._create_plot_widget(
            "Accuracy vs Speed",
            "Iteration Time (ms)",
            "Accuracy"
        )

        self.plot_acc_params = self._create_plot_widget(
            "Accuracy vs Parameters",
            "Parameters (M)",
            "Accuracy"
        )

        self.plot_speed_params = self._create_plot_widget(
            "Speed vs Parameters",
            "Parameters (M)",
            "Iteration Time (ms)"
        )

        # Add to layout in 2x2 grid
        grid_widget = QWidget()
        grid_layout = QVBoxLayout(grid_widget)

        # Add plots in 2x2 grid
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.addWidget(self.plot_acc_ppl)
        top_layout.addWidget(self.plot_acc_speed)

        bottom_row = QWidget()
        bottom_layout = QHBoxLayout(bottom_row)
        bottom_layout.addWidget(self.plot_acc_params)
        bottom_layout.addWidget(self.plot_speed_params)

        grid_layout.addWidget(top_row)
        grid_layout.addWidget(bottom_row)

        layout.addWidget(grid_widget)

        # Color map for models
        self.colors = {}
        for i, spec in enumerate(MODEL_REGISTRY):
            self.colors[spec.name] = spec.color

        return group

    def create_status_table(self):
        # This method is deprecated since we now use the experiment table as primary interface
        # Creating a minimal placeholder to avoid errors
        group = QGroupBox("Statistics")
        layout = QVBoxLayout(group)

        # Statistics
        self.stats_label = QLabel("No trials yet")
        self.stats_label.setStyleSheet("padding: 5px; font-size: 11px;")
        layout.addWidget(self.stats_label)

        return group

    def create_best_configs(self):
        # This method is deprecated since we now use the experiment table as primary interface
        # Creating a minimal placeholder to avoid errors
        group = QGroupBox("Best Configurations (See Experiment List)")
        layout = QVBoxLayout(group)

        info_label = QLabel("Best configurations are displayed in the main experiment list.\nSort by accuracy, perplexity, or other metrics to find top performers.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

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

    def on_baseline_completed(self, model_name, results):
        """Update baseline results when baseline completes."""
        self.baseline_results[model_name] = results

        # Update the scheduler's baseline results
        self.scheduler.baseline_results = self.baseline_results

        # Update the baseline table if it exists
        if hasattr(self, 'baseline_table') and self.baseline_table:
            for i, name in enumerate(self.model_names):
                if name == model_name:
                    self.baseline_table.setItem(i, 1, QTableWidgetItem(f"{results.get('accuracy', 0):.3f}"))
                    self.baseline_table.setItem(i, 2, QTableWidgetItem(f"{results.get('perplexity', 0):.2f}"))
                    self.baseline_table.setItem(i, 3, QTableWidgetItem(f"{results.get('walltime', 0):.1f}"))
                    self.baseline_table.setItem(i, 4, QTableWidgetItem(f"{results.get('parameters', 0):.2f}"))
                    break

    def on_experiment_progress(self, trial_id, metrics):
        """Update experiment progress."""
        # Update status table with latest metrics
        self.update_status_table()

    def on_progress_update(self, percentage, phase, message):
        """Update progress bar and phase information."""
        self.overall_progress.setValue(percentage)
        self.overall_progress.setFormat(f"{phase}: {message}")
        self.phase_label.setText(f"Phase: {phase}")

    def on_insight_update(self, insight_message):
        """Update insights panel with new information."""
        # Get current text and append new insight
        current_text = self.insights_text.toHtml()
        new_text = f"<font color='#808090'>{insight_message}</font><br>" + current_text
        # Limit to last 5 lines to prevent overflow
        lines = new_text.split('<br>')
        if len(lines) > 5:
            lines = lines[:5]
        self.insights_text.setHtml('<br>'.join(lines))

    def update_status_info(self):
        """Update status information more frequently for real-time feedback."""
        try:
            # Check if storage is initialized
            if not self.storage or not self.optimizer:
                self.status_label.setText("Waiting for dataset selection...")
                return

            # Get statistics from optimizer
            stats = self.optimizer.get_statistics()

            # Update status label with current status
            if self.worker and hasattr(self.worker, 'paused') and self.worker.paused:
                status_text = "Paused"
            else:
                status_text = "Running"

            # Count completed trials
            all_trials = self.storage.get_all_trials()
            completed_trials = [t for t in all_trials if t.status == 'completed']
            running_trials = [t for t in all_trials if t.status == 'running']

            status_msg = f"[{status_text}] | Completed: {len(completed_trials)} | Running: {len(running_trials)} | Total: {len(all_trials)}"

            # Add generation info
            if 'generation' in stats:
                status_msg += f" | Gen: {stats['generation']}"

            self.status_label.setText(status_msg)

            # Update stats label with more detailed information
            stats_text = f"Generation {stats.get('generation', 0)} | "
            for model_name, model_stats in stats.get('models', {}).items():
                stats_text += f"{model_name.split('(')[0].strip()}: {model_stats.get('completed', 0)}/{model_stats.get('total_trials', 0)} | "
            if hasattr(self, 'stats_label'):
                self.stats_label.setText(stats_text)

        except Exception as e:
            # Don't let errors interrupt the timer
            pass

    def update_visualizations(self):
        """Update all plots and tables."""
        # Check if storage is initialized
        if not self.storage:
            return

        # Get all trials (not just completed)
        all_trials = self.storage.get_all_trials()

        if not all_trials:
            return

        # Update experiment table with all trials
        self.update_experiment_table(all_trials)

        # Update Pareto plots (only completed trials for cleaner visualization)
        completed_trials = [t for t in all_trials if t.status == 'completed']

        if completed_trials:
            # Update Pareto plots
            self.plot_acc_ppl.clear()
            self.plot_acc_speed.clear()
            self.plot_acc_params.clear()
            self.plot_speed_params.clear()

            # Group by model
            trials_by_model = {}
            for trial in completed_trials:
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

        # Update statistics
        stats = self.optimizer.get_statistics()
        stats_text = f"Generation {stats['generation']} | "
        for model_name, model_stats in stats['models'].items():
            stats_text += f"{model_name.split('(')[0].strip()}: {model_stats['completed']}/{model_stats['total_trials']} | "
        if hasattr(self, 'stats_label'):
            self.stats_label.setText(stats_text)

    def update_experiment_table(self, trials):
        """Update the experiment table with the given trials."""
        # Disable sorting temporarily to prevent issues during update
        self.experiment_table.setSortingEnabled(False)

        # Sort trials by trial_id in descending order (most recent first)
        sorted_trials = sorted(trials, key=lambda t: t.trial_id, reverse=True)
        
        # Filter pruned trials if checkbox is not checked
        if hasattr(self, 'show_pruned_cb') and not self.show_pruned_cb.isChecked():
            sorted_trials = [t for t in sorted_trials if t.status.lower() not in ['pruned', 'terminated', 'stopped']]

        self.experiment_table.setRowCount(len(sorted_trials))

        for i, trial in enumerate(sorted_trials):
            self.experiment_table.setItem(i, 0, QTableWidgetItem(str(trial.trial_id)))
            self.experiment_table.setItem(i, 1, QTableWidgetItem(trial.model_name.split('(')[0].strip()))
            self.experiment_table.setItem(i, 2, QTableWidgetItem(trial.status))
            self.experiment_table.setItem(i, 3, QTableWidgetItem(f"{trial.epochs_completed}"))
            self.experiment_table.setItem(i, 4, QTableWidgetItem(f"{trial.accuracy:.4f}" if trial.accuracy else "--"))
            self.experiment_table.setItem(i, 5, QTableWidgetItem(f"{trial.perplexity:.2f}" if trial.perplexity else "--"))
            self.experiment_table.setItem(i, 6, QTableWidgetItem(f"{trial.iteration_time*1000:.1f}" if trial.iteration_time else "--"))
            self.experiment_table.setItem(i, 7, QTableWidgetItem(f"{trial.param_count:.2f}" if trial.param_count else "--"))

            # Calculate and add Accuracy per Parameter
            acc_per_param = "--"
            if trial.accuracy and trial.param_count and trial.param_count > 0:
                acc_per_param = f"{trial.accuracy / trial.param_count:.6f}"
            self.experiment_table.setItem(i, 8, QTableWidgetItem(acc_per_param))

            # Calculate and add Accuracy per Iteration Walltime
            acc_per_time = "--"
            if trial.accuracy and trial.iteration_time and trial.iteration_time > 0:
                acc_per_time = f"{trial.accuracy / trial.iteration_time:.6f}"
            self.experiment_table.setItem(i, 9, QTableWidgetItem(acc_per_time))

            # Highlight pruned trials in red
            if trial.status.lower() in ['pruned', 'terminated', 'stopped']:
                for col in range(10):
                    item = self.experiment_table.item(i, col)
                    if item:
                        item.setBackground(pg.mkColor('#ffcccc'))  # Light red background

        # Re-enable sorting
        self.experiment_table.setSortingEnabled(True)

    def update_status_table(self):
        # This method is deprecated since we now use the experiment table as primary interface
        pass

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

    
    def spawn_trainer_context_menu(self, position):
        """Show context menu for spawning trainer."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        
        index = self.experiment_table.indexAt(position)
        if not index.isValid():
            return
            
        menu = QMenu()
        spawn_action = QAction("🚀 Spawn EqProp Trainer with this config", self)
        spawn_action.triggered.connect(lambda: self.launch_trainer_for_selected_trial())
        menu.addAction(spawn_action)
        menu.exec(self.experiment_table.viewport().mapToGlobal(position))
        
    def launch_trainer_for_selected_trial(self):
        """Launch trainer for the currently selected trial."""
        selected_rows = self.experiment_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        # Get trial ID from the first column of the selected row
        row = selected_rows[0].row()
        trial_id_item = self.experiment_table.item(row, 0)
        if not trial_id_item:
            return
            
        try:
            trial_id = int(trial_id_item.text())
            self.launch_trainer_for_trial(trial_id)
        except ValueError:
            pass

    def launch_trainer_for_trial(self, trial_id):
        """Spawn the eqprop_trainer UI with the specific trial configuration."""
        import subprocess
        import os
        import json
        
        trial = self.storage.get_trial(trial_id)
        if not trial:
            self.status_label.setText(f"❌ Could not find trial {trial_id}")
            return
            
        config = trial.config
        # Ensure model name is in config
        if 'model_name' not in config:
            config['model_name'] = trial.model_name
            
        # Prepare valid JSON string for CLI
        config_json = json.dumps(config)
        
        try:
            cmd = [sys.executable, "-m", "eqprop_trainer", "--config", config_json]
            
            # Launch in background
            process = subprocess.Popen(cmd)
            self.status_label.setText(f"🚀 Launched EqProp Trainer for Trial {trial_id}")
            
        except Exception as e:
            self.status_label.setText(f"❌ Failed to launch EqProp Trainer: {str(e)}")

    def reset_optimization(self):
        """Reset the optimization process and clear all results."""
        reply = QMessageBox.question(
            self,
            'Reset Confirmation',
            'Are you sure you want to reset the optimization? All current results will be lost.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Stop the worker thread
            if self.worker:
                self.worker.stop()

            # Clear the database
            self.storage.clear_all_trials()

            # Reinitialize optimizer and runner
            self.optimizer = EvolutionaryOptimizer(
                self.model_names,
                OptimizationConfig(population_size=self.optimizer.config.population_size,
                                 n_generations=self.optimizer.config.n_generations),
                self.storage
            )

            self.runner = TrialRunner(self.storage, task=self.task, quick_mode=self.runner.quick_mode)

            # Reinitialize scheduler
            self.scheduler = ExperimentScheduler(self.baseline_results)

            # Restart the worker
            self.worker = HyperoptSearchWorker(self.optimizer, self.runner, self.scheduler)
            self.worker.trial_completed.connect(self.on_trial_completed)
            self.worker.status_update.connect(self.on_status_update)
            self.worker.baseline_completed.connect(self.on_baseline_completed)
            self.worker.experiment_progress.connect(self.on_experiment_progress)
            self.worker.progress_update.connect(self.on_progress_update)
            self.worker.insight_update.connect(self.on_insight_update)
            self.worker.start()
            self.worker.pause()  # Start paused

            # Update UI
            self.status_label.setText("Ready - Reset complete. Run baselines first...")
            self.start_btn.setText("▶ Start Optimization")
            self.start_btn.setStyleSheet(
                "font-weight: bold; background-color: #27ae60; color: white; padding: 10px; font-size: 14px;"
            )

            # Clear visualizations
            self.update_visualizations()

            self.status_label.setText("✅ Optimization reset complete. Ready to start.")

    def closeEvent(self, event):
        """Handle application shutdown gracefully."""
        reply = QMessageBox.question(
            self,
            'Shutdown Confirmation',
            'Are you sure you want to quit? Optimization results will be saved.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.status_label.setText("Shutting down... Please wait...")
            QApplication.processEvents()  # Allow UI to update

            # Stop worker thread and wait for it to finish
            if self.worker:
                self.worker.stop()
                self.worker.wait()  # Wait for thread to finish

            # Close storage
            if self.storage:
                self.storage.close()

            # Stop update timers
            if self.update_timer:
                self.update_timer.stop()
            if self.status_timer:
                self.status_timer.stop()

            event.accept()
        else:
            event.ignore()