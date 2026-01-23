"""
Leaderboard Window - Redesigned Dashboard

Rich, visual dashboard for benchmark results with summary cards,
optimization progress chart, expandable trial cards, and auto-generated insights.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QPushButton, QComboBox, QApplication, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
import pyqtgraph as pg
import numpy as np

from bioplausible_ui.leaderboard_data import (
    load_trials, load_trials_timeseries, compute_pareto_frontier, compute_statistics
)
from bioplausible_ui.leaderboard_widgets import (
    SummaryCard, ExpandableTrialCard, InsightWidget
)
from bioplausible_ui.leaderboard_insights import generate_insights


class LeaderboardWindow(QMainWindow):
    """Modern dashboard-style leaderboard with automatic insights."""
    
    def __init__(self, db_path="examples/shallow_benchmark.db"):
        super().__init__()
        self.db_path = db_path
        self.trials = []
        self.pareto_ids = []
        self.statistics = {}
        self.current_filter = "all"
        
        self.setWindowTitle("🧬 Bioplausible Leaderboard")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Apply dark theme
        self.apply_theme()
        
        self.init_ui()
        
        # Auto-refresh timer (5 seconds)
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(5000)
        
        # Initial load
        self.refresh_data()
    
    def apply_theme(self):
        """Apply global dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
            }
            QPushButton {
                background-color: #9333ea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #a855f7;
            }
            QPushButton:pressed {
                background-color: #7e22ce;
            }
            QComboBox {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #475569;
                padding: 8px 12px;
                border-radius: 6px;
                min-width: 150px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
    
    def init_ui(self):
        """Initialize the dashboard UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("🧬 Bioplausible Leaderboard")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #a855f7;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Toolbar
        self.status_label = QLabel("● Loading...")
        self.status_label.setStyleSheet("font-size: 13px; color: #10b981; margin-right: 16px;")
        header_layout.addWidget(self.status_label)
        
        header_layout.addWidget(QLabel("Filter:"))
        self.model_filter = QComboBox()
        self.model_filter.addItem("All Models")
        self.model_filter.currentTextChanged.connect(self.filter_changed)
        header_layout.addWidget(self.model_filter)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        header_layout.addWidget(self.refresh_btn)
        
        main_layout.addLayout(header_layout)
        
        # Create scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        self.content_layout = QVBoxLayout(scroll_content)
        self.content_layout.setSpacing(24)
        
        # Summary cards container
        self.summary_container = QHBoxLayout()
        self.summary_container.setSpacing(16)
        self.content_layout.addLayout(self.summary_container)
        
        # Progress chart container
        self.progress_chart = pg.PlotWidget(title="<span style='color: #e2e8f0; font-size: 18px;'>📈 Optimization Progress</span>")
        self.progress_chart.setBackground('#1e293b')
        self.progress_chart.setLabel('left', 'Accuracy', color='#e2e8f0', size='12pt')
        self.progress_chart.setLabel('bottom', 'Trial Number', color='#e2e8f0', size='12pt')
        self.progress_chart.showGrid(x=True, y=True, alpha=0.2)
        self.progress_chart.setMinimumHeight(300)
        self.content_layout.addWidget(self.progress_chart)
        
        # Top trials section
        trials_header = QLabel("🎯 Top Trials")
        trials_header.setStyleSheet("font-size: 20px; font-weight: bold; color: #e2e8f0; margin-top: 16px;")
        self.content_layout.addWidget(trials_header)
        
        self.trials_container = QVBoxLayout()
        self.trials_container.setSpacing(8)
        self.content_layout.addLayout(self.trials_container)
        
        # Insights section
        insights_header = QLabel("💡 Insights & Recommendations")
        insights_header.setStyleSheet("font-size: 20px; font-weight: bold; color: #e2e8f0; margin-top: 24px;")
        self.content_layout.addWidget(insights_header)
        
        self.insights_container = QVBoxLayout()
        self.insights_container.setSpacing(8)
        self.content_layout.addLayout(self.insights_container)
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
    
    def refresh_data(self):
        """Refresh data from database."""
        try:
            self.trials = load_trials(self.db_path)
            
            if not self.trials:
                self.status_label.setText("● No trials found")
                self.status_label.setStyleSheet("color: #f59e0b;")
                return
            
            self.pareto_ids = compute_pareto_frontier(self.trials)
            self.statistics = compute_statistics(self.trials)
            
            # Update model filter
            models = list(set(t['model_name'] for t in self.trials))
            current = self.model_filter.currentText()
            self.model_filter.clear()
            self.model_filter.addItem("All Models")
            self.model_filter.addItems(sorted(models))
            if current in models or current == "All Models":
                self.model_filter.setCurrentText(current)
            
            self.update_ui()
            self.status_label.setText(f"● Live ({len(self.trials)} trials)")
            self.status_label.setStyleSheet("color: #10b981;")
            
        except Exception as e:
            self.status_label.setText(f"● Error: {str(e)}")
            self.status_label.setStyleSheet("color: #ef4444;")
            import traceback
            traceback.print_exc()
    
    def filter_changed(self, text):
        """Handle model filter change."""
        self.current_filter = text
        self.update_ui()
    
    def get_filtered_trials(self):
        """Get trials matching current filter."""
        if self.current_filter == "All Models":
            return self.trials
        return [t for t in self.trials if t['model_name'] == self.current_filter]
    
    def update_ui(self):
        """Update all UI components."""
        trials = self.get_filtered_trials()
        
        if not trials:
            return
        
        # Update summary cards
        self.update_summary_cards(trials)
        
        # Update progress chart
        self.update_progress_chart()
        
        # Update top trials
        self.update_top_trials(trials)
        
        # Update insights
        self.update_insights(trials)
    
    def update_summary_cards(self, trials):
        """Update the summary cards at the top."""
        # Clear existing cards
        while self.summary_container.count():
            child = self.summary_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # 1. Winner card (best accuracy)
        best_trial = max(trials, key=lambda t: t['accuracy'])
        winner_card = SummaryCard(
            title="WINNER",
            value=f"{best_trial['accuracy']*100:.1f}%",
            subtitle=best_trial['model_name'],
            icon="🏆",
            color="#fbbf24"
        )
        self.summary_container.addWidget(winner_card)
        
        # 2. Pareto card
        pareto_count = len([t for t in trials if t['trial_id'] in self.pareto_ids])
        pareto_card = SummaryCard(
            title="PARETO",
             value=str(pareto_count),
            subtitle="Optimal trials",
            icon="📊",
            color="#9333ea"
        )
        self.summary_container.addWidget(pareto_card)
        
        # 3. Fastest card
        fastest_trial = min(trials, key=lambda t: t['iteration_time'])
        fastest_card = SummaryCard(
            title="FASTEST",
            value=f"{fastest_trial['iteration_time']*1000:.2f}ms",
            subtitle=fastest_trial['model_name'],
            icon="⚡",
            color="#f59e0b"
        )
        self.summary_container.addWidget(fastest_card)
        
        # 4. Smallest card (achieving 80%+ accuracy)
        efficient_trials = [t for t in trials if t['accuracy'] >= 0.80]
        if efficient_trials:
            smallest_trial = min(efficient_trials, key=lambda t: t['param_count'])
            smallest_card = SummaryCard(
                title="SMALLEST",
                value=f"{smallest_trial['param_count']:.2f}M",
                subtitle=f"{smallest_trial['model_name']} @ {smallest_trial['accuracy']*100:.1f}%",
                icon="🎯",
                color="#06b6d4"
            )
            self.summary_container.addWidget(smallest_card)
    
    def update_progress_chart(self):
        """Update the optimization progress line chart."""
        self.progress_chart.clear()
        
        try:
            timeseries = load_trials_timeseries(self.db_path)
            
            # Colors for different models
            colors = [
                (147, 51, 234),   # Purple
                (6, 182, 212),    # Cyan
                (16, 185, 129),   # Green
                (245, 158, 11),   # Orange
                (239, 68, 68),    # Red
            ]
            
            for idx, (model, model_trials) in enumerate(timeseries.items()):
                if self.current_filter != "All Models" and model != self.current_filter:
                    continue
                
                color = colors[idx % len(colors)]
                
                x = list(range(len(model_trials)))
                y = [t['accuracy'] for t in model_trials]
                
 # Running best (show optimization progress)
                running_best = []
                best_so_far = 0
                for acc in y:
                    best_so_far = max(best_so_far, acc)
                    running_best.append(best_so_far)
                
                # Plot running best as line
                self.progress_chart.plot(
                    x, running_best,
                    pen=pg.mkPen(color=color, width=2),
                    name=model
                )
                
                # Plot actual trials as scatter
                scatter = pg.ScatterPlotItem(
                    x=x, y=y,
                    size=8,
                    pen=pg.mkPen(color=color, width=1),
                    brush=pg.mkBrush(*color, 100)
                )
                self.progress_chart.addItem(scatter)
            
            self.progress_chart.addLegend()
            
        except Exception as e:
            print(f"Error updating progress chart: {e}")
            import traceback
            traceback.print_exc()
    
    def update_top_trials(self, trials):
        """Update the top trials expandable cards."""
        # Clear existing cards
        while self.trials_container.count():
            child = self.trials_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Sort by accuracy and show top 10
        sorted_trials = sorted(trials, key=lambda t: t['accuracy'], reverse=True)
        top_trials = sorted_trials[:min(10, len(sorted_trials))]
        
        for rank, trial in enumerate(top_trials, 1):
            is_pareto = trial['trial_id'] in self.pareto_ids
            card = ExpandableTrialCard(trial, rank, is_pareto)
            self.trials_container.addWidget(card)
    
    def update_insights(self, trials):
        """Update the insights section."""
        # Clear existing insights
        while self.insights_container.count():
            child = self.insights_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Generate insights
        insights = generate_insights(trials, self.pareto_ids)
        
        for insight_text, insight_type in insights:
            widget = InsightWidget(insight_text, insight_type)
            self.insights_container.addWidget(widget)
        
        if not insights:
            no_insights = QLabel("No insights available yet. Run more trials!")
            no_insights.setStyleSheet("color: #94a3b8; font-style: italic; padding: 20px;")
            self.insights_container.addWidget(no_insights)


def main():
    """Launch leaderboard window."""
    app = QApplication(sys.argv)
    
    # Get db path from args
    db_path = "examples/shallow_benchmark.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    window = LeaderboardWindow(db_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
