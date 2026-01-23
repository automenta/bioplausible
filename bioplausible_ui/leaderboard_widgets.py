"""
Leaderboard Custom Widgets

Reusable UI components for the leaderboard redesign.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, pyqtProperty
from PyQt6.QtGui import QColor, QFont


class SummaryCard(QFrame):
    """Card widget displaying a key metric."""
    
    def __init__(self, title: str, value: str, subtitle: str, icon: str, color: str):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            SummaryCard {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b,
                    stop:1 #0f172a
                );
                border: 1px solid #475569;
                border-radius: 12px;
                padding: 20px;
            }}
            SummaryCard:hover {{
                border: 1px solid {color};
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #334155,
                    stop:1 #1e293b
                );
            }}
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Icon and title row
        header = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"font-size: 32px; color: {color};")
        header.addWidget(icon_label)
        header.addStretch()
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase;")
        layout.addLayout(header)
        layout.addWidget(title_label)
        
        # Value
        value_label = QLabel(value)
        value_label.setStyleSheet(f"font-size: 36px; font-weight: bold; color: {color};")
        layout.addWidget(value_label)
        
        # Subtitle
        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("font-size: 13px; color: #cbd5e1;")
        layout.addWidget(subtitle_label)
        
        self.setFixedHeight(160)


class ExpandableTrialCard(QFrame):
    """Collapsible card showing trial details."""
    
    def __init__(self, trial_data: dict, rank: int, is_pareto: bool = False):
        super().__init__()
        self.trial_data = trial_data
        self.rank = rank
        self.is_pareto = is_pareto
        self.expanded = False
        
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the card UI."""
        border_color = "#9333ea" if self.is_pareto else "#475569"
        
        self.setStyleSheet(f"""
            ExpandableTrialCard {{
                background-color: #1e293b;
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 12px;
                margin: 4px 0;
            }}
            ExpandableTrialCard:hover {{
                background-color: #334155;
                border-color: #9333ea;
            }}
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(8)
        
        # Header (always visible)
        self.create_header()
        
        # Details (expandable)
        self.details_widget = QWidget()
        self.details_layout = QVBoxLayout(self.details_widget)
        self.create_details()
        self.details_widget.setVisible(False)
        self.layout.addWidget(self.details_widget)
        
        # Make clickable
        self.mousePressEvent = lambda e: self.toggle_expand()
    
    def create_header(self):
        """Create the always-visible header."""
        header = QHBoxLayout()
        
        # Rank with medal
        medal = ""
        medal_color = "#e2e8f0"
        if self.rank == 1:
            medal = "🥇"
            medal_color = "#fbbf24"
        elif self.rank == 2:
            medal = "🥈"
            medal_color = "#94a3b8"
        elif self.rank == 3:
            medal = "🥉"
            medal_color = "#fb923c"
        
        rank_text = f"{medal} #{self.rank}" if medal else f"#{self.rank}"
        rank_label = QLabel(rank_text)
        rank_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {medal_color}; min-width: 60px;")
        header.addWidget(rank_label)
        
        # Model name
        model_name = self.trial_data['model_name']
        if self.is_pareto:
            model_name += " ⭐"
        model_label = QLabel(model_name)
        model_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #e2e8f0;")
        header.addWidget(model_label)
        
        header.addStretch()
        
        # Metrics
        acc = self.trial_data['accuracy'] * 100
        params = self.trial_data['param_count']
        time = self.trial_data['iteration_time']
        
        metrics_html = f"""
            <span style='color: #10b981; font-weight: bold;'>{acc:.2f}%</span>
            <span style='color: #64748b; margin: 0 8px;'>•</span>
            <span style='color: #06b6d4;'>{params:.2f}M</span>
            <span style='color: #64748b; margin: 0 8px;'>•</span>
            <span style='color: #f59e0b;'>{time:.4f}s</span>
        """
        metrics_label = QLabel(metrics_html)
        metrics_label.setStyleSheet("font-size: 14px;")
        header.addWidget(metrics_label)
        
        # Expand indicator
        self.expand_label = QLabel("▼")
        self.expand_label.setStyleSheet("font-size: 12px; color: #94a3b8;")
        header.addWidget(self.expand_label)
        
        self.layout.addLayout(header)
    
    def create_details(self):
        """Create the expandable details section."""
        # Trial ID
        trial_label = QLabel(f"Trial ID: {self.trial_data['trial_id']}")
        trial_label.setStyleSheet("font-size: 12px; color: #94a3b8; margin-top: 8px;")
        self.details_layout.addWidget(trial_label)
        
        # Hyperparameters
        if 'config' in self.trial_data and self.trial_data['config']:
            params_title = QLabel("Hyperparameters:")
            params_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #e2e8f0; margin-top: 12px;")
            self.details_layout.addWidget(params_title)
            
            config = self.trial_data['config']
            for key, value in sorted(config.items()):
                if key != 'epochs':  # Skip epochs
                    # Format value
                    if isinstance(value, float):
                        value_str = f"{value:.6f}" if value < 0.01 else f"{value:.4f}"
                    else:
                        value_str = str(value)
                    
                    param_html = f"""
                        <span style='color: #a855f7; font-weight: 600;'>{key}:</span>
                        <span style='color: #cbd5e1;'> {value_str}</span>
                    """
                    param_label = QLabel(param_html)
                    param_label.setStyleSheet("font-size: 12px; margin-left: 16px;")
                    self.details_layout.addWidget(param_label)
    
    def toggle_expand(self):
        """Toggle the expanded state."""
        self.expanded = not self.expanded
        self.details_widget.setVisible(self.expanded)
        self.expand_label.setText("▲" if self.expanded else "▼")
        
        # Animate height change
        self.animate_height()
    
    def animate_height(self):
        """Animate the height transition."""
        # Note: Simple toggle for now, can add QPropertyAnimation for smoother effect
        pass


class InsightWidget(QFrame):
    """Widget displaying an auto-generated insight."""
    
    def __init__(self, insight_text: str, insight_type: str = "info"):
        super().__init__()
        
        # Icon and color based on type
        icons = {
            "info": "💡",
            "success": "✅",
            "warning": "⚠️",
            "tip": "🎯",
        }
        colors = {
            "info": "#3b82f6",
            "success": "#10b981",
            "warning": "#f59e0b",
            "tip": "#9333ea",
        }
        
        icon = icons.get(insight_type, "💡")
        color = colors.get(insight_type, "#3b82f6")
        
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            InsightWidget {{
                background-color: #1e293b;
                border-left: 4px solid {color};
                border-radius: 6px;
                padding: 12px 16px;
                margin: 4px 0;
            }}
        """)
        
        layout = QHBoxLayout(self)
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"font-size: 20px; color: {color};")
        layout.addWidget(icon_label)
        
        text_label = QLabel(insight_text)
        text_label.setStyleSheet("font-size: 13px; color: #e2e8f0;")
        text_label.setWordWrap(True)
        layout.addWidget(text_label)
