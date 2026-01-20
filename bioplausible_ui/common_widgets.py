"""
Common UI widgets and utilities for Bioplausible Trainer.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QPushButton, QProgressBar, QLabel, QSlider, QTextEdit,
    QToolBox, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt
from typing import Dict, Any, Callable
import pyqtgraph as pg
from .themes import PLOT_COLORS


def create_control_group(title: str, controls: list, parent=None) -> QGroupBox:
    """Create a group box with labeled controls."""
    group = QGroupBox(title)
    layout = QGridLayout(group)
    for i, (label, widget) in enumerate(controls):
        layout.addWidget(QLabel(label), i, 0)
        layout.addWidget(widget, i, 1)
    return group


def create_plot_widget(title: str, ylabel: str, xlabel: str = 'Epoch', yrange: tuple = None, parent=None):
    """Create a standardized plot widget."""
    plot_widget = pg.PlotWidget()
    plot_widget.setBackground('#0a0a0f')
    plot_widget.setLabel('left', ylabel, color=PLOT_COLORS.get(ylabel.lower().split()[0], '#ffffff'))
    plot_widget.setLabel('bottom', xlabel)
    plot_widget.showGrid(x=True, y=True, alpha=0.2)
    if yrange:
        plot_widget.setYRange(*yrange)
    curve = plot_widget.plot(pen=pg.mkPen(PLOT_COLORS.get(ylabel.lower().split()[0], '#ffffff'), width=2))
    return plot_widget, curve


def create_standard_buttons(start_callback: Callable, stop_callback: Callable, reset_callback: Callable, 
                          clear_callback: Callable, parent=None) -> tuple:
    """Create standard training control buttons."""
    btn_layout = QHBoxLayout()
    
    train_btn = QPushButton("▶ Train")
    train_btn.setObjectName("trainButton")
    train_btn.clicked.connect(start_callback)
    btn_layout.addWidget(train_btn)

    stop_btn = QPushButton("⏹ Stop")
    stop_btn.setObjectName("stopButton")
    stop_btn.setEnabled(False)
    stop_btn.clicked.connect(stop_callback)
    btn_layout.addWidget(stop_btn)

    pause_btn = QPushButton("⏸ Pause")
    pause_btn.setObjectName("resetButton")
    pause_btn.setCheckable(True)
    pause_btn.setEnabled(False)
    btn_layout.addWidget(pause_btn)

    reset_btn = QPushButton("↺ Reset")
    reset_btn.setObjectName("resetButton")
    reset_btn.clicked.connect(reset_callback)
    btn_layout.addWidget(reset_btn)

    clear_btn = QPushButton("🗑️ Clear")
    clear_btn.setObjectName("resetButton")
    clear_btn.clicked.connect(clear_callback)
    btn_layout.addWidget(clear_btn)
    
    return btn_layout, train_btn, stop_btn, pause_btn, reset_btn, clear_btn


def create_hyperparam_group(parent_widget, model_combo_callback: Callable) -> tuple:
    """Create a dynamic hyperparameter group."""
    hyperparam_group = QGroupBox("Dynamic Params")
    hyperparam_layout = QGridLayout(hyperparam_group)
    hyperparam_widgets = {}
    hyperparam_group.setVisible(False)
    model_combo_callback(hyperparam_group)
    return hyperparam_group, hyperparam_layout, hyperparam_widgets


def update_theme_for_plots(widget, theme_colors: Dict[str, str], plot_colors: Dict[str, str]):
    """Update plot colors based on theme."""
    bg = theme_colors.get('background', '#0a0a0f')

    # Update Plots if they exist
    if hasattr(widget, 'loss_plot') and widget.loss_plot:
        widget.loss_plot.setBackground(bg)
        if hasattr(widget, 'loss_curve'):
            widget.loss_curve.setPen(pg.mkPen(plot_colors.get('loss', 'w'), width=2))
        if widget.loss_plot.getAxis('left'):
            widget.loss_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
        if widget.loss_plot.getAxis('bottom'):
            widget.loss_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))

    if hasattr(widget, 'acc_plot') and widget.acc_plot:
        widget.acc_plot.setBackground(bg)
        if hasattr(widget, 'acc_curve'):
            widget.acc_curve.setPen(pg.mkPen(plot_colors.get('accuracy', 'w'), width=2))
        if widget.acc_plot.getAxis('left'):
            widget.acc_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
        if widget.acc_plot.getAxis('bottom'):
            widget.acc_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))

    if hasattr(widget, 'lip_plot') and widget.lip_plot:
        widget.lip_plot.setBackground(bg)
        if hasattr(widget, 'lip_curve'):
            widget.lip_curve.setPen(pg.mkPen(plot_colors.get('lipschitz', 'w'), width=2))
        if widget.lip_plot.getAxis('left'):
            widget.lip_plot.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
        if widget.lip_plot.getAxis('bottom'):
            widget.lip_plot.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))