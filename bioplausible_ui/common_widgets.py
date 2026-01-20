"""
Common UI widgets and utilities for Bioplausible Trainer.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QPushButton, QProgressBar, QLabel, QSlider, QTextEdit,
    QToolBox, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt
from typing import Dict, Any, Callable, Optional, Tuple, Union
import pyqtgraph as pg
from .themes import PLOT_COLORS


class ControlGroupFactory:
    """Factory class for creating standardized control groups."""

    @staticmethod
    def create_control_group(title: str, controls: list, parent: Optional[QWidget] = None) -> QGroupBox:
        """
        Create a group box with labeled controls.

        Args:
            title: Title for the group box
            controls: List of tuples containing (label, widget) pairs
            parent: Parent widget

        Returns:
            QGroupBox with the specified controls
        """
        group = QGroupBox(title)
        layout = QGridLayout(group)

        for i, (label, widget) in enumerate(controls):
            if isinstance(label, str):
                layout.addWidget(QLabel(label), i, 0)
            else:
                layout.addWidget(label, i, 0)  # Allow custom widgets as labels
            layout.addWidget(widget, i, 1)

        return group


class PlotWidgetFactory:
    """Factory class for creating standardized plot widgets."""

    @staticmethod
    def create_plot_widget(
        title: str,
        ylabel: str,
        xlabel: str = 'Epoch',
        yrange: Optional[Tuple[float, float]] = None,
        parent: Optional[QWidget] = None,
        theme_colors: Optional[Dict[str, str]] = None
    ) -> Tuple[pg.PlotWidget, pg.PlotDataItem]:
        """
        Create a standardized plot widget.

        Args:
            title: Title for the plot
            ylabel: Label for the y-axis
            xlabel: Label for the x-axis (default: 'Epoch')
            yrange: Range for the y-axis (optional)
            parent: Parent widget
            theme_colors: Theme colors to use (optional)

        Returns:
            Tuple of (plot_widget, curve)
        """
        if theme_colors is None:
            from .themes import BASE_COLORS
            theme_colors = BASE_COLORS['dark']

        plot_widget = pg.PlotWidget()
        plot_widget.setBackground(theme_colors.get('background', '#0a0a0f'))
        plot_widget.setLabel('left', ylabel, color=theme_colors.get('text_accent', '#00ffff'))
        plot_widget.setLabel('bottom', xlabel)
        plot_widget.showGrid(x=True, y=True, alpha=0.2)

        # Determine color based on ylabel
        color_key = ylabel.lower().split()[0]
        plot_color = PLOT_COLORS.get(color_key, theme_colors.get('text_accent', '#00ffff'))

        if yrange:
            plot_widget.setYRange(*yrange)
        curve = plot_widget.plot(pen=pg.mkPen(plot_color, width=2))
        return plot_widget, curve


class ButtonFactory:
    """Factory class for creating standardized buttons."""

    @staticmethod
    def create_standard_buttons(
        start_callback: Callable,
        stop_callback: Callable,
        reset_callback: Callable,
        clear_callback: Callable,
        parent: Optional[QWidget] = None
    ) -> Tuple[QHBoxLayout, QPushButton, QPushButton, QPushButton, QPushButton, QPushButton]:
        """
        Create standard training control buttons.

        Args:
            start_callback: Callback for start button
            stop_callback: Callback for stop button
            reset_callback: Callback for reset button
            clear_callback: Callback for clear button
            parent: Parent widget

        Returns:
            Tuple of (layout, train_btn, stop_btn, pause_btn, reset_btn, clear_btn)
        """
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
        pause_btn.setObjectName("pauseButton")
        pause_btn.setCheckable(True)
        pause_btn.setEnabled(False)
        btn_layout.addWidget(pause_btn)

        reset_btn = QPushButton("↺ Reset")
        reset_btn.setObjectName("resetButton")
        reset_btn.clicked.connect(reset_callback)
        btn_layout.addWidget(reset_btn)

        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.setObjectName("clearButton")
        clear_btn.clicked.connect(clear_callback)
        btn_layout.addWidget(clear_btn)

        return btn_layout, train_btn, stop_btn, pause_btn, reset_btn, clear_btn


class HyperparamWidgetFactory:
    """Factory class for creating dynamic hyperparameter widgets."""

    @staticmethod
    def create_hyperparam_group(
        parent_widget: QWidget,
        model_combo_callback: Callable
    ) -> Tuple[QGroupBox, QGridLayout, dict]:
        """
        Create a dynamic hyperparameter group.

        Args:
            parent_widget: Parent widget
            model_combo_callback: Callback function for model combo

        Returns:
            Tuple of (group, layout, widgets_dict)
        """
        hyperparam_group = QGroupBox("Dynamic Parameters")
        hyperparam_layout = QGridLayout(hyperparam_group)
        hyperparam_widgets = {}
        hyperparam_group.setVisible(False)

        # Connect the callback to the group
        if hasattr(parent_widget, 'model_combo'):
            parent_widget.model_combo.currentTextChanged.connect(
                lambda text: model_combo_callback(text)
            )

        return hyperparam_group, hyperparam_layout, hyperparam_widgets


def update_theme_for_plots(widget, theme_colors: Dict[str, str], plot_colors: Dict[str, str]):
    """
    Update plot colors based on theme.

    Args:
        widget: Widget containing plot elements
        theme_colors: Dictionary of theme colors
        plot_colors: Dictionary of plot colors
    """
    bg = theme_colors.get('background', '#0a0a0f')

    # Update Plots if they exist
    plot_attrs = [
        ('loss_plot', 'loss_curve', 'loss'),
        ('acc_plot', 'acc_curve', 'accuracy'),
        ('lip_plot', 'lip_curve', 'lipschitz'),
        ('perplexity_plot', 'perplexity_curve', 'perplexity'),
        ('memory_plot', 'memory_curve', 'memory'),
        ('gradient_plot', 'gradient_curve', 'gradient'),
        ('backprop_plot', 'backprop_curve', 'backprop'),
        ('eqprop_plot', 'eqprop_curve', 'eqprop')
    ]

    for plot_attr, curve_attr, color_key in plot_attrs:
        if hasattr(widget, plot_attr):
            plot_widget = getattr(widget, plot_attr)
            if plot_widget:
                plot_widget.setBackground(bg)

                if hasattr(widget, curve_attr):
                    curve = getattr(widget, curve_attr)
                    curve.setPen(pg.mkPen(plot_colors.get(color_key, 'w'), width=2))

                # Update axis colors
                if plot_widget.getAxis('left'):
                    plot_widget.getAxis('left').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))
                if plot_widget.getAxis('bottom'):
                    plot_widget.getAxis('bottom').setPen(pg.mkPen(theme_colors.get('text_secondary', 'w')))


def create_control_group(title: str, controls: list, parent=None) -> QGroupBox:
    """Create a group box with labeled controls."""
    return ControlGroupFactory.create_control_group(title, controls, parent)


def create_plot_widget(title: str, ylabel: str, xlabel: str = 'Epoch', yrange: tuple = None,
                      parent=None, theme_colors: Optional[Dict[str, str]] = None):
    """Create a standardized plot widget."""
    return PlotWidgetFactory.create_plot_widget(title, ylabel, xlabel, yrange, parent, theme_colors)


def create_standard_buttons(start_callback: Callable, stop_callback: Callable, reset_callback: Callable,
                          clear_callback: Callable, parent=None) -> tuple:
    """Create standard training control buttons."""
    return ButtonFactory.create_standard_buttons(start_callback, stop_callback, reset_callback, clear_callback, parent)


def create_hyperparam_group(parent_widget, model_combo_callback: Callable) -> tuple:
    """Create a dynamic hyperparameter group."""
    return HyperparamWidgetFactory.create_hyperparam_group(parent_widget, model_combo_callback)


def create_numeric_widget(widget_type: str, min_val: Union[int, float], max_val: Union[int, float],
                         default_val: Union[int, float], step: Optional[Union[int, float]] = None,
                         tooltip: Optional[str] = None) -> Union[QSpinBox, QDoubleSpinBox]:
    """
    Create a numeric input widget (spinbox or doublespinbox).

    Args:
        widget_type: 'int' for QSpinBox or 'float' for QDoubleSpinBox
        min_val: Minimum value
        max_val: Maximum value
        default_val: Default value
        step: Step increment (optional)
        tooltip: Tooltip text (optional)

    Returns:
        QSpinBox or QDoubleSpinBox widget
    """
    if widget_type == 'int':
        widget = QSpinBox()
        widget.setRange(min_val, max_val)
        widget.setValue(default_val)
        if step:
            widget.setSingleStep(step)
    else:  # float
        widget = QDoubleSpinBox()
        widget.setRange(min_val, max_val)
        widget.setValue(default_val)
        widget.setDecimals(3)
        if step:
            widget.setSingleStep(step)

    if tooltip:
        widget.setToolTip(tooltip)

    return widget


def create_checkbox(label: str, checked: bool = False, tooltip: Optional[str] = None) -> Tuple[QCheckBox, QLabel]:
    """
    Create a checkbox with associated label.

    Args:
        label: Label text
        checked: Initial checked state
        tooltip: Tooltip text (optional)

    Returns:
        Tuple of (checkbox, label)
    """
    checkbox = QCheckBox()
    checkbox.setChecked(checked)

    label_widget = QLabel(label)

    if tooltip:
        checkbox.setToolTip(tooltip)
        label_widget.setToolTip(tooltip)

    return checkbox, label_widget