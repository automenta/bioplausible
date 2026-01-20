"""
Bioplausible Trainer Dashboard - Helper methods extension

Additional methods for hyperparameters, visualization, and generation.
This file extends the dashboard with new functionality.
"""

import numpy as np
from typing import Dict
from PyQt6.QtWidgets import QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QGridLayout, QGroupBox

try:
    import pyqtgraph as pg
    HAS_PG = True
except:
    HAS_PG = False


def _clear_widgets_from_layout(layout: QGridLayout, widgets: dict):
    """
    Clear existing widgets from layout and dictionary.

    Args:
        layout: Grid layout containing the widgets
        widgets: Dictionary of widgets to clear
    """
    for widget in widgets.values():
        if isinstance(widget, (QSpinBox, QDoubleSpinBox, QCheckBox, QLabel)):
            widget.deleteLater()
    widgets.clear()

    # Remove all items from layout
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()


def _create_widget_for_spec(spec: 'HyperparamSpec'):
    """
    Create a widget based on the hyperparameter specification.

    Args:
        spec: Hyperparameter specification object

    Returns:
        Created widget or None if type is unsupported
    """
    if spec.type == 'int':
        widget = QSpinBox()
        widget.setRange(spec.min_val or 0, spec.max_val or 1000)
        widget.setValue(spec.default)
        if spec.step:
            widget.setSingleStep(spec.step)
    elif spec.type == 'float':
        widget = QDoubleSpinBox()
        widget.setRange(spec.min_val or 0.0, spec.max_val or 10.0)
        widget.setValue(spec.default)
        widget.setDecimals(3)
        if spec.step:
            widget.setSingleStep(spec.step)
    elif spec.type == 'bool':
        widget = QCheckBox()
        widget.setChecked(spec.default)
    else:
        return None

    # Tooltip
    if spec.description:
        widget.setToolTip(spec.description)

    return widget


def update_hyperparams_generic(self, model_name: str, layout: QGridLayout, widgets: dict, group: QGroupBox):
    """
    Generic method to update hyperparameter widgets.

    Args:
        self: Reference to the dashboard instance
        model_name: Name of the model to get hyperparameters for
        layout: Layout to add widgets to
        widgets: Dictionary to store widget references
        group: Group box to show/hide
    """
    from .hyperparams import get_hyperparams_for_model

    # Clear existing widgets
    _clear_widgets_from_layout(layout, widgets)

    # Get hyperparameters for this model
    specs = get_hyperparams_for_model(model_name)

    if not specs:
        group.setVisible(False)
        return

    # Create widgets for each hyperparameter
    group.setVisible(True)
    for i, spec in enumerate(specs):
        # Label
        label = QLabel(f"{spec.label}:")
        layout.addWidget(label, i, 0)
        widgets[f"{spec.name}_label"] = label

        # Widget based on type
        widget = _create_widget_for_spec(spec)
        if widget is None:
            continue

        layout.addWidget(widget, i, 1)
        widgets[spec.name] = widget


def get_current_hyperparams_generic(widgets: dict) -> dict:
    """
    Extract current values from hyperparameter widgets.

    Args:
        widgets: Dictionary of hyperparameter widgets

    Returns:
        Dictionary mapping parameter names to current values
    """
    hyperparams = {}
    for name, widget in widgets.items():
        if name.endswith('_label'):
            continue
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            hyperparams[name] = widget.value()
        elif isinstance(widget, QCheckBox):
            hyperparams[name] = widget.isChecked()
    return hyperparams


def update_weight_visualization_generic(self, weights: Dict[str, np.ndarray]):
    """
    Update weight visualization heatmaps.

    Args:
        self: Reference to the dashboard instance
        weights: Dictionary mapping layer names to weight arrays
    """
    if not HAS_PG:
        return

    from .viz_utils import format_weight_for_display, normalize_weights_for_display, get_layer_description

    # Use vision weights container (same pattern for both tabs)
    if not hasattr(self, 'vis_weight_widgets') or not self.vis_weight_widgets:
        # Create weight visualization widgets dynamically
        create_weight_viz_widgets_generic(self, weights)

    # Update each heatmap
    for i, (name, W) in enumerate(weights.items()):
        if i >= len(self.vis_weight_widgets):
            break

        # Normalize and format
        W_display = format_weight_for_display(W)
        W_norm = normalize_weights_for_display(W_display)

        # Update ImageView
        try:
            self.vis_weight_widgets[i].setImage(W_norm.T, levels=(0, 1))
            self.vis_weight_labels[i].setText(get_layer_description(name))
        except Exception:
            pass


def create_weight_viz_widgets_generic(self, weights: Dict[str, np.ndarray]):
    """
    Create weight visualization widgets based on model weights.

    Args:
        self: Reference to the dashboard instance
        weights: Dictionary mapping layer names to weight arrays
    """
    if not HAS_PG:
        return

    from .viz_utils import get_layer_description

    # Clear existing
    while self.vis_weights_layout.count():
        item = self.vis_weights_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

    self.vis_weight_widgets = []
    self.vis_weight_labels = []

    # Create a widget for each weight matrix
    for name, W in list(weights.items())[:3]:  # Max 3 visualizations
        # Label
        label = QLabel(get_layer_description(name))
        label.setStyleSheet("color: #00d4ff; font-weight: bold;")
        self.vis_weights_layout.addWidget(label)
        self.vis_weight_labels.append(label)

        # ImageView for heatmap
        img_view = pg.ImageView()
        img_view.setFixedHeight(150)
        img_view.ui.histogram.hide()  # Hide histogram
        img_view.ui.roiBtn.hide()     # Hide ROI button
        img_view.ui.menuBtn.hide()    # Hide menu button

        self.vis_weights_layout.addWidget(img_view)
        self.vis_weight_widgets.append(img_view)


