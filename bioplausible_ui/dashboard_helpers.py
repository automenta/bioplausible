"""
Bioplausible Trainer Dashboard - Helper methods extension

Additional methods for hyperparameters, visualization, and generation.
This file extends the dashboard with new functionality.
"""

import numpy as np
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QGridLayout, QGroupBox

try:
    import pyqtgraph as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False


def _clear_widgets_from_layout(layout: QGridLayout, widgets: dict) -> None:
    """
    Clear existing widgets from layout and dictionary.

    Args:
        layout: Grid layout containing the widgets
        widgets: Dictionary of widgets to clear
    """
    # Remove widgets from layout and schedule for deletion
    for widget in widgets.values():
        if isinstance(widget, (QSpinBox, QDoubleSpinBox, QCheckBox, QLabel)):
            layout.removeWidget(widget)
            widget.setParent(None)  # Remove from parent
            widget.deleteLater()    # Schedule for deletion

    # Clear the dictionary
    widgets.clear()

    # Remove any remaining items in layout
    for i in reversed(range(layout.count())):
        item = layout.itemAt(i)
        if item.widget():
            layout.removeWidget(item.widget())
            item.widget().setParent(None)
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


def update_hyperparams_generic(self, model_name: str, layout: QGridLayout, widgets: dict, group: QGroupBox) -> None:
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

    # Early exit if no model name
    if not model_name:
        group.setVisible(False)
        return

    # Clear existing widgets
    _clear_widgets_from_layout(layout, widgets)

    # Get hyperparameters for this model
    specs = get_hyperparams_for_model(model_name)

    if not specs:
        group.setVisible(False)
        return

    # Create widgets for each hyperparameter
    group.setVisible(True)

    # Batch add widgets to layout for better performance
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


def get_current_hyperparams_generic(widgets: dict) -> Dict[str, Any]:
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
        if isinstance(widget, QSpinBox):
            hyperparams[name] = widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            hyperparams[name] = widget.value()
        elif isinstance(widget, QCheckBox):
            hyperparams[name] = widget.isChecked()
    return hyperparams


def update_weight_visualization_generic(self, weights: Dict[str, np.ndarray]) -> None:
    """
    Update weight visualization heatmaps.

    Args:
        self: Reference to the dashboard instance
        weights: Dictionary mapping layer names to weight arrays
    """
    if not HAS_PG or not weights:
        return

    from .viz_utils import format_weight_for_display, normalize_weights_for_display, get_layer_description

    # Check if we need to create visualization widgets
    if not hasattr(self, 'vis_weight_widgets') or not self.vis_weight_widgets:
        create_weight_viz_widgets_generic(self, weights)
        return  # Widgets created, need to call again to update

    # Update each heatmap with bounds checking
    max_widgets = len(self.vis_weight_widgets)
    for i, (name, W) in enumerate(weights.items()):
        if i >= max_widgets:
            break

        try:
            # Normalize and format
            W_display = format_weight_for_display(W)
            W_norm = normalize_weights_for_display(W_display)

            # Update ImageView
            self.vis_weight_widgets[i].setImage(W_norm.T, levels=(0, 1))
            self.vis_weight_labels[i].setText(get_layer_description(name))
        except Exception:
            # Continue with other widgets even if one fails
            continue


def create_weight_viz_widgets_generic(self, weights: Dict[str, np.ndarray]) -> None:
    """
    Create weight visualization widgets based on model weights.

    Args:
        self: Reference to the dashboard instance
        weights: Dictionary mapping layer names to weight arrays
    """
    if not HAS_PG or not weights:
        return

    from .viz_utils import get_layer_description

    # Clear existing widgets
    _clear_weight_viz_widgets(self)

    # Create widgets for up to 3 weight matrices
    weight_items = list(weights.items())[:3]

    for i, (name, W) in enumerate(weight_items):
        # Label
        label = QLabel(get_layer_description(name))
        label.setStyleSheet("color: #00d4ff; font-weight: bold;")
        self.vis_weights_layout.addWidget(label)

        # ImageView for heatmap
        img_view = pg.ImageView()
        img_view.setFixedHeight(150)
        img_view.ui.histogram.hide()  # Hide histogram
        img_view.ui.roiBtn.hide()     # Hide ROI button
        img_view.ui.menuBtn.hide()    # Hide menu button

        self.vis_weights_layout.addWidget(img_view)

        # Store references
        if not hasattr(self, 'vis_weight_labels'):
            self.vis_weight_labels = []
        if not hasattr(self, 'vis_weight_widgets'):
            self.vis_weight_widgets = []

        self.vis_weight_labels.append(label)
        self.vis_weight_widgets.append(img_view)


def _clear_weight_viz_widgets(self) -> None:
    """
    Clear existing weight visualization widgets.

    Args:
        self: Reference to the dashboard instance
    """
    # Clear existing widgets if they exist
    if hasattr(self, 'vis_weights_layout'):
        for i in reversed(range(self.vis_weights_layout.count())):
            item = self.vis_weights_layout.itemAt(i)
            if item.widget():
                self.vis_weights_layout.removeWidget(item.widget())
                item.widget().setParent(None)
                item.widget().deleteLater()

    # Clear stored references
    if hasattr(self, 'vis_weight_widgets'):
        for widget in self.vis_weight_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self.vis_weight_widgets.clear()

    if hasattr(self, 'vis_weight_labels'):
        for label in self.vis_weight_labels:
            label.setParent(None)
            label.deleteLater()
        self.vis_weight_labels.clear()


def validate_hyperparams(hyperparams: Dict[str, Any], expected_types: Dict[str, type]) -> bool:
    """
    Validate hyperparameters against expected types.

    Args:
        hyperparams: Dictionary of hyperparameter values
        expected_types: Dictionary mapping param names to expected types

    Returns:
        True if all params are valid, False otherwise
    """
    for param_name, expected_type in expected_types.items():
        if param_name in hyperparams:
            if not isinstance(hyperparams[param_name], expected_type):
                return False
    return True

