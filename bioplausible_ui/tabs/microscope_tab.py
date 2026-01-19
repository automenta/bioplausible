
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox,
    QGroupBox, QPushButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from bioplausible.models.looped_mlp import LoopedMLP
from bioplausible.models.conv_eqprop import ConvEqProp
from bioplausible_ui.themes import PLOT_COLORS

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
import torch

class MicroscopeTab(QWidget):
    """Dynamics Visualization Tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None # Reference to current model
        self._setup_ui()

    def _create_control_group(self, title, controls):
        from PyQt6.QtWidgets import QGridLayout
        group = QGroupBox(title)
        layout = QGridLayout(group)
        for i, (label, widget) in enumerate(controls):
            layout.addWidget(QLabel(label), i, 0)
            layout.addWidget(widget, i, 1)
        return group

    def _create_plot_widget(self, title, ylabel, xlabel='Step', yrange=None):
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('#0a0a0f')
        plot_widget.setLabel('left', ylabel, color=PLOT_COLORS.get(ylabel.lower().split()[0], '#ffffff'))
        plot_widget.setLabel('bottom', xlabel)
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        if yrange:
            plot_widget.setYRange(*yrange)
        curve = plot_widget.plot(pen=pg.mkPen(PLOT_COLORS.get(ylabel.lower().split()[0], '#ffffff'), width=2))
        return plot_widget, curve

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(15)

        # Left panel
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)

        # Model Selection
        self.micro_model_combo = QComboBox()
        self.micro_model_combo.addItems([
            "LoopedMLP (Default)",
            "LoopedMLP (Deep)",
            "ConvEqProp (MNIST)",
        ])

        self.micro_steps_spin = QSpinBox()
        self.micro_steps_spin.setRange(10, 200)
        self.micro_steps_spin.setValue(50)

        model_controls = [
            ("Model:", self.micro_model_combo),
            ("Equilibrium Steps:", self.micro_steps_spin),
        ]
        model_group = self._create_control_group("🔬 Setup", model_controls)
        left_panel.addWidget(model_group)

        # Analysis Controls
        self.micro_run_btn = QPushButton("▶ Run Analysis")
        self.micro_run_btn.clicked.connect(self._run_microscope_analysis)
        left_panel.addWidget(self.micro_run_btn)

        self.micro_capture_btn = QPushButton("📸 One-Click Capture")
        self.micro_capture_btn.setToolTip("Auto-configure steps and run on current model immediately")
        self.micro_capture_btn.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
        self.micro_capture_btn.clicked.connect(self._one_click_capture)
        left_panel.addWidget(self.micro_capture_btn)

        # Info
        info_label = QLabel(
            "Visualizes the settling process of the network.\n"
            "Checks if the network converges to a fixed point (L < 1)."
        )
        info_label.setStyleSheet("color: #808090; font-style: italic;")
        info_label.setWordWrap(True)
        left_panel.addWidget(info_label)
        left_panel.addStretch()

        # Right panel: Plots
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)

        if HAS_PYQTGRAPH:
            metrics_group = QGroupBox("📈 Dynamics")
            metrics_layout = QVBoxLayout(metrics_group)
            self.micro_conv_plot, self.micro_conv_curve = self._create_plot_widget(
                "Convergence (||Δh||)", "Delta Norm", xlabel="Step"
            )
            self.micro_conv_plot.setLogMode(x=False, y=True)
            metrics_layout.addWidget(self.micro_conv_plot)
            self.micro_act_plot, self.micro_act_curve = self._create_plot_widget(
                "Mean Activity", "Activity", xlabel="Step"
            )
            metrics_layout.addWidget(self.micro_act_plot)
            right_panel.addWidget(metrics_group)

        self.status_label = QLabel("")
        left_panel.addWidget(self.status_label)

    def _one_click_capture(self):
        """Auto-configure and run immediately."""
        self.micro_steps_spin.setValue(100) # Ensure enough steps
        self._run_microscope_analysis()

    def _run_microscope_analysis(self):
        try:
            steps = self.micro_steps_spin.value()

            # Use current model if available
            if self.model is not None:
                model = self.model
                # Determine input shape from model
                if hasattr(model, 'input_dim'):
                     # Flattened input
                     input_shape = (1, model.input_dim)
                elif hasattr(model, 'embed'):
                     # LM input (tokens)
                     input_shape = (1, 128) # Fake sequence
                     x = torch.randint(0, model.embed.num_embeddings, input_shape)
                else:
                     # Fallback
                     input_shape = (1, 784)
            else:
                # Fallback to creating a new model if none loaded
                model_name = self.micro_model_combo.currentText()
                if "Conv" in model_name:
                    model = ConvEqProp(1, 16, 10, max_steps=steps)
                    input_shape = (1, 1, 28, 28)
                else:
                    model = LoopedMLP(784, 256, 10, max_steps=steps, use_spectral_norm=True)
                    input_shape = (1, 784)

            # Create random input if not created above (LM case)
            if 'x' not in locals():
                x = torch.randn(*input_shape)
                if hasattr(model, 'device'):
                     x = x.to(model.device)

            # Run forward with dynamics
            model.eval()
            with torch.no_grad():
                kwargs = {}
                import inspect
                sig = inspect.signature(model.forward)
                if 'return_dynamics' in sig.parameters:
                    kwargs['return_dynamics'] = True
                if 'return_trajectory' in sig.parameters:
                    kwargs['return_trajectory'] = True
                if 'steps' in sig.parameters:
                    kwargs['steps'] = steps

                # Preprocess input if needed (LM embedding)
                if hasattr(model, 'has_embed') and model.has_embed:
                     h = model.embed(x)
                     # If model expects flattened input (e.g. MLP LM), average
                     if isinstance(model, LoopedMLP):
                         h = h.mean(dim=1)
                     out = model(h, **kwargs)
                else:
                     out = model(x, **kwargs)

                # Handle output format
                if isinstance(out, tuple):
                    dynamics = out[1]
                else:
                    if hasattr(model, 'dynamics'):
                        dynamics = model.dynamics
                    else:
                        raise ValueError("Model did not return dynamics")

            deltas = dynamics.get('deltas', [])
            traj = dynamics.get('trajectory', [])

            if not deltas:
                 self.status_label.setText("No dynamics data returned.")
                 return

            if traj:
                activities = [h.abs().mean().item() for h in traj]
            else:
                activities = [0.0] * len(deltas)

            if hasattr(self, 'micro_conv_curve'):
                self.micro_conv_curve.setData(deltas)
                self.micro_act_curve.setData(activities)

            self.status_label.setText(f"Analysis complete. Final delta: {deltas[-1]:.2e}")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", str(e))
            import traceback
            traceback.print_exc()

    def update_model_ref(self, model):
        self.model = model

    def update_plots_from_data(self, deltas, activities):
        if hasattr(self, 'micro_conv_curve'):
            self.micro_conv_curve.setData(deltas)
            self.micro_act_curve.setData(activities)
