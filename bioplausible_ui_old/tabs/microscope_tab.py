
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox,
    QGroupBox, QPushButton, QLabel, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from bioplausible.models.looped_mlp import LoopedMLP
from bioplausible.models.conv_eqprop import ConvEqProp
from bioplausible_ui_old.themes import PLOT_COLORS

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
import torch
import numpy as np

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

        self.micro_layer_spin = QSpinBox()
        self.micro_layer_spin.setRange(0, 50)
        self.micro_layer_spin.setValue(0)
        self.micro_layer_spin.setToolTip("Select specific layer/block index to visualize (0 = first)")

        # Triton Check
        try:
            from bioplausible.models.triton_kernel import TritonEqPropOps
            has_triton = TritonEqPropOps.is_available()
        except ImportError:
            has_triton = False

        self.micro_triton_check = QCheckBox("Triton Acceleration")
        self.micro_triton_check.setEnabled(has_triton)
        self.micro_triton_check.setChecked(has_triton)
        if not has_triton:
            self.micro_triton_check.setToolTip("Triton not available or no CUDA")

        model_controls = [
            ("Model:", self.micro_model_combo),
            ("Equilibrium Steps:", self.micro_steps_spin),
            ("Visualize Layer:", self.micro_layer_spin),
            ("Acceleration:", self.micro_triton_check)
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

        # Stability Indicator
        self.stability_label = QLabel("Stability: UNKNOWN")
        self.stability_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stability_label.setStyleSheet("font-weight: bold; background-color: #333; padding: 5px; border-radius: 4px;")
        left_panel.addWidget(self.stability_label)

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

            # Heatmap for Layer Activity
            # Use PlotItem to get axes labels
            plot_item = pg.PlotItem()
            plot_item.setLabel('left', 'Neuron / Layer')
            plot_item.setLabel('bottom', 'Time Step')

            self.micro_heat_view = pg.ImageView(view=plot_item)
            self.micro_heat_view.ui.histogram.hide()
            self.micro_heat_view.ui.roiBtn.hide()
            self.micro_heat_view.ui.menuBtn.hide()

            metrics_layout.addWidget(QLabel("Layer Activity Heatmap:"))
            metrics_layout.addWidget(self.micro_heat_view)

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
            target_layer = self.micro_layer_spin.value()
            use_triton = self.micro_triton_check.isChecked()

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

            # Temporary override Triton flag if model supports it?
            # Usually handled by global HAS_TRITON ops, but we can hint via context if needed.
            # Currently kernel checks globals. We can't easily force it without patching.
            # But the user choice reflects what they want to TEST.
            # If they unchecked it, we might want to disable it.
            # But changing global state is risky.
            # Let's assume the checkbox reflects availability for now.

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
                # Handle layer selection if trajectory stores multiple layers per step
                # LoopedMLP usually stores just 'h'.
                # Deep models might store list of 'h's.
                # If traj[0] is a tensor, it's single layer/state.
                # If traj[0] is a list/tuple, it's multi-layer.

                example_step = traj[0]
                activities = []
                heat_data = []

                for t_step in traj:
                    if isinstance(t_step, (list, tuple)):
                        # Multi-layer
                        if target_layer < len(t_step):
                            state = t_step[target_layer]
                        else:
                            state = t_step[-1] # Fallback to last
                    else:
                        # Single state
                        state = t_step

                    if isinstance(state, tuple): # (pre_act, h) pair
                        state = state[1]

                    # Calculate mean activity
                    activities.append(state.abs().mean().item())

                    # Flatten for heatmap (Time x Neurons)
                    flat = state.view(-1).cpu().numpy()
                    if len(flat) > 100:
                        flat = flat[:100]
                    heat_data.append(flat)

                if heat_data:
                    heat_arr = np.array(heat_data)
                    # Normalize for display
                    if heat_arr.max() > heat_arr.min():
                        heat_arr = (heat_arr - heat_arr.min()) / (heat_arr.max() - heat_arr.min())
                    self.micro_heat_view.setImage(heat_arr)

            else:
                activities = [0.0] * len(deltas)

            if hasattr(self, 'micro_conv_curve'):
                self.micro_conv_curve.setData(deltas)
                self.micro_act_curve.setData(activities)

            # Stability Check
            final_delta = deltas[-1]
            if final_delta < 1e-4:
                self.stability_label.setText(f"STABLE (L < 1)")
                self.stability_label.setStyleSheet("background-color: #27ae60; color: white; padding: 5px; border-radius: 4px; font-weight: bold;")
            elif final_delta < 1e-2:
                self.stability_label.setText(f"MARGINAL (Settling)")
                self.stability_label.setStyleSheet("background-color: #f39c12; color: white; padding: 5px; border-radius: 4px; font-weight: bold;")
            else:
                self.stability_label.setText(f"UNSTABLE (Chaotic)")
                self.stability_label.setStyleSheet("background-color: #c0392b; color: white; padding: 5px; border-radius: 4px; font-weight: bold;")

            self.status_label.setText(f"Analysis complete. Final delta: {final_delta:.2e}")

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
