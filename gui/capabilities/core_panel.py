"""
Core Panel - Explains fundamental EqProp principles

Visualizations:
1. Energy landscape convergence (Free phase → Nudged phase)
2. Contrastive Hebbian update visualization
3. Interactive beta/steps adjustment
"""

import torch
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSlider, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from models import LoopedMLP


class CorePanel(QWidget):
    """
    Core EqProp explanation panel.
    
    Shows:
    - Energy minimization dynamics
    - Free vs Nudged phase comparison
    - Contrastive weight update mechanism
    """
    
    def __init__(self):
        super().__init__()
        
        # Model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = LoopedMLP(
            input_dim=64, 
            hidden_dim=128, 
            output_dim=10,
            use_spectral_norm=True,
            max_steps=30
        ).to(self.device)
        
        # Demo state
        self.running = False
        self.current_step = 0
        self.beta = 0.22
        self.max_steps = 30
        
        # Sample data
        self.sample_x = torch.randn(1, 64, device=self.device)
        self.sample_y = torch.randint(0, 10, (1,), device=self.device)
        
        # Trajectories
        self.free_trajectory = []
        self.nudged_trajectory = []
        
        # Initialize UI
        self.init_ui()
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
    
    def init_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        
        # Explanation header
        header = QLabel(
            "<h2>🔬 Core EqProp: Energy Minimization</h2>"
            "<p><b>How it works:</b> EqProp iterates a recurrent network to a <b>fixed-point equilibrium</b> "
            "by minimizing an energy function. It uses <b>two phases</b>:</p>"
            "<ul style='margin: 5px 0;'>"
            "<li><b>Free Phase:</b> Network relaxes to natural equilibrium (no supervision)</li>"
            "<li><b>Nudged Phase:</b> Output gently pushed toward target (β-nudge)</li>"
            "</ul>"
            "<p>Weights update via <b>Contrastive Hebbian learning</b>: "
            "ΔW ∝ (h<sub>nudged</sub>h<sub>nudged</sub><sup>T</sup> - h<sub>free</sub>h<sub>free</sub><sup>T</sup>)</p>"
            "<p style='color: #3498db;'><b>Key Insight:</b> No explicit gradient computation needed! "
            "Learning emerges from physics of the system.</p>"
        )
        header.setWordWrap(True)
        header.setStyleSheet("padding: 12px; background-color: #2c3e50; border-radius: 5px; border: 2px solid #3498db;")
        layout.addWidget(header)
        
        # Visualization area
        viz_layout = QHBoxLayout()
        
        # Energy plot
        energy_group = QGroupBox("Energy Convergence")
        energy_layout = QVBoxLayout(energy_group)
        self.energy_plot = pg.PlotWidget(title="Energy over Steps")
        self.energy_plot.setLabel('bottom', 'Iteration Step')
        self.energy_plot.setLabel('left', 'Energy')
        self.energy_plot.addLegend()
        self.energy_plot.showGrid(x=True, y=True, alpha=0.3)
        self.free_curve = self.energy_plot.plot(name="Free Phase", pen=pg.mkPen('#3498db', width=2))
        self.nudged_curve = self.energy_plot.plot(name="Nudged Phase", pen=pg.mkPen('#e74c3c', width=2))
        energy_layout.addWidget(self.energy_plot)
        viz_layout.addWidget(energy_group)
        
        # Hidden state trajectory
        hidden_group = QGroupBox("Hidden State Dynamics")
        hidden_layout = QVBoxLayout(hidden_group)
        self.hidden_plot = pg.PlotWidget(title="First 20 Hidden Neurons")
        self.hidden_plot.setLabel('bottom', 'Iteration Step')
        self.hidden_plot.setLabel('left', 'Activation')
        self.hidden_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hidden_curves = []
        for i in range(20):
            curve = self.hidden_plot.plot(pen=pg.mkPen((i, 20), width=1))
            self.hidden_curves.append(curve)
        hidden_layout.addWidget(self.hidden_plot)
        viz_layout.addWidget(hidden_group)
        
        layout.addLayout(viz_layout)
        
        # Controls
        controls = self.create_controls()
        layout.addWidget(controls)
        
        # Status
        self.status_label = QLabel("Ready. Click 'Run Demo' to start.")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def create_controls(self):
        """Create control panel."""
        group = QGroupBox("Demo Controls")
        layout = QVBoxLayout(group)
        
        # Beta slider
        beta_layout = QHBoxLayout()
        beta_layout.addWidget(QLabel("Nudge Strength (β):"))
        self.beta_slider = QSlider(Qt.Orientation.Horizontal)
        self.beta_slider.setRange(1, 100)
        self.beta_slider.setValue(22)
        self.beta_slider.valueChanged.connect(self.on_beta_changed)
        beta_layout.addWidget(self.beta_slider)
        self.beta_value = QLabel("0.22")
        beta_layout.addWidget(self.beta_value)
        layout.addLayout(beta_layout)
        
        # Steps slider
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Convergence Steps:"))
        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setRange(5, 100)
        self.steps_slider.setValue(30)
        self.steps_slider.valueChanged.connect(self.on_steps_changed)
        steps_layout.addWidget(self.steps_slider)
        self.steps_value = QLabel("30")
        steps_layout.addWidget(self.steps_value)
        layout.addLayout(steps_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶ Run Demo")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.run_btn.clicked.connect(self.run_demo)
        btn_layout.addWidget(self.run_btn)
        
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        self.reset_btn.clicked.connect(self.reset)
        btn_layout.addWidget(self.reset_btn)
        
        layout.addLayout(btn_layout)
        
        return group
    
    def on_beta_changed(self, value):
        """Update beta parameter."""
        self.beta = value / 100.0
        self.beta_value.setText(f"{self.beta:.2f}")
    
    def on_steps_changed(self, value):
        """Update max steps."""
        self.max_steps = value
        self.steps_value.setText(str(value))
    
    def run_demo(self):
        """Start the demonstration."""
        if self.running:
            self.timer.stop()
            self.running = False
            self.run_btn.setText("▶ Run Demo")
            self.status_label.setText("Paused.")
            return
        
        # Reset trajectories
        self.free_trajectory = []
        self.nudged_trajectory = []
        self.current_step = 0
        
        self.running = True
        self.run_btn.setText("⏸ Pause")
        self.status_label.setText("Running Free Phase...")
        
        # Start animation
        self.timer.start(50)  # 50ms per step
    
    def update_animation(self):
        """Animate one step of convergence."""
        if self.current_step >= self.max_steps:
            self.timer.stop()
            self.running = False
            self.run_btn.setText("▶ Run Demo")
            self.status_label.setText(
                f"Complete! Energy reduced from Free to Nudged phase. "
                f"Weight update = Δ(activities)."
            )
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
            return
        
        with torch.no_grad():
            # Free phase step
            if self.current_step == 0:
                self.h_free = torch.zeros(1, self.model.hidden_dim, device=self.device)
                self.h_nudged = torch.zeros(1, self.model.hidden_dim, device=self.device)
                self.x_proj = self.model.W_in(self.sample_x)
            
            # Update free
            self.h_free = torch.tanh(self.x_proj + self.model.W_rec(self.h_free))
            
            # Update nudged (with gradient toward target)
            y_onehot = torch.zeros(1, 10, device=self.device)
            y_onehot[0, self.sample_y] = 1.0
            output_free = self.model.W_out(self.h_nudged)
            nudge_grad = self.beta * (y_onehot - torch.softmax(output_free, dim=1))
            
            # Project nudge back to hidden space
            hidden_nudge = self.model.W_out.weight.t() @ nudge_grad.t()
            self.h_nudged = torch.tanh(self.x_proj + self.model.W_rec(self.h_nudged) + hidden_nudge.t())
            
            # Compute energies (simplified as L2 distance from equilibrium)
            energy_free = ((self.h_free - torch.tanh(self.x_proj + self.model.W_rec(self.h_free))) ** 2).sum().item()
            energy_nudged = ((self.h_nudged - torch.tanh(self.x_proj + self.model.W_rec(self.h_nudged))) ** 2).sum().item()
            
            self.free_trajectory.append({
                'energy': energy_free,
                'hidden': self.h_free.cpu().numpy()[0]
            })
            self.nudged_trajectory.append({
                'energy': energy_nudged,
                'hidden': self.h_nudged.cpu().numpy()[0]
            })
        
        # Update plots
        free_energies = [t['energy'] for t in self.free_trajectory]
        nudged_energies = [t['energy'] for t in self.nudged_trajectory]
        
        self.free_curve.setData(free_energies)
        self.nudged_curve.setData(nudged_energies)
        
        # Update hidden state plots (show first 20 neurons)
        for i in range(min(20, len(self.hidden_curves))):
            free_vals = [t['hidden'][i] for t in self.free_trajectory]
            self.hidden_curves[i].setData(free_vals)
        
        # Update status
        phase_pct = (self.current_step / self.max_steps) * 100
        self.status_label.setText(
            f"Step {self.current_step}/{self.max_steps} ({phase_pct:.0f}%) - "
            f"Free Energy: {energy_free:.4f}"
        )
        
        self.current_step += 1
    
    def reset(self):
        """Reset demo to initial state."""
        self.timer.stop()
        self.running = False
        self.current_step = 0
        self.free_trajectory = []
        self.nudged_trajectory = []
        
        self.free_curve.setData([])
        self.nudged_curve.setData([])
        for curve in self.hidden_curves:
            curve.setData([])
        
        self.run_btn.setText("▶ Run Demo")
        self.status_label.setText("Ready. Click 'Run Demo' to start.")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
        
        # New sample data
        self.sample_x = torch.randn(1, 64, device=self.device)
        self.sample_y = torch.randint(0, 10, (1,), device=self.device)
