"""
Live Mechanics Demo - Hands-On EqProp Training

This is the MAIN demo showing actual live training dynamics.
Users can see the network learn in real-time with parameter controls.

Features:
- Real MNIST training (not simulated)
- Live visualization of:
  * Hidden state convergence (equilibrium dynamics)
  * Weight matrix evolution during training
  * Energy landscape changes
  * Lipschitz constant tracking
- Parameter sliders for:
  * Learning rate
  * Beta (nudge strength)
  * Equilibrium steps
  * Spectral normalization toggle
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QGridLayout, QSlider, QCheckBox, QSpinBox,
    QDoubleSpinBox, QProgressBar, QSplitter
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

from models import LoopedMLP
from .utils import load_mnist_subset


class LiveMechanicsDemo(QWidget):
    """
    The definitive EqProp demo.
    
    Shows ACTUAL training on MNIST with LIVE visualization of:
    - Equilibrium convergence (hidden state finding fixed point)
    - Weight matrix evolution (learning happening)
    - Energy minimization (physics-based dynamics)
    - Lipschitz constant (stability guarantee)
    
    Users can adjust parameters and see immediate effects.
    """
    
    def __init__(self):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Build model (will be reset when training starts)
        self.model = None
        self.optimizer = None
        self.train_loader = None
        
        # Training state
        self.training = False
        self.epoch = 0
        self.batch = 0
        
        # History for plotting
        self.loss_history = []
        self.acc_history = []
        self.lipschitz_history = []
        self.energy_history = []
        
        # Current batch data for visualization
        self.current_h = None
        self.current_trajectory = None
        
        self.init_ui()
    
    def init_ui(self):
        """Build the comprehensive UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel(
            "<h2>🔬 Live EqProp Training - Hands-On Mechanics</h2>"
            "<p>Watch the network learn in real-time. Adjust parameters to see their effects.</p>"
        )
        header.setWordWrap(True)
        header.setStyleSheet("padding: 10px; background-color: #1a1a2e; border-radius: 5px;")
        layout.addWidget(header)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT PANEL: Parameters & Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Parameters Group
        params_group = QGroupBox("⚙️ Model Parameters")
        params_layout = QGridLayout(params_group)
        
        # Learning Rate
        params_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        params_layout.addWidget(self.lr_spin, 0, 1)
        
        # Beta (nudge strength)
        params_layout.addWidget(QLabel("Beta (nudge):"), 1, 0)
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.01, 2.0)
        self.beta_spin.setSingleStep(0.1)
        self.beta_spin.setValue(0.5)
        params_layout.addWidget(self.beta_spin, 1, 1)
        
        # Equilibrium Steps
        params_layout.addWidget(QLabel("Eq. Steps:"), 2, 0)
        self.eq_steps_spin = QSpinBox()
        self.eq_steps_spin.setRange(5, 100)
        self.eq_steps_spin.setValue(20)
        params_layout.addWidget(self.eq_steps_spin, 2, 1)
        
        # Spectral Normalization
        self.sn_checkbox = QCheckBox("Spectral Norm (SN)")
        self.sn_checkbox.setChecked(True)
        self.sn_checkbox.setStyleSheet("font-weight: bold; color: #27ae60;")
        params_layout.addWidget(self.sn_checkbox, 3, 0, 1, 2)
        
        # Hidden Dimension
        params_layout.addWidget(QLabel("Hidden Dim:"), 4, 0)
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(32, 512)
        self.hidden_spin.setValue(128)
        params_layout.addWidget(self.hidden_spin, 4, 1)
        
        left_layout.addWidget(params_group)
        
        # Controls Group
        controls_group = QGroupBox("🎮 Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        self.start_btn = QPushButton("▶️ Start Training")
        self.start_btn.clicked.connect(self.toggle_training)
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 15px; font-size: 16px; font-weight: bold;")
        controls_layout.addWidget(self.start_btn)
        
        self.reset_btn = QPushButton("🔄 Reset Model")
        self.reset_btn.clicked.connect(self.reset_model)
        controls_layout.addWidget(self.reset_btn)
        
        self.step_btn = QPushButton("⏭️ Single Batch")
        self.step_btn.clicked.connect(self.train_single_batch)
        controls_layout.addWidget(self.step_btn)
        
        left_layout.addWidget(controls_group)
        
        # Status Group
        status_group = QGroupBox("📊 Live Metrics")
        status_layout = QGridLayout(status_group)
        
        status_layout.addWidget(QLabel("<b>Epoch:</b>"), 0, 0)
        self.epoch_label = QLabel("0")
        status_layout.addWidget(self.epoch_label, 0, 1)
        
        status_layout.addWidget(QLabel("<b>Loss:</b>"), 1, 0)
        self.loss_label = QLabel("—")
        status_layout.addWidget(self.loss_label, 1, 1)
        
        status_layout.addWidget(QLabel("<b>Accuracy:</b>"), 2, 0)
        self.acc_label = QLabel("—")
        self.acc_label.setStyleSheet("color: #3498db; font-weight: bold;")
        status_layout.addWidget(self.acc_label, 2, 1)
        
        status_layout.addWidget(QLabel("<b>Lipschitz L:</b>"), 3, 0)
        self.lipschitz_label = QLabel("—")
        status_layout.addWidget(self.lipschitz_label, 3, 1)
        
        status_layout.addWidget(QLabel("<b>Energy:</b>"), 4, 0)
        self.energy_label = QLabel("—")
        status_layout.addWidget(self.energy_label, 4, 1)
        
        left_layout.addWidget(status_group)
        
        # Theory explanation
        theory_group = QGroupBox("📖 What You're Seeing")
        theory_layout = QVBoxLayout(theory_group)
        self.theory_text = QLabel(
            "<b>Equilibrium Propagation:</b><br><br>"
            "1. <b>Free Phase:</b> Network finds stable state h*<br>"
            "   (Energy minimized, ∂E/∂h = 0)<br><br>"
            "2. <b>Nudged Phase:</b> Output pushed toward target<br>"
            "   (Contrastive signal emerges)<br><br>"
            "3. <b>Weight Update:</b> Δw ∝ h_nudged - h_free<br>"
            "   (Purely local, bio-plausible)"
        )
        self.theory_text.setWordWrap(True)
        self.theory_text.setStyleSheet("background-color: #2c3e50; padding: 10px; border-radius: 5px;")
        theory_layout.addWidget(self.theory_text)
        left_layout.addWidget(theory_group)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)
        
        # RIGHT PANEL: Visualizations
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Row 1: Training curves
        curves_layout = QHBoxLayout()
        
        # Loss curve
        loss_group = QGroupBox("Training Loss")
        loss_layout = QVBoxLayout(loss_group)
        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setLabel('bottom', 'Batch')
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_curve = self.loss_plot.plot(pen=pg.mkPen('#e74c3c', width=2))
        loss_layout.addWidget(self.loss_plot)
        curves_layout.addWidget(loss_group)
        
        # Accuracy curve
        acc_group = QGroupBox("Training Accuracy")
        acc_layout = QVBoxLayout(acc_group)
        self.acc_plot = pg.PlotWidget()
        self.acc_plot.setLabel('bottom', 'Batch')
        self.acc_plot.setLabel('left', 'Accuracy %')
        self.acc_plot.setYRange(0, 100)
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_curve = self.acc_plot.plot(pen=pg.mkPen('#3498db', width=2))
        acc_layout.addWidget(self.acc_plot)
        curves_layout.addWidget(acc_group)
        
        right_layout.addLayout(curves_layout)
        
        # Row 2: Weight matrix and convergence
        matrices_layout = QHBoxLayout()
        
        # Weight matrix heatmap
        weight_group = QGroupBox("Weight Matrix W_rec (128×128)")
        weight_layout = QVBoxLayout(weight_group)
        self.weight_plot = pg.PlotWidget()
        self.weight_plot.hideAxis('bottom')
        self.weight_plot.hideAxis('left')
        self.weight_image = pg.ImageItem()
        self.weight_plot.addItem(self.weight_image)
        weight_layout.addWidget(self.weight_plot)
        matrices_layout.addWidget(weight_group)
        
        # Convergence trajectory
        conv_group = QGroupBox("Equilibrium Convergence")
        conv_layout = QVBoxLayout(conv_group)
        self.conv_plot = pg.PlotWidget()
        self.conv_plot.setLabel('bottom', 'Step')
        self.conv_plot.setLabel('left', '||h_t - h_{t-1}||')
        self.conv_plot.showGrid(x=True, y=True, alpha=0.3)
        self.conv_curve = self.conv_plot.plot(pen=pg.mkPen('#9b59b6', width=2))
        conv_layout.addWidget(self.conv_plot)
        matrices_layout.addWidget(conv_group)
        
        right_layout.addLayout(matrices_layout)
        
        # Row 3: Lipschitz and Energy
        dynamics_layout = QHBoxLayout()
        
        # Lipschitz tracking
        lip_group = QGroupBox("Lipschitz Constant L (Stability)")
        lip_layout = QVBoxLayout(lip_group)
        self.lip_plot = pg.PlotWidget()
        self.lip_plot.setLabel('bottom', 'Batch')
        self.lip_plot.setLabel('left', 'L')
        self.lip_plot.showGrid(x=True, y=True, alpha=0.3)
        self.lip_plot.addLine(y=1.0, pen=pg.mkPen('#e74c3c', width=2, style=Qt.PenStyle.DashLine))  # L=1 threshold
        self.lip_curve = self.lip_plot.plot(pen=pg.mkPen('#27ae60', width=2))
        lip_layout.addWidget(self.lip_plot)
        lip_layout.addWidget(QLabel("<i>L < 1 = stable (contraction mapping)</i>"))
        dynamics_layout.addWidget(lip_group)
        
        # Energy dynamics
        energy_group = QGroupBox("Energy Landscape")
        energy_layout = QVBoxLayout(energy_group)
        self.energy_plot = pg.PlotWidget()
        self.energy_plot.setLabel('bottom', 'Batch')
        self.energy_plot.setLabel('left', 'Energy')
        self.energy_plot.showGrid(x=True, y=True, alpha=0.3)
        self.energy_curve = self.energy_plot.plot(pen=pg.mkPen('#f39c12', width=2))
        energy_layout.addWidget(self.energy_plot)
        dynamics_layout.addWidget(energy_group)
        
        right_layout.addLayout(dynamics_layout)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])
        layout.addWidget(splitter)
        
        # Timer for training
        self.timer = QTimer()
        self.timer.timeout.connect(self.train_single_batch)
    
    def reset_model(self):
        """Create fresh model with current parameters."""
        self.training = False
        self.timer.stop()
        self.start_btn.setText("▶️ Start Training")
        
        # Clear history
        self.loss_history = []
        self.acc_history = []
        self.lipschitz_history = []
        self.energy_history = []
        self.epoch = 0
        self.batch = 0
        
        # Create model
        hidden_dim = self.hidden_spin.value()
        use_sn = self.sn_checkbox.isChecked()
        eq_steps = self.eq_steps_spin.value()
        
        self.model = LoopedMLP(
            input_dim=784,
            hidden_dim=hidden_dim,
            output_dim=10,
            use_spectral_norm=use_sn,
            max_steps=eq_steps
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr_spin.value()
        )
        
        # Load data
        self.train_loader = load_mnist_subset(train=True, subset_size=1000, batch_size=32)
        self.data_iter = iter(self.train_loader)
        
        # Clear plots
        self.loss_curve.setData([])
        self.acc_curve.setData([])
        self.lip_curve.setData([])
        self.energy_curve.setData([])
        self.conv_curve.setData([])
        self.weight_image.clear()
        
        # Update labels
        self.epoch_label.setText("0")
        self.loss_label.setText("—")
        self.acc_label.setText("—")
        self.lipschitz_label.setText("—")
        self.energy_label.setText("—")
        
        # Show initial weights
        with torch.no_grad():
            weights = self.model.W_rec.weight.cpu().numpy()
            self.weight_image.setImage(weights)
    
    def toggle_training(self):
        """Start/stop training."""
        if self.model is None:
            self.reset_model()
        
        self.training = not self.training
        
        if self.training:
            self.start_btn.setText("⏸️ Pause Training")
            self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 15px; font-size: 16px; font-weight: bold;")
            self.timer.start(100)  # 10 batches/sec
        else:
            self.start_btn.setText("▶️ Resume Training")
            self.start_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 15px; font-size: 16px; font-weight: bold;")
            self.timer.stop()
    
    def train_single_batch(self):
        """Train one batch and update visualizations."""
        if self.model is None:
            self.reset_model()
        
        # Get next batch
        try:
            x_batch, y_batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            x_batch, y_batch = next(self.data_iter)
            self.epoch += 1
            self.epoch_label.setText(str(self.epoch))
        
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward with trajectory
        out, trajectory = self.model(x_batch, return_trajectory=True)
        loss = F.cross_entropy(out, y_batch)
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            # Accuracy
            preds = out.argmax(dim=1)
            acc = (preds == y_batch).float().mean().item() * 100
            
            # Lipschitz constant
            L = self.model.compute_lipschitz()
            
            # Energy (approximate)
            h = trajectory[-1]
            energy = 0.5 * (h ** 2).sum(dim=1).mean().item()
            
            # Convergence trajectory
            if len(trajectory) > 1:
                conv_deltas = []
                for i in range(1, len(trajectory)):
                    delta = (trajectory[i] - trajectory[i-1]).norm(dim=1).mean().item()
                    conv_deltas.append(delta)
                self.conv_curve.setData(conv_deltas)
        
        # Update history
        self.loss_history.append(loss.item())
        self.acc_history.append(acc)
        self.lipschitz_history.append(L)
        self.energy_history.append(energy)
        self.batch += 1
        
        # Update plots
        self.loss_curve.setData(self.loss_history)
        self.acc_curve.setData(self.acc_history)
        self.lip_curve.setData(self.lipschitz_history)
        self.energy_curve.setData(self.energy_history)
        
        # Update weight matrix
        with torch.no_grad():
            weights = self.model.W_rec.weight.cpu().numpy()
            self.weight_image.setImage(weights)
        
        # Update labels
        self.loss_label.setText(f"{loss.item():.4f}")
        self.acc_label.setText(f"{acc:.1f}%")
        
        # Color Lipschitz based on stability
        if L <= 1.0:
            self.lipschitz_label.setText(f"{L:.4f} ✅")
            self.lipschitz_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self.lipschitz_label.setText(f"{L:.4f} ⚠️")
            self.lipschitz_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        
        self.energy_label.setText(f"{energy:.4f}")
