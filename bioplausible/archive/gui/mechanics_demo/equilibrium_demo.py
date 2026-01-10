"""
Equilibrium Convergence Demo

Shows actual hidden state dynamics converging to equilibrium on a real MNIST digit.
Animates the convergence process step-by-step with heatmap visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

from models import LoopedMLP
from .utils import get_single_digit, load_mnist_subset, train_quick_epoch


class EquilibriumDemo(QWidget):
    """
    Demonstrate equilibrium convergence on real MNIST.
    
    Shows:
    - Hidden state heatmap evolving over 30 steps
    - L2 distance to final equilibrium (exponential decay)
    - Prediction confidence increasing
    """
    
    def __init__(self):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
       # Model (small for speed)
        self.model = LoopedMLP(
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
            use_spectral_norm=True,
            max_steps=30
        ).to(self.device)
        
        # State
        self.trained = False
        self.running = False
        self.current_step = 0
        self.trajectory = []
        self.digit_img = None
        self.digit_label = None
        
        self.init_ui()
    
    def init_ui(self):
        """Build UI."""
        layout = QVBoxLayout(self)
        
        # Info header
        header = QLabel(
            "<h2>Equilibrium Convergence on Real MNIST</h2>"
            "<p>Watch hidden states converge to a fixed point over 30 iterations. "
            "This is <b>actual computation</b> on a real digit image, not a simulation.</p>"
        )
        header.setWordWrap(True)
        header.setStyleSheet("padding: 10px; background-color: #2c3e50; border-radius: 5px;")
        layout.addWidget(header)
        
        # Visualization area
        viz_layout = QHBoxLayout()
        
        # Left: Hidden state heatmap
        heatmap_group = QGroupBox("Hidden State Heatmap (128 neurons)")
        heatmap_layout = QVBoxLayout(heatmap_group)
        self.heatmap_plot = pg.PlotWidget()
        self.heatmap_plot.setAspectLocked(True)
        self.heatmap_plot.hideAxis('bottom')
        self.heatmap_plot.hideAxis('left')
        self.heatmap_image = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_image)
        heatmap_layout.addWidget(self.heatmap_plot)
        viz_layout.addWidget(heatmap_group)
        
        # Right: Convergence metrics
        metrics_group = QGroupBox("Convergence Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Distance to equilibrium plot
        self.distance_plot = pg.PlotWidget(title="Distance to Equilibrium")
        self.distance_plot.setLabel('bottom', 'Step')
        self.distance_plot.setLabel('left', 'L2 Distance')
        self.distance_plot.showGrid(x=True, y=True, alpha=0.3)
        self.distance_curve = self.distance_plot.plot(pen=pg.mkPen('#3498db', width=2))
        metrics_layout.addWidget(self.distance_plot)
        
        # Prediction confidence
        self.confidence_plot = pg.PlotWidget(title="Prediction Confidence")
        self.confidence_plot.setLabel('bottom', 'Step')
        self.confidence_plot.setLabel('left', 'Max Softmax Prob')
        self.confidence_plot.setYRange(0, 1)
        self.confidence_plot.showGrid(x=True, y=True, alpha=0.3)
        self.confidence_curve = self.confidence_plot.plot(pen=pg.mkPen('#27ae60', width=2))
        metrics_layout.addWidget(self.confidence_plot)
        
        viz_layout.addWidget(metrics_group)
        
        layout.addLayout(viz_layout)
        
        # Controls
        controls = QHBoxLayout()
        
        self.train_btn = QPushButton("1️⃣ Train Model (Quick)")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setStyleSheet("background-color: #3498db; color: white; padding: 10px; font-weight: bold;")
        controls.addWidget(self.train_btn)
        
        self.run_btn = QPushButton("2️⃣ Run Convergence Animation")
        self.run_btn.clicked.connect(self.run_convergence)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")
        controls.addWidget(self.run_btn)
        
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self.reset)
        controls.addWidget(self.reset_btn)
        
        layout.addLayout(controls)
        
        # Status
        self.status_label = QLabel("Click 'Train Model' to begin")
        self.status_label.setStyleSheet("color: #888; padding: 5px; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
    
    def train_model(self):
        """Quick training on MNIST subset."""
        self.status_label.setText("Training on 1000 MNIST samples...")
        self.train_btn.setEnabled(False)
        
        # Load small dataset
        train_loader = load_mnist_subset(train=True, subset_size=1000, batch_size=64)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train for 2 epochs (should be fast)
        for epoch in range(2):
            loss, acc = train_quick_epoch(self.model, train_loader, optimizer, self.device)
            self.status_label.setText(f"Epoch {epoch+1}/2: Loss={loss:.3f}, Acc={acc*100:.1f}%")
        
        self.trained = True
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"✓ Training complete! Accuracy: {acc*100:.1f}%. Now click 'Run Convergence'")
        self.status_label.setStyleSheet("color: #27ae60; padding: 5px; font-weight: bold;")
    
    def run_convergence(self):
        """Run convergence animation on a single digit."""
        if self.running:
            self.timer.stop()
            self.running = False
            self.run_btn.setText("2️⃣ Run Convergence Animation")
            return
        
        # Get a test digit
        self.digit_img, self.digit_label = get_single_digit(digit=np.random.randint(0, 10))
        self.digit_img = self.digit_img.to(self.device)
        
        # Compute full trajectory
        self.model.eval()
        with torch.no_grad():
            _, trajectory = self.model.forward(self.digit_img, steps=30, return_trajectory=True)
            self.trajectory = [h.cpu().numpy()[0] for h in trajectory]
        
        # Reset animation
        self.current_step = 0
        self.running = True
        self.run_btn.setText("⏸ Pause")
        self.timer.start(50)  # 50ms per frame = 20 fps
    
    def update_animation(self):
        """Update one frame of animation."""
        if self.current_step >= len(self.trajectory):
            self.timer.stop()
            self.running = False
            self.run_btn.setText("2️⃣ Run Convergence Animation")
            self.status_label.setText("✓ Convergence complete!")
            return
        
        h_current = self.trajectory[self.current_step]
        h_final = self.trajectory[-1]
        
        # Update heatmap (reshape 128 to 8x16 for visualization)
        heatmap_2d = h_current.reshape(8, 16)
        self.heatmap_image.setImage(heatmap_2d.T)
        
        # Update distance plot
        distances = []
        for i in range(self.current_step + 1):
            dist = np.linalg.norm(self.trajectory[i] - h_final)
            distances.append(dist)
        self.distance_curve.setData(distances)
        
        # Update confidence plot
        confidences = []
        for i in range(self.current_step + 1):
            h = torch.from_numpy(self.trajectory[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model.W_out(h)
                prob = torch.softmax(output, dim=1).max().item()
            confidences.append(prob)
        self.confidence_curve.setData(confidences)
        
        # Update status
        self.status_label.setText(
            f"Step {self.current_step}/{len(self.trajectory)-1} - "
            f"Distance={distances[-1]:.4f}, Confidence={confidences[-1]*100:.1f}%"
        )
        
        self.current_step += 1
    
    def reset(self):
        """Reset demo."""
        self.timer.stop()
        self.running = False
        self.current_step = 0
        self.trajectory = []
        self.heatmap_image.clear()
        self.distance_curve.setData([])
        self.confidence_curve.setData([])
        self.run_btn.setText("2️⃣ Run Convergence Animation")
        self.status_label.setText("Click 'Train Model' to begin" if not self.trained else "Click 'Run Convergence'")
        self.status_label.setStyleSheet("color: #888; padding: 5px; font-size: 11px;")
