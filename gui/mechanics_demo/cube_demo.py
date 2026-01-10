"""
3D Neural Cube Demo

Visualizes 3D lattice neural network with local connectivity.
Shows TRAINING dynamics and activation propagation through 3D space.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QGridLayout
)
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from models import NeuralCube
from .utils import get_single_digit, load_mnist_subset, train_quick_epoch, get_cube_3d_positions


class CubeDemo(QWidget):
    """
    Demonstrate 3D neural cube with local connectivity.
    
    Shows:
    - Training dynamics: watch weights form during learning
    - Flowing activations: see signal propagate through 3D space
    - Topology advantage: 91% connection reduction
    """
    
    def __init__(self):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Build 6x6x6 cube (216 neurons)
        self.model = NeuralCube(
            cube_size=6,
            input_dim=784,
            output_dim=10,
            max_steps=15
        ).to(self.device)
        
        self.cube_size = 6
        self.positions = get_cube_3d_positions(self.cube_size)
        
        self.trajectory = []
        self.weight_history = []
        self.running = False
        self.training = False
        self.current_step = 0
        
        self.init_ui()
    
    def init_ui(self):
        """Build UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel(
            "<h2>3D Neural Cube (6×6×6 = 216 Neurons)</h2>"
            "<p>Watch <b>training dynamics</b> and <b>activation flow</b> in 3D space. "
            "Local 26-neighbor connectivity provides <b>91% connection reduction</b> "
            "while mimicking biological neural tissue.</p>"
        )
        header.setWordWrap(True)
        header.setStyleSheet("padding: 10px; background-color: #2c3e50; border-radius: 5px;")
        layout.addWidget(header)
        
        # Visualization area
        viz_layout = QHBoxLayout()
        
        # Try 3D visualization first
        try:
            # 3D View
            self.view_3d = gl.GLViewWidget()
            self.view_3d.opts['distance'] = 20
            self.view_3d.setBackgroundColor('k')
            
            # Add grid
            grid = gl.GLGridItem()
            grid.scale(2, 2, 1)
            self.view_3d.addItem(grid)
            
            # Scatter plot for neurons
            self.scatter = gl.GLScatterPlotItem(
                pos=self.positions,
                color=(0.5, 0.5, 0.5, 0.8),
                size=0.3
            )
            self.view_3d.addItem(self.scatter)
            
            viz_layout.addWidget(self.view_3d)
            self.use_3d = True
            
        except Exception as e:
            # Fallback to 2D slices
            print(f"3D visualization failed: {e}. Using 2D slices.")
            self.use_3d = False
            
            slice_group = QGroupBox("2D Slice View (z-slices)")
            slice_layout = QVBoxLayout(slice_group)
            
            self.slice_plot = pg.PlotWidget()
            self.slice_plot.setAspectLocked(True)
            self.slice_plot.hideAxis('bottom')
            self.slice_plot.hideAxis('left')
            self.slice_image = pg.ImageItem()
            self.slice_plot.addItem(self.slice_image)
            slice_layout.addWidget(self.slice_plot)
            
            viz_layout.addWidget(slice_group)
        
        # Stats + Weight Matrix panel
        right_panel = QVBoxLayout()
        
        # Stats panel
        stats_group = QGroupBox("Topology Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        stats = self.model.get_topology_stats()
        stats_text = (
            f"<b>Cube Size:</b> {stats['cube_size']}×{stats['cube_size']}×{stats['cube_size']}<br>"
            f"<b>Total Neurons:</b> {stats['n_neurons']}<br>"
            f"<b>Local Connections:</b> {stats['local_connections']:,}<br>"
            f"<b>Fully-Connected Equiv:</b> {stats['fully_connected_equivalent']:,}<br>"
            f"<b>Reduction:</b> <span style='color:#27ae60;'>{stats['connection_reduction']*100:.1f}%</span>"
        )
        stats_label = QLabel(stats_text)
        stats_label.setWordWrap(True)
        stats_label.setStyleSheet("padding: 10px; background-color: #34495e; border-radius: 5px;")
        stats_layout.addWidget(stats_label)
        
        right_panel.addWidget(stats_group)
        
        # Weight Matrix Heatmap
        weight_group = QGroupBox("Weight Matrix (Local Connectivity)")
        weight_layout = QVBoxLayout(weight_group)
        self.weight_plot = pg.PlotWidget()
        self.weight_plot.setAspectLocked(True)
        self.weight_plot.hideAxis('bottom')
        self.weight_plot.hideAxis('left')
        self.weight_image = pg.ImageItem()
        self.weight_plot.addItem(self.weight_image)
        weight_layout.addWidget(self.weight_plot)
        
        right_panel.addWidget(weight_group)
        
        viz_layout.addLayout(right_panel)
        
        layout.addLayout(viz_layout)
        
        # Controls
        controls = QGridLayout()
        
        self.train_btn = QPushButton("1️⃣ Train Cube (2 epochs)")
        self.train_btn.clicked.connect(self.train_cube)
        self.train_btn.setStyleSheet("background-color: #3498db; color: white; padding: 10px; font-weight: bold;")
        controls.addWidget(self.train_btn, 0, 0)
        
        self.flow_btn = QPushButton("2️⃣ Show Activation Flow")
        self.flow_btn.clicked.connect(self.run_activation_flow)
        self.flow_btn.setEnabled(False)
        self.flow_btn.setStyleSheet("background-color: #9b59b6; color: white; padding: 10px; font-weight: bold;")
        controls.addWidget(self.flow_btn, 0, 1)
        
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self.reset)
        controls.addWidget(self.reset_btn, 1, 0, 1, 2)
        
        layout.addLayout(controls)
        
        # Status
        self.status_label = QLabel("Click 'Train Cube' to begin")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
    
    def train_cube(self):
        """Train cube model and show weight evolution."""
        self.status_label.setText("Training cube on MNIST...")
        self.train_btn.setEnabled(False)
        
        train_loader = load_mnist_subset(train=True, subset_size=500, batch_size=32)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train for 2 epochs
        for epoch in range(2):
            loss, acc = train_quick_epoch(self.model, train_loader, optimizer, self.device)
            self.status_label.setText(f"Epoch {epoch+1}/2: Loss={loss:.3f}, Acc={acc*100:.1f}%")
            
            # Update weight visualization
            with torch.no_grad():
                weights = self.model.W_local.cpu().numpy()
                # Show first 20 neurons' connections as 20x27 matrix
                self.weight_image.setImage(weights[:20, :])
        
        self.flow_btn.setEnabled(True)
        self.status_label.setText(f"✓ Training complete! Accuracy: {acc*100:.1f}%")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
    
    def run_activation_flow(self):
        """Run activation flow animation."""
        if self.running:
            self.timer.stop()
            self.running = False
            self.flow_btn.setText("2️⃣ Show Activation Flow")
            return
        
        # Get test digit
        digit_img, label = get_single_digit(digit=np.random.randint(0, 10))
        digit_img = digit_img.to(self.device)
        
        # Compute trajectory
        self.model.eval()
        with torch.no_grad():
            _, trajectory = self.model.forward(digit_img, steps=15, return_trajectory=True)
            self.trajectory = [h.cpu().numpy()[0] for h in trajectory]
        
        # Start animation
        self.current_step = 0
        self.running = True
        self.flow_btn.setText("⏸ Pause")
        self.status_label.setText(f"Showing activation flow for digit {label}...")
        self.status_label.setStyleSheet("color: #3498db;")
        self.timer.start(150)  # 150ms per frame for visible flow
    
    def update_animation(self):
        """Update animation frame."""
        if self.current_step >= len(self.trajectory):
            self.timer.stop()
            self.running = False
            self.flow_btn.setText("2️⃣ Show Activation Flow")
            self.status_label.setText("✓ Convergence complete!")
            return
        
        activations = self.trajectory[self.current_step]
        
        if self.use_3d:
            # Update 3D scatter plot with flowing activations
            # Color and size based on activation magnitude
            act_norm = (activations - activations.min()) / (activations.max() - activations.min() + 1e-8)
            
            colors = np.zeros((len(activations), 4))
            # Use red-to-blue colormap for activation strength
            colors[:, 0] = act_norm  # Red channel (active)
            colors[:, 2] = 1 - act_norm  # Blue channel (inactive)
            colors[:, 3] = 0.8  # Alpha
            
            sizes = 0.2 + act_norm * 0.8  # Vary size with activation
            
            self.scatter.setData(pos=self.positions, color=colors, size=sizes)
        else:
            # Update 2D slice (middle slice)
            z_mid = self.cube_size // 2
            slice_2d = activations.reshape(self.cube_size, self.cube_size, self.cube_size)[z_mid]
            self.slice_image.setImage(slice_2d)
        
        self.status_label.setText(f"Convergence step {self.current_step+1}/15 - Watch activations flow!")
        self.current_step += 1
    
    def reset(self):
        """Reset demo."""
        self.timer.stop()
        self.running = False
        self.current_step = 0
        self.trajectory = []
        
        if self.use_3d:
            self.scatter.setData(pos=self.positions, color=(0.5, 0.5, 0.5, 0.8), size=0.3)
        else:
            self.slice_image.clear()
        
        self.weight_image.clear()
        self.flow_btn.setText("2️⃣ Show Activation Flow")
        self.status_label.setText("Click 'Train Cube' to begin")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
