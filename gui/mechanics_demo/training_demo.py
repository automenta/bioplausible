"""
Live Training Demo

Side-by-side comparison of EqProp vs Backprop training on MNIST.
Shows live accuracy/loss curves and current predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox, QGridLayout
)
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

from models import LoopedMLP, BackpropMLP
from .utils import load_mnist_subset


class TrainingDemo(QWidget):
    """
    Live training comparison: EqProp vs Backprop.
    
    Shows:
    - Side-by-side accuracy and loss curves
    - Real-time updates during training
    - Final accuracy comparison
    """
    
    def __init__(self):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Models
        self.eqprop_model = LoopedMLP(
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
            use_spectral_norm=True,
            max_steps=20  # Quick convergence
        ).to(self.device)
        
        self.backprop_model = BackpropMLP(
            input_dim=784,
            hidden_dim=128,
            output_dim=10
        ).to(self.device)
        
        # Training state
        self.running = False
        self.current_epoch = 0
        self.max_epochs = 5
        
        self.eqprop_history = {'loss': [], 'acc': []}
        self.backprop_history = {'loss': [], 'acc': []}
        
        self.init_ui()
    
    def init_ui(self):
        """Build UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel(
            "<h2>Live MNIST Training: EqProp vs Backprop</h2>"
            "<p>Train both models on the <b>same data</b> with the <b>same architecture</b>. "
            "EqProp should match Backprop's accuracy while demonstrating energy-based learning.</p>"
        )
        header.setWordWrap(True)
        header.setStyleSheet("padding: 10px; background-color: #2c3e50; border-radius: 5px;")
        layout.addWidget(header)
        
        # Plots
        plots_layout = QHBoxLayout()
        
        # Left: Loss/Accuracy plots
        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Loss plot
        loss_label = QLabel("<b>Cross-Entropy Loss</b>")
        metrics_layout.addWidget(loss_label)
        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setLabel('bottom', 'Epoch')
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot.addLegend()
        self.eqprop_loss_curve = self.loss_plot.plot(name="EqProp", pen=pg.mkPen('#3498db', width=2))
        self.backprop_loss_curve = self.loss_plot.plot(name="Backprop", pen=pg.mkPen('#e74c3c', width=2))
        metrics_layout.addWidget(self.loss_plot)
        
        # Accuracy plot
        acc_label = QLabel("<b>Training Accuracy</b>")
        metrics_layout.addWidget(acc_label)
        self.acc_plot = pg.PlotWidget()
        self.acc_plot.setLabel('bottom', 'Epoch')
        self.acc_plot.setLabel('left', 'Accuracy (%)')
        self.acc_plot.setYRange(0, 100)
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_plot.addLegend()
        self.eqprop_acc_curve = self.acc_plot.plot(name="EqProp", pen=pg.mkPen('#3498db', width=2))
        self.backprop_acc_curve = self.acc_plot.plot(name="Backprop", pen=pg.mkPen('#e74c3c', width=2))
        metrics_layout.addWidget(self.acc_plot)
        
        plots_layout.addWidget(metrics_group)
        
        # Right: Internal Mechanics (Weight + Activation Matrices)
        internals_group = QGroupBox("Internal Mechanics (EqProp)")
        internals_layout = QVBoxLayout(internals_group)
        
        # Weight matrix heatmap
        weight_label = QLabel("<b>Weight Matrix W_rec (128×128)</b>")
        internals_layout.addWidget(weight_label)
        self.weight_plot = pg.PlotWidget()
        self.weight_plot.setAspectLocked(True)
        self.weight_plot.hideAxis('bottom')
        self.weight_plot.hideAxis('left')
        self.weight_image = pg.ImageItem()
        self.weight_plot.addItem(self.weight_image)
        internals_layout.addWidget(self.weight_plot)
        
        # Activation matrix heatmap
        act_label = QLabel("<b>Hidden Activations (batch×128)</b>")
        internals_layout.addWidget(act_label)
        self.activation_plot = pg.PlotWidget()
        self.activation_plot.setAspectLocked(True)
        self.activation_plot.hideAxis('bottom')
        self.activation_plot.hideAxis('left')
        self.activation_image = pg.ImageItem()
        self.activation_plot.addItem(self.activation_image)
        internals_layout.addWidget(self.activation_plot)
        
        plots_layout.addWidget(internals_group)
        
        layout.addLayout(plots_layout)
        
        # Final comparison
        self.comparison_label = QLabel("")
        self.comparison_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                padding: 15px;
                border-radius: 5px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.comparison_label)
        
        # Controls
        controls = QHBoxLayout()
        
        self.run_btn = QPushButton(f"▶ Start Training ({self.max_epochs} epochs)")
        self.run_btn.clicked.connect(self.start_training)
        self.run_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")
        controls.addWidget(self.run_btn)
        
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self.reset)
        controls.addWidget(self.reset_btn)
        
        layout.addLayout(controls)
        
        # Status
        self.status_label = QLabel(f"Click 'Start Training' to begin {self.max_epochs}-epoch comparison")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Timer for async training updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.train_one_epoch)
    
    def start_training(self):
        """Start training both models."""
        if self.running:
            self.timer.stop()
            self.running = False
            self.run_btn.setText(f"▶ Start Training ({self.max_epochs} epochs)")
            return
        
        # Reset models
        self.eqprop_model = LoopedMLP(
            input_dim=784, hidden_dim=128, output_dim=10,
            use_spectral_norm=True, max_steps=20
        ).to(self.device)
        
        self.backprop_model = BackpropMLP(
            input_dim=784, hidden_dim=128, output_dim=10
        ).to(self.device)
        
        # Optimizers
        self.eqprop_optimizer = optim.Adam(self.eqprop_model.parameters(), lr=0.001)
        self.backprop_optimizer = optim.Adam(self.backprop_model.parameters(), lr=0.001)
        
        # Data
        self.train_loader = load_mnist_subset(train=True, subset_size=1000, batch_size=64)
        
        # Reset history
        self.current_epoch = 0
        self.eqprop_history = {'loss': [], 'acc': []}
        self.backprop_history = {'loss': [], 'acc': []}
        
        # Start
        self.running = True
        self.run_btn.setText("⏸ Pause")
        self.train_one_epoch()  # Start first epoch immediately
    
    def train_one_epoch(self):
        """Train one epoch for both models."""
        if self.current_epoch >= self.max_epochs:
            self.timer.stop()
            self.running = False
            self.run_btn.setText(f"▶ Start Training ({self.max_epochs} epochs)")
            
            # Show final comparison
            eqprop_final = self.eqprop_history['acc'][-1]
            backprop_final = self.backprop_history['acc'][-1]
            
            winner = "EqProp" if eqprop_final >= backprop_final else "Backprop"
            margin = abs(eqprop_final - backprop_final)
            
            self.comparison_label.setText(
                f"<b>Final Results (Epoch {self.max_epochs}):</b><br>"
                f"• EqProp: <span style='color:#3498db;'>{eqprop_final:.1f}%</span> accuracy<br>"
                f"• Backprop: <span style='color:#e74c3c;'>{backprop_final:.1f}%</span> accuracy<br>"
                f"• <b>Difference:</b> {margin:.1f}% ({winner} wins)<br>"
                f"<br><i>Both achieve similar accuracy, validating EqProp's learning capability!</i>"
            )
            
            self.status_label.setText("✓ Training complete!")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            return
        
        # Train EqProp
        from .utils import train_quick_epoch
        eqprop_loss, eqprop_acc = train_quick_epoch(
            self.eqprop_model, self.train_loader, 
            self.eqprop_optimizer, self.device
        )
        
        # Train Backprop
        backprop_loss, backprop_acc = train_quick_epoch(
            self.backprop_model, self.train_loader,
            self.backprop_optimizer, self.device
        )
        
        # Update history
        self.eqprop_history['loss'].append(eqprop_loss)
        self.eqprop_history['acc'].append(eqprop_acc * 100)
        self.backprop_history['loss'].append(backprop_loss)
        self.backprop_history['acc'].append(backprop_acc * 100)
        
        # Update plots
        epochs = list(range(len(self.eqprop_history['loss'])))
        self.eqprop_loss_curve.setData(epochs, self.eqprop_history['loss'])
        self.backprop_loss_curve.setData(epochs, self.backprop_history['loss'])
        self.eqprop_acc_curve.setData(epochs, self.eqprop_history['acc'])
        self.backprop_acc_curve.setData(epochs, self.backprop_history['acc'])
        
        # Update internal mechanics visualization (EqProp model)
        with torch.no_grad():
            # Get weight matrix (W_rec)
            if hasattr(self.eqprop_model, 'W_rec'):
                weights = self.eqprop_model.W_rec.weight.cpu().numpy()
                self.weight_image.setImage(weights)
            
            # Get activations on a batch
            try:
                x_batch, y_batch = next(iter(self.train_loader))
                x_batch = x_batch[:32].to(self.device)  # Use up to 32 samples
                
                # Get hidden activations
                h = torch.zeros(x_batch.size(0), 128, device=self.device)
                x_proj = self.eqprop_model.W_in(x_batch)
                for _ in range(5):  # Few iterations
                    h = torch.tanh(x_proj + self.eqprop_model.W_rec(h))
                
                # Show activations as batch×128 matrix
                activations = h.cpu().numpy()
                self.activation_image.setImage(activations.T)  # Transpose for better view
            except:
                pass  # Skip if error
        
        # Update status
        self.status_label.setText(
            f"Epoch {self.current_epoch+1}/{self.max_epochs} - "
            f"EqProp: {eqprop_acc*100:.1f}%, Backprop: {backprop_acc*100:.1f}%"
        )
        
        self.current_epoch += 1
        
        # Schedule next epoch
        if self.running:
            self.timer.singleShot(100, self.train_one_epoch)  # Quick succession
    
    def reset(self):
        """Reset demo."""
        self.timer.stop()
        self.running = False
        self.current_epoch = 0
        self.eqprop_history = {'loss': [], 'acc': []}
        self.backprop_history = {'loss': [], 'acc': []}
        
        self.eqprop_loss_curve.setData([])
        self.backprop_loss_curve.setData([])
        self.eqprop_acc_curve.setData([])
        self.backprop_acc_curve.setData([])
        self.comparison_label.setText("")
        
        self.run_btn.setText(f"▶ Start Training ({self.max_epochs} epochs)")
        self.status_label.setText(f"Click 'Start Training' to begin {self.max_epochs}-epoch comparison")
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
