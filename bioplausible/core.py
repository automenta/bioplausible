"""
EqProp-Torch Core Trainer

High-level API for training EqProp models with automatic acceleration,
checkpointing, and ONNX export.
"""

import time
import warnings
import os
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Optional tqdm for progress bars
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Mock tqdm if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .acceleration import compile_model, enable_tf32, get_optimal_backend

# Import kernel components globally if available to avoid import loops inside methods
try:
    from .kernel import HAS_CUPY, cross_entropy, to_numpy
    from .kernel import EqPropKernel as KernelEqPropKernel
except ImportError:
    HAS_CUPY = False
    KernelEqPropKernel = None
    cross_entropy = None
    to_numpy = lambda x: x.cpu().numpy() if hasattr(x, 'cpu') else x


class EqPropTrainer:
    """
    High-level trainer for Equilibrium Propagation models.

    Supports two modes:
    1. PyTorch Autograd (BPTT): Uses standard PyTorch backpropagation through time.
       This is accurate but memory intensive O(T).
    2. Kernel Mode (EqProp): Uses custom NumPy/CuPy kernel for Equilibrium Propagation.
       This is memory efficient O(1) but requires a compatible kernel implementation.

    Features:
        - Automatic torch.compile for 2-3x speedup (PyTorch mode)
        - Optional CuPy kernel mode for O(1) memory
        - Checkpoint saving/loading
        - ONNX export for deployment
        - Progress callbacks
        - Learning Rate Scheduling
        - Gradient Clipping
        - Automatic Mixed Precision (AMP)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: str = "adam",
        lr: float = 0.001,
        weight_decay: float = 0.0,
        use_compile: bool = True,
        use_kernel: bool = False,
        device: Optional[str] = None,
        compile_mode: str = "reduce-overhead",
        allow_tf32: bool = True,
        use_amp: bool = False,
    ) -> None:
        """
        Initialize the EqProp trainer.

        Args:
            model: EqProp model (LoopedMLP, ConvEqProp, TransformerEqProp, etc.)
            optimizer: Optimizer name ('adam', 'adamw', 'sgd')
            lr: Learning rate
            weight_decay: L2 regularization
            use_compile: If True, wrap model with torch.compile
            use_kernel: If True, use CuPy kernel (NVIDIA only, O(1) memory)
            device: Device to train on (auto-detected if None)
            compile_mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune')
            allow_tf32: If True, enable TensorFloat-32 on Ampere+ GPUs (default: True)
            use_amp: If True, use Automatic Mixed Precision (AMP) for training
        """
        # Enable TF32 by default for performance
        enable_tf32(allow_tf32)

        # Validate inputs
        self._validate_inputs(optimizer, compile_mode, lr, weight_decay)

        self.device = device or get_optimal_backend()
        self.use_kernel = use_kernel
        self.use_amp = use_amp
        self._epoch = 0
        self._step = 0
        self._best_metric = float('inf')
        self._history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self._kernel = None
        self.optimizer = None
        self.scaler = None

        # Move model to device
        self._setup_model(model, use_compile, compile_mode)

        # Kernel mode initialization attempt
        if self.use_kernel:
            try:
                self._init_kernel_mode()
            except Exception as e:
                warnings.warn(f"Kernel mode initialization failed: {e}. Falling back to PyTorch BPTT mode.", UserWarning)
                self.use_kernel = False

        if self.use_amp and self.use_kernel:
            warnings.warn("AMP is not supported in Kernel mode. Ignoring use_amp=True.", UserWarning)
            self.use_amp = False

        # Create optimizer and scaler (only for PyTorch mode)
        if not self.use_kernel:
            self.optimizer = self._create_optimizer(optimizer, lr, weight_decay)
            if self.use_amp:
                self.scaler = torch.amp.GradScaler(self.device if str(self.device).startswith('cuda') else 'cpu')

    def _validate_inputs(self, optimizer: str, compile_mode: str, lr: float, weight_decay: float) -> None:
        """Validate initialization parameters."""
        valid_optimizers = ["adam", "adamw", "sgd"]
        if optimizer not in valid_optimizers:
            raise ValueError(f"Invalid optimizer '{optimizer}'. Must be one of: {', '.join(valid_optimizers)}")

        valid_compile_modes = ["default", "reduce-overhead", "max-autotune"]
        if compile_mode not in valid_compile_modes:
            raise ValueError(f"Invalid compile_mode '{compile_mode}'. Must be one of: {', '.join(valid_compile_modes)}")

        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")

    def _setup_model(self, model: nn.Module, use_compile: bool, compile_mode: str) -> None:
        """Setup model on device and apply compilation if requested."""
        try:
            self.model = model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to move model to device '{self.device}': {e}")

        # Apply torch.compile if requested
        if use_compile and not self.use_kernel:
            if not hasattr(torch, 'compile'):
                warnings.warn("torch.compile not available. Model will run without compilation.", UserWarning)
            else:
                try:
                    self.model = compile_model(self.model, mode=compile_mode)
                except Exception as e:
                    warnings.warn(f"torch.compile failed: {e}. Using uncompiled model.", UserWarning)

    def _create_optimizer(self, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        """Create optimizer by name."""
        factories = {
            "adam": lambda: torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            "adamw": lambda: torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            "sgd": lambda: torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        }
        try:
            return factories[name]()
        except Exception as e:
            raise RuntimeError(f"Failed to create optimizer: {e}")

    def _init_kernel_mode(self) -> None:
        """Initialize CuPy kernel for O(1) memory training."""
        # Allow fallback to NumPy if CuPy is missing, handled by kernel internal logic
        # if not HAS_CUPY:
        #    raise RuntimeError("CuPy not available.")

        if hasattr(self.model, 'input_dim'):
            dims = (self.model.input_dim, self.model.hidden_dim, self.model.output_dim)
        else:
            raise RuntimeError("Model dimensions not detected. Kernel mode disabled.")

        # Pass use_gpu=True only if CuPy is available, otherwise False for NumPy fallback
        self._kernel = KernelEqPropKernel(*dims, use_gpu=HAS_CUPY)

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_interval: int = 100,
        checkpoint_path: Optional[str] = None,
        progress_bar: bool = True,
        scheduler: Optional[Any] = None,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            epochs: Number of epochs
            val_loader: Validation data loader (optional)
            loss_fn: Loss function (default: CrossEntropyLoss)
            callback: Called after each epoch with metrics dict
            log_interval: Print progress every N batches
            checkpoint_path: Save best checkpoint to this path
            progress_bar: Show tqdm progress bar
            scheduler: Learning rate scheduler (e.g. torch.optim.lr_scheduler.StepLR)
            max_grad_norm: Gradient clipping norm (default: None)

        Returns:
            History dict with train/val losses and accuracies
        """
        self._validate_loader(train_loader, "train_loader")
        if val_loader:
             self._validate_loader(val_loader, "val_loader")

        if scheduler and self.use_kernel:
             warnings.warn("Learning rate scheduler is not supported in Kernel mode.", UserWarning)

        loss_fn = loss_fn or nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self._epoch = epoch + 1
            epoch_start = time.perf_counter()

            # Training phase
            train_loss, train_acc = self._run_epoch(
                train_loader, loss_fn, is_training=True,
                log_interval=log_interval, progress_bar=progress_bar,
                desc=f"Epoch {self._epoch}/{epochs} [Train]",
                max_grad_norm=max_grad_norm
            )
            self._history['train_loss'].append(train_loss)
            self._history['train_acc'].append(train_acc)

            # Step scheduler if it's an epoch-based scheduler
            if scheduler and not self.use_kernel:
                try:
                    scheduler.step()
                except Exception as e:
                    warnings.warn(f"Scheduler step failed: {e}", RuntimeWarning)

            # Validation phase
            val_loss, val_acc = None, None
            if val_loader:
                val_metrics = self.evaluate(val_loader, loss_fn, progress_bar=False)
                val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']
                self._history['val_loss'].append(val_loss)
                self._history['val_acc'].append(val_acc)

                if checkpoint_path and val_loss < self._best_metric:
                    self._best_metric = val_loss
                    self.save_checkpoint(checkpoint_path)

            epoch_time = time.perf_counter() - epoch_start

            # Print epoch summary
            if (not progress_bar or not HAS_TQDM) and log_interval > 0:
                print(f"Epoch {self._epoch}/{epochs}: "
                      f"Train Loss={train_loss:.4f} Acc={train_acc:.2%}"
                      + (f" | Val Loss={val_loss:.4f} Acc={val_acc:.2%}" if val_loss is not None else ""))

            if callback:
                callback({
                    'epoch': self._epoch,
                    'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc,
                    'time': epoch_time,
                })

        return self._history

    def _validate_loader(self, loader: Any, name: str) -> None:
        if not hasattr(loader, '__iter__') or isinstance(loader, str):
             raise ValueError(f"{name} must be an iterable DataLoader (not a string)")

    def _run_epoch(
        self,
        loader: DataLoader,
        loss_fn: Callable,
        is_training: bool,
        log_interval: int = 0,
        progress_bar: bool = False,
        desc: str = "",
        max_grad_norm: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Unified epoch runner for both training and evaluation."""
        if is_training and not self.use_kernel:
            self.model.train()
        elif not self.use_kernel:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Select context manager: no_grad() for PyTorch eval, nullcontext() otherwise
        context = torch.no_grad() if (not is_training and not self.use_kernel) else nullcontext()

        # Wrap loader with tqdm if requested
        use_tqdm = progress_bar and HAS_TQDM
        iterator = tqdm(loader, desc=desc, leave=False) if use_tqdm else loader

        with context:
            for batch_idx, (x, y) in enumerate(iterator):
                try:
                    if self.use_kernel:
                        loss, batch_correct, batch_size = self._process_batch_kernel(x, y, is_training)
                    elif hasattr(self.model, 'train_step') and is_training:
                        # Delegate to model's custom training step (e.g. for Algorithms)
                        loss, batch_correct, batch_size = self._process_batch_custom(x, y, is_training)
                    else:
                        loss, batch_correct, batch_size = self._process_batch_pytorch(
                            x, y, loss_fn, is_training, max_grad_norm
                        )

                    total_loss += loss
                    correct += batch_correct
                    total += batch_size

                    if is_training:
                        self._step += 1

                    # Update progress bar
                    if use_tqdm and isinstance(iterator, tqdm):
                        avg_loss = total_loss / total if total > 0 else 0
                        avg_acc = correct / total if total > 0 else 0
                        iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2%}")

                    # Log to console if not using progress bar
                    elif is_training and log_interval > 0 and batch_idx % log_interval == 0:
                        avg_loss = total_loss / total if total > 0 else 0
                        print(f'Batch {batch_idx}: Loss = {avg_loss:.4f}')

                except Exception as e:
                    stage = "training" if is_training else "evaluation"
                    mode = "kernel" if self.use_kernel else "PyTorch"
                    raise RuntimeError(f"Error processing {mode} {stage} batch {batch_idx}: {str(e)}")

        avg_loss = total_loss / total if total > 0 else float('inf')
        avg_acc = correct / total if total > 0 else 0.0
        return avg_loss, avg_acc

    def _process_batch_custom(
        self, x: torch.Tensor, y: torch.Tensor, is_training: bool
    ) -> Tuple[float, int, int]:
        """Process a single batch using model's custom train_step."""
        x, y = x.to(self.device), y.to(self.device)

        # Flatten if needed, though algorithms might handle it
        if x.dim() == 4 and hasattr(self.model, 'input_dim'):
             x = x.view(x.size(0), -1)

        metrics = self.model.train_step(x, y)

        batch_size = x.size(0)
        # Handle potential missing keys or different names
        loss = metrics.get('loss', 0.0)
        acc = metrics.get('accuracy', 0.0)

        return loss * batch_size, int(acc * batch_size), batch_size

    def _process_batch_pytorch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        is_training: bool,
        max_grad_norm: Optional[float] = None
    ) -> Tuple[float, int, int]:
        """Process a single batch using PyTorch, optionally with AMP and clipping."""
        x, y = x.to(self.device), y.to(self.device)

        # Flatten input if necessary
        if x.dim() == 4 and hasattr(self.model, 'input_dim'):
            x = x.view(x.size(0), -1)

        if is_training:
            self.optimizer.zero_grad()

            # Use AMP if enabled
            if self.use_amp:
                device_type = 'cuda' if str(self.device).startswith('cuda') else 'cpu'
                with torch.amp.autocast(device_type=device_type):
                    output = self.model(x)
                    loss = loss_fn(output, y)

                self.scaler.scale(loss).backward()

                if max_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(x)
                loss = loss_fn(output, y)
                loss.backward()

                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()
        else:
            # Eval mode - no need for scaler
            if self.use_amp:
                device_type = 'cuda' if str(self.device).startswith('cuda') else 'cpu'
                with torch.amp.autocast(device_type=device_type):
                    output = self.model(x)
                    loss = loss_fn(output, y)
            else:
                output = self.model(x)
                loss = loss_fn(output, y)

        total_loss = loss.item() * x.size(0)
        _, predicted = output.max(1)
        correct = predicted.eq(y).sum().item()

        return total_loss, correct, x.size(0)

    def _process_batch_kernel(
        self, x: Any, y: Any, is_training: bool
    ) -> Tuple[float, int, int]:
        """Process a single batch using EqProp Kernel."""
        if isinstance(x, torch.Tensor): x = x.cpu().numpy()
        if isinstance(y, torch.Tensor): y = y.cpu().numpy()
        if x.ndim == 4: x = x.reshape(x.shape[0], -1)

        if is_training:
            metrics = self._kernel.train_step(x, y)
        else:
            metrics = self._kernel.evaluate(x, y)

        batch_size = x.shape[0]
        total_loss = metrics['loss'] * batch_size
        correct = int(metrics['accuracy'] * batch_size)

        return total_loss, correct, batch_size

    def evaluate(
        self,
        loader: DataLoader,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        progress_bar: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            loader: Data loader
            loss_fn: Loss function (default: CrossEntropyLoss)
            progress_bar: Show tqdm progress bar

        Returns:
            Dict with 'loss' and 'accuracy'
        """
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        loss, acc = self._run_epoch(loader, loss_fn, is_training=False, progress_bar=progress_bar, desc="Evaluating")
        return {'loss': loss, 'accuracy': acc}

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        try:
            checkpoint = {
                'epoch': self._epoch,
                'step': self._step,
                'best_metric': self._best_metric,
                'history': self._history,
                'use_kernel': self.use_kernel,
                'use_amp': self.use_amp,
            }

            if self.use_kernel:
                checkpoint.update({
                    'kernel_weights': self._kernel.weights,
                    'kernel_biases': self._kernel.biases,
                    'kernel_sn_state': self._kernel.sn_state,
                    'kernel_adam_state': self._kernel.adam_state
                })
            else:
                model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
                checkpoint.update({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                })

            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()

            torch.save(checkpoint, path)
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint to {path}: {str(e)}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {str(e)}")

        self._epoch = checkpoint.get('epoch', 0)
        self._step = checkpoint.get('step', 0)
        self._best_metric = checkpoint.get('best_metric', float('inf'))
        self._history = checkpoint.get('history', self._history)

        if checkpoint.get('use_kernel', False) != self.use_kernel:
            warnings.warn("Checkpoint mode mismatch (kernel vs torch).", UserWarning)

        try:
            if self.use_kernel and 'kernel_weights' in checkpoint:
                 self._kernel.weights = checkpoint['kernel_weights']
                 self._kernel.biases = checkpoint['kernel_biases']
                 self._kernel.sn_state = checkpoint.get('kernel_sn_state', self._kernel.sn_state)
                 self._kernel.adam_state = checkpoint.get('kernel_adam_state', self._kernel.adam_state)
            elif not self.use_kernel and 'model_state_dict' in checkpoint:
                model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
                model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except KeyError as e:
            raise ValueError(f"Checkpoint missing required key: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {e}")

    def export_onnx(
        self,
        path: str,
        input_shape: Tuple[int, ...],
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> None:
        """
        Export model to ONNX format.

        Note: Only supported in PyTorch mode.
        """
        if self.use_kernel:
             warnings.warn("ONNX export is not supported in Kernel mode.", UserWarning)
             return

        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        try:
            model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
            model.eval()
            dummy_input = torch.randn(*input_shape, device=self.device)
            dynamic_axes = dynamic_axes or {'input': {0: 'batch'}, 'output': {0: 'batch'}}

            torch.onnx.export(
                model,
                dummy_input,
                path,
                opset_version=opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                export_params=True,
            )
        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {str(e)}")

    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history

    @property
    def current_epoch(self) -> int:
        return self._epoch

    def compute_lipschitz(self) -> float:
        """Compute Lipschitz constant if model supports it."""
        if self.use_kernel:
            return 0.0

        model = self.model
        if hasattr(model, 'compute_lipschitz'):
            return model.compute_lipschitz()

        # Check wrapped model
        if hasattr(model, '_orig_mod') and hasattr(model._orig_mod, 'compute_lipschitz'):
            return model._orig_mod.compute_lipschitz()

        return 0.0


__all__ = ['EqPropTrainer']
