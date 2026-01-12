"""
EqProp-Torch Core Trainer

High-level API for training EqProp models with automatic acceleration,
checkpointing, and ONNX export.
"""

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .acceleration import compile_model, enable_tf32, get_optimal_backend
# Import kernel components globally if available to avoid import loops inside methods
try:
    from .kernel import HAS_CUPY, cross_entropy, to_numpy
    from .kernel import EqPropKernel as KernelEqPropKernel
except ImportError:
    HAS_CUPY = False
    KernelEqPropKernel = None
    cross_entropy = None
    to_numpy = lambda x: x


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

    Example:
        >>> from bioplausible import EqPropTrainer, LoopedMLP
        >>> model = LoopedMLP(784, 256, 10)
        >>> trainer = EqPropTrainer(model, use_compile=True)
        >>> trainer.fit(train_loader, epochs=10)
        >>> print(trainer.evaluate(test_loader))
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

        Raises:
            ValueError: If invalid optimizer or compile_mode
            RuntimeError: If use_kernel=True but model incompatible
        """
        # Enable TF32 by default for performance
        enable_tf32(allow_tf32)

        # Validate inputs
        self._validate_inputs(optimizer, compile_mode, lr, weight_decay)

        self.device = device or get_optimal_backend()
        self.use_kernel = use_kernel
        self._epoch = 0
        self._step = 0
        self._best_metric = float('inf')
        self._history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self._kernel = None
        self.optimizer = None # Initialized below

        # Move model to device
        self._setup_model(model, use_compile, compile_mode)

        # Kernel mode initialization attempt
        if self.use_kernel:
            try:
                self._init_kernel_mode()
            except Exception as e:
                warnings.warn(f"Kernel mode initialization failed: {e}. Falling back to PyTorch BPTT mode.", UserWarning)
                self.use_kernel = False

        # Create optimizer (only for PyTorch mode)
        # We do this AFTER potential kernel fallback
        if not self.use_kernel:
            self.optimizer = self._create_optimizer(optimizer, lr, weight_decay)

    def _validate_inputs(self, optimizer: str, compile_mode: str, lr: float, weight_decay: float) -> None:
        """Validate initialization parameters."""
        self._validate_optimizer(optimizer)
        self._validate_compile_mode(compile_mode)
        self._validate_lr(lr)
        self._validate_weight_decay(weight_decay)

    def _validate_optimizer(self, optimizer: str) -> None:
        """Validate optimizer name."""
        valid_optimizers = self._get_valid_optimizers()
        if optimizer not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer '{optimizer}'. Must be one of: {', '.join(valid_optimizers)}"
            )

    def _validate_compile_mode(self, compile_mode: str) -> None:
        """Validate compile mode."""
        valid_compile_modes = self._get_valid_compile_modes()
        if compile_mode not in valid_compile_modes:
            raise ValueError(
                f"Invalid compile_mode '{compile_mode}'. "
                f"Must be one of: {', '.join(valid_compile_modes)}"
            )

    def _validate_lr(self, lr: float) -> None:
        """Validate learning rate."""
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

    def _validate_weight_decay(self, weight_decay: float) -> None:
        """Validate weight decay."""
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")

    def _get_valid_optimizers(self) -> List[str]:
        """Return list of valid optimizers."""
        return ["adam", "adamw", "sgd"]

    def _get_valid_compile_modes(self) -> List[str]:
        """Return list of valid compile modes."""
        return ["default", "reduce-overhead", "max-autotune"]

    def _setup_model(self, model: nn.Module, use_compile: bool, compile_mode: str) -> None:
        """Setup model on device and apply compilation if requested."""
        try:
            self.model = model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to move model to device '{self.device}': {e}")

        # Apply torch.compile if requested
        if use_compile and not self.use_kernel:
            if not hasattr(torch, 'compile'):
                warnings.warn(
                    "torch.compile not available (requires PyTorch 2.0+). "
                    "Model will run without compilation.",
                    UserWarning
                )
            else:
                try:
                    self.model = compile_model(self.model, mode=compile_mode)
                except Exception as e:
                    warnings.warn(f"torch.compile failed: {e}. Using uncompiled model.", UserWarning)

    def _create_optimizer(self, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        """Create optimizer by name."""
        optimizer_factory = self._get_optimizer_factory(name, lr, weight_decay)
        if optimizer_factory is None:
            raise ValueError(f"Unknown optimizer: {name}. Use 'adam', 'adamw', or 'sgd'.")

        try:
            return optimizer_factory()
        except Exception as e:
            raise RuntimeError(f"Failed to create optimizer: {e}")

    def _get_optimizer_factory(self, name: str, lr: float, weight_decay: float) -> Optional[Callable[[], torch.optim.Optimizer]]:
        """Get optimizer factory function by name."""
        optimizer_factories = {
            "adam": lambda: torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            "adamw": lambda: torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            "sgd": lambda: torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        }
        return optimizer_factories.get(name)

    def _init_kernel_mode(self) -> None:
        """Initialize CuPy kernel for O(1) memory training."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available. Falling back to PyTorch.")

        # Extract model dimensions
        if hasattr(self.model, 'input_dim'):
            input_dim = self.model.input_dim
            hidden_dim = self.model.hidden_dim
            output_dim = self.model.output_dim
        else:
            raise RuntimeError("Model dimensions not detected. Kernel mode disabled.")

        self._kernel = KernelEqPropKernel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_gpu=True,
        )

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        log_interval: int = 100,
        checkpoint_path: Optional[str] = None,
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

        Returns:
            History dict with train/val losses and accuracies
        """
        # Validate loaders
        if not hasattr(train_loader, '__iter__') or isinstance(train_loader, str):
             raise ValueError("train_loader must be an iterable DataLoader (not a string)")
        if val_loader is not None and (not hasattr(val_loader, '__iter__') or isinstance(val_loader, str)):
             raise ValueError("val_loader must be an iterable DataLoader (not a string)")

        loss_fn = loss_fn or nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self._epoch = epoch + 1
            epoch_start = time.perf_counter()

            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, loss_fn, log_interval)
            self._history['train_loss'].append(train_loss)
            self._history['train_acc'].append(train_acc)

            # Validation phase
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, loss_fn)
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']
                self._history['val_loss'].append(val_loss)
                self._history['val_acc'].append(val_acc)

                # Checkpoint best model
                if checkpoint_path and val_loss < self._best_metric:
                    self._best_metric = val_loss
                    self.save_checkpoint(checkpoint_path)

            epoch_time = time.perf_counter() - epoch_start

            # Callback
            if callback:
                callback({
                    'epoch': self._epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'time': epoch_time,
                })

        return self._history

    def _train_epoch(
        self,
        loader: DataLoader,
        loss_fn: Callable,
        log_interval: int,
    ) -> Tuple[float, float]:
        """
        Run one training epoch.

        Args:
            loader: Training data loader
            loss_fn: Loss function to use
            log_interval: Print progress every N batches

        Returns:
            Average loss and accuracy for the epoch
        """
        if not self.use_kernel:
            self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(loader):
            try:
                # Prepare and Process batch
                if self.use_kernel:
                     batch_loss, batch_correct, batch_total = self._process_batch_kernel(x, y)
                else:
                     x, y = self._prepare_batch(x, y)
                     batch_loss, batch_correct, batch_total = self._process_batch_pytorch(x, y, loss_fn)

                # Update metrics
                total_loss += batch_loss
                correct += batch_correct
                total += batch_total
                self._step += 1

                # Log progress
                if log_interval > 0 and batch_idx % log_interval == 0:
                    avg_loss = batch_loss / batch_total if batch_total > 0 else 0
                    print(f'Batch {batch_idx}: Loss = {avg_loss:.4f}')

            except Exception as e:
                raise RuntimeError(f"Error processing batch {batch_idx}: {str(e)}")

        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

    def _prepare_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move to device and flatten input if necessary."""
        x, y = x.to(self.device), y.to(self.device)
        x = self._maybe_flatten_input(x)
        return x, y

    def _maybe_flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten input tensor if it's 4D and model has input_dim attribute."""
        if x.dim() == 4 and hasattr(self.model, 'input_dim'):
            return x.view(x.size(0), -1)
        return x

    def _process_batch(self, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> Tuple[float, int, int]:
        """Deprecated: Use _process_batch_pytorch or _process_batch_kernel instead."""
        return self._process_batch_pytorch(x, y, loss_fn)

    def _process_batch_pytorch(self, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> Tuple[float, int, int]:
        """Process a single batch using PyTorch BPTT."""
        self.optimizer.zero_grad()

        output = self.model(x)
        loss = loss_fn(output, y)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate metrics
        return self._calculate_batch_metrics(loss, output, y, x.size(0))

    def _process_batch_kernel(self, x: Any, y: Any) -> Tuple[float, int, int]:
        """Process a single batch using EqProp Kernel."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Flatten if needed
        if x.ndim == 4:
            x = x.reshape(x.shape[0], -1)

        metrics = self._kernel.train_step(x, y)

        loss = metrics['loss']
        accuracy = metrics['accuracy']
        batch_size = x.shape[0]
        correct = int(accuracy * batch_size)
        total_loss = loss * batch_size

        return total_loss, correct, batch_size

    def _calculate_batch_metrics(self, loss: torch.Tensor, output: torch.Tensor,
                                targets: torch.Tensor, batch_size: int) -> Tuple[float, int, int]:
        """Calculate loss, correct predictions, and batch size for a batch."""
        total_loss = loss.item() * batch_size
        _, predicted = output.max(1)
        correct = predicted.eq(targets).sum().item()
        return total_loss, correct, batch_size

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            loader: Data loader
            loss_fn: Loss function (default: CrossEntropyLoss)

        Returns:
            Dict with 'loss' and 'accuracy'
        """
        loss_fn = loss_fn or nn.CrossEntropyLoss()

        if not self.use_kernel:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        try:
            for batch_idx, (x, y) in enumerate(loader):
                try:
                    if self.use_kernel:
                        # Kernel Evaluation
                         if isinstance(x, torch.Tensor):
                            x = x.cpu().numpy()
                         if isinstance(y, torch.Tensor):
                            y = y.cpu().numpy()
                         if x.ndim == 4:
                            x = x.reshape(x.shape[0], -1)

                         batch_size = x.shape[0]

                         # Single forward pass for both loss and accuracy
                         # Solve equilibrium to get fixed point
                         h_star, _, _ = self._kernel.solve_equilibrium(x)
                         logits = self._kernel.compute_output(h_star)

                         # Calculate metrics using kernel utils
                         loss_val = cross_entropy(logits, y, self._kernel.xp)
                         batch_loss = float(to_numpy(loss_val)) * batch_size

                         # Calculate accuracy safely
                         preds = self._kernel.xp.argmax(logits, axis=1)
                         # Explicit conversion to numpy to handle potential CuPy vs NumPy issues
                         preds_np = to_numpy(preds)
                         y_np = to_numpy(y) if not isinstance(y, np.ndarray) else y
                         batch_correct = np.sum(preds_np == y_np)

                         total_loss += batch_loss
                         correct += batch_correct
                         total += batch_size

                    else:
                        # PyTorch Evaluation
                        x, y = self._prepare_batch(x, y)
                        output = self.model(x)
                        loss = loss_fn(output, y)

                        batch_loss, batch_correct, batch_total = self._calculate_batch_metrics(loss, output, y, x.size(0))

                        total_loss += batch_loss
                        correct += batch_correct
                        total += batch_total

                except Exception as e:
                    print(f"Warning: Error processing evaluation batch {batch_idx}: {str(e)}. Skipping...")
                    continue

        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

        return {
            'loss': total_loss / total if total > 0 else float('inf'),
            'accuracy': correct / total if total > 0 else 0.0,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        try:
            checkpoint = {
                'epoch': self._epoch,
                'step': self._step,
                'best_metric': self._best_metric,
                'history': self._history,
                'use_kernel': self.use_kernel,
            }

            if self.use_kernel:
                # Save kernel weights
                checkpoint['kernel_weights'] = self._kernel.weights
                checkpoint['kernel_biases'] = self._kernel.biases
                checkpoint['kernel_sn_state'] = self._kernel.sn_state
                checkpoint['kernel_adam_state'] = self._kernel.adam_state
            else:
                # Handle compiled models
                model_to_save = self.model
                if hasattr(self.model, '_orig_mod'):
                    model_to_save = self.model._orig_mod

                checkpoint['model_state_dict'] = model_to_save.state_dict()
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

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

        self._epoch = checkpoint['epoch']
        self._step = checkpoint['step']
        self._best_metric = checkpoint.get('best_metric', float('inf'))
        self._history = checkpoint.get('history', self._history)

        # Check if mode matches
        if checkpoint.get('use_kernel', False) != self.use_kernel:
            warnings.warn("Checkpoint mode (kernel vs torch) does not match current trainer mode.", UserWarning)

        try:
            if self.use_kernel and 'kernel_weights' in checkpoint:
                 self._kernel.weights = checkpoint['kernel_weights']
                 self._kernel.biases = checkpoint['kernel_biases']
                 self._kernel.sn_state = checkpoint.get('kernel_sn_state', self._kernel.sn_state)
                 self._kernel.adam_state = checkpoint.get('kernel_adam_state', self._kernel.adam_state)
            elif not self.use_kernel and 'model_state_dict' in checkpoint:
                # Handle compiled models
                model_to_load = self.model
                if hasattr(self.model, '_orig_mod'):
                    model_to_load = self.model._orig_mod

                model_to_load.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except KeyError as e:
            raise ValueError(f"Checkpoint file missing required key: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model state from checkpoint: {str(e)}")

    def export_onnx(
        self,
        path: str,
        input_shape: Tuple[int, ...],
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> None:
        """
        Export model to ONNX format for deployment.

        Note: Only supported in PyTorch mode.

        Args:
            path: Output path (.onnx)
            input_shape: Example input shape, e.g. (1, 784)
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axis specification (optional)
        """
        if self.use_kernel:
             warnings.warn("ONNX export is not supported in Kernel mode.", UserWarning)
             return

        try:
            # Get uncompiled model
            model = self.model
            if hasattr(self.model, '_orig_mod'):
                model = self.model._orig_mod

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
        """Return training history."""
        return self._history

    @property
    def current_epoch(self) -> int:
        """Return current epoch number."""
        return self._epoch

    def compute_lipschitz(self) -> float:
        """
        Compute Lipschitz constant if model supports it.

        Returns:
            Lipschitz constant L (or 0.0 if not supported)
        """
        if self.use_kernel:
            # In kernel mode, we can compute it from weights if supported
            # But the kernel API doesn't expose it easily yet.
            return 0.0

        if hasattr(self.model, 'compute_lipschitz'):
            return self.model.compute_lipschitz()

        # Try to find underlying model (e.g. if compiled)
        if hasattr(self.model, '_orig_mod') and hasattr(self.model._orig_mod, 'compute_lipschitz'):
            return self.model._orig_mod.compute_lipschitz()

        return 0.0


__all__ = ['EqPropTrainer']
