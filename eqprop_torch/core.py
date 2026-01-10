"""
EqProp-Torch Core Trainer

High-level API for training EqProp models with automatic acceleration,
checkpointing, and ONNX export.
"""

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .acceleration import compile_model, enable_tf32, get_optimal_backend


class EqPropTrainer:
    """
    High-level trainer for Equilibrium Propagation models.

    Features:
        - Automatic torch.compile for 2-3x speedup
        - Optional CuPy kernel mode for O(1) memory
        - Checkpoint saving/loading
        - ONNX export for deployment
        - Progress callbacks

    Example:
        >>> from eqprop_torch import EqPropTrainer, LoopedMLP
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

        # Move model to device
        self._setup_model(model, use_compile, compile_mode)

        # Create optimizer
        self.optimizer = self._create_optimizer(optimizer, lr, weight_decay)

        # Kernel mode initialization
        if use_kernel:
            self._init_kernel_mode_safe()

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

    def _init_kernel_mode_safe(self) -> None:
        """Initialize CuPy kernel for O(1) memory training with error handling."""
        try:
            self._init_kernel_mode()
        except Exception as e:
            warnings.warn(f"Kernel mode initialization failed: {e}", UserWarning)
            self.use_kernel = False

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
        from .kernel import HAS_CUPY
        from .kernel import EqPropKernel as KernelEqPropKernel

        if not HAS_CUPY:
            warnings.warn(
                "CuPy not available. Falling back to PyTorch. "
                "Install CuPy for kernel mode: pip install cupy-cuda12x",
                RuntimeWarning
            )
            self.use_kernel = False
            return

        # Extract model dimensions
        if hasattr(self.model, 'input_dim'):
            input_dim = self.model.input_dim
            hidden_dim = self.model.hidden_dim
            output_dim = self.model.output_dim
        else:
            warnings.warn("Model dimensions not detected. Kernel mode disabled.", RuntimeWarning)
            self.use_kernel = False
            return

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
        loss_fn = loss_fn or nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self._epoch = epoch + 1
            epoch_start = time.time()

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

            epoch_time = time.time() - epoch_start

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
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(loader):
            try:
                batch_result = self._process_training_batch(x, y, loss_fn, batch_idx, log_interval)

                # Track metrics
                total_loss += batch_result['loss']
                correct += batch_result['correct']
                total += batch_result['total']
                self._step += 1

            except Exception as e:
                raise RuntimeError(f"Error processing batch {batch_idx}: {str(e)}")

        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

    def _process_training_batch(self, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable,
                              batch_idx: int, log_interval: int) -> Dict[str, float]:
        """Process a single training batch and return metrics."""
        x, y = self._move_to_device(x, y)

        # Flatten images if needed
        x = self._maybe_flatten_input(x)

        # Process batch
        batch_loss, batch_correct, batch_total = self._process_batch(x, y, loss_fn)

        # Log progress
        if log_interval > 0 and batch_idx % log_interval == 0:
            print(f'Batch {batch_idx}: Loss = {batch_loss / batch_total:.4f}')

        return {
            'loss': batch_loss,
            'correct': batch_correct,
            'total': batch_total
        }

    def _move_to_device(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move tensors to the trainer's device.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Tuple of tensors moved to the trainer's device
        """
        return x.to(self.device), y.to(self.device)

    def _maybe_flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten input tensor if it's 4D and model has input_dim attribute."""
        if x.dim() == 4 and hasattr(self.model, 'input_dim'):
            return x.view(x.size(0), -1)
        return x

    def _process_batch(self, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> Tuple[float, int, int]:
        """Process a single batch and return loss, correct count, and total count."""
        self.optimizer.zero_grad()

        output = self.model(x)
        loss = loss_fn(output, y)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate metrics
        return self._calculate_batch_metrics(loss, output, y, x.size(0))

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
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        try:
            for batch_idx, (x, y) in enumerate(loader):
                try:
                    batch_metrics = self._evaluate_batch(x, y, loss_fn)
                    total_loss += batch_metrics['loss']
                    correct += batch_metrics['correct']
                    total += batch_metrics['total']

                except Exception as e:
                    print(f"Warning: Error processing evaluation batch {batch_idx}: {str(e)}. Skipping...")
                    continue

        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

        return {
            'loss': total_loss / total if total > 0 else float('inf'),
            'accuracy': correct / total if total > 0 else 0.0,
        }

    def _evaluate_batch(self, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable) -> Dict[str, float]:
        """Evaluate a single batch and return metrics."""
        x, y = self._move_to_device(x, y)
        x = self._maybe_flatten_input(x)

        output = self.model(x)
        loss = loss_fn(output, y)

        batch_size = x.size(0)
        total_loss = loss.item() * batch_size
        _, predicted = output.max(1)
        correct = predicted.eq(y).sum().item()

        return {
            'loss': total_loss,
            'correct': correct,
            'total': batch_size
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        try:
            # Handle compiled models
            model_to_save = self.model
            if hasattr(self.model, '_orig_mod'):
                model_to_save = self.model._orig_mod

            checkpoint = {
                'epoch': self._epoch,
                'step': self._step,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric': self._best_metric,
                'history': self._history,
            }

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

        # Handle compiled models
        model_to_load = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_load = self.model._orig_mod

        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._epoch = checkpoint['epoch']
            self._step = checkpoint['step']
            self._best_metric = checkpoint.get('best_metric', float('inf'))
            self._history = checkpoint.get('history', self._history)
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

        Args:
            path: Output path (.onnx)
            input_shape: Example input shape, e.g. (1, 784)
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axis specification (optional)
        """
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
        if hasattr(self.model, 'compute_lipschitz'):
            return self.model.compute_lipschitz()

        # Try to find underlying model (e.g. if compiled)
        if hasattr(self.model, '_orig_mod') and hasattr(self.model._orig_mod, 'compute_lipschitz'):
            return self.model._orig_mod.compute_lipschitz()

        return 0.0


__all__ = ['EqPropTrainer']
