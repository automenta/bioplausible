import torch
import torch.nn as nn
import time
import numpy as np
import warnings
from typing import Dict, Any, Optional

from bioplausible.training.base import BaseTrainer
from bioplausible.models.hebbian_chain import DeepHebbianChain
from bioplausible.acceleration import compile_model, enable_tf32, get_optimal_backend

# Optional imports for Kernel mode
try:
    from bioplausible.kernel import HAS_CUPY, cross_entropy, to_numpy
    from bioplausible.kernel import EqPropKernel as KernelEqPropKernel
except ImportError:
    HAS_CUPY = False
    KernelEqPropKernel = None

class SupervisedTrainer(BaseTrainer):
    """
    Trainer for Supervised Learning (LM, Vision).
    Combines simplicity of ExperimentAlgorithm with power of EqPropTrainer.
    """

    def __init__(
        self,
        model: nn.Module,
        task: Optional[Any] = None,  # BaseTask, optional
        device: str = "cpu",
        lr: float = 0.001,
        batches_per_epoch: int = 100,
        eval_batches: int = 20,
        steps: int = 20, # EqProp steps
        use_compile: bool = True,
        use_kernel: bool = False,
        compile_mode: str = "reduce-overhead",
        task_type: str = "vision", # Fallback task type
        **kwargs
    ):
        super().__init__(model, device)
        self.task = task
        self.task_type = task.task_type if task else task_type
        self.batches_per_epoch = batches_per_epoch
        self.eval_batches = eval_batches
        self.steps = steps
        self.use_kernel = use_kernel
        self.kernel = None

        # Check for embeddings
        self.has_embed = getattr(model, 'has_embed', False)
        self.embed = getattr(model, 'embed', None)

        # Setup model compilation
        if use_compile and not self.use_kernel:
            try:
                self.model = compile_model(self.model, mode=compile_mode)
            except Exception as e:
                warnings.warn(f"Compilation failed: {e}")

        # Kernel Initialization
        if self.use_kernel:
            if hasattr(self.model, "input_dim"):
                dims = (self.model.input_dim, self.model.hidden_dim, self.model.output_dim)
                # Pass use_gpu=True only if CuPy is available
                self.kernel = KernelEqPropKernel(*dims, use_gpu=HAS_CUPY)
            else:
                warnings.warn("Model dimensions not detected. Kernel mode disabled.")
                self.use_kernel = False

        # Optimizer (PyTorch mode only)
        if not self.use_kernel:
            if not hasattr(self.model, 'optimizer'):
                params = list(self.model.parameters())
                if self.has_embed and self.embed:
                    params.extend(list(self.embed.parameters()))
                self.opt = torch.optim.Adam(params, lr=lr)
            else:
                self.opt = None # Model manages optimizer

        self.criterion = nn.CrossEntropyLoss()

        # Handle Hebbian-specific updates
        if isinstance(self.model, DeepHebbianChain):
            if 'hebbian_lr' in kwargs:
                self.model.hebbian_lr = kwargs['hebbian_lr']
            if 'use_oja' in kwargs:
                self.model.use_oja = kwargs['use_oja']
                for layer in self.model.chain:
                    if hasattr(layer, 'original_layer'): # If spectral normed
                        layer.original_layer.use_oja = kwargs['use_oja']
                        layer.original_layer.learning_rate = self.model.hebbian_lr
                    else:
                        layer.use_oja = kwargs['use_oja']
                        layer.learning_rate = self.model.hebbian_lr

    def _prepare_input(self, x):
        """Prepare input tensor (embedding, flattening, etc.)."""
        # If Kernel mode, return flattened numpy/cupy array
        if self.use_kernel:
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy() # Kernel handles transfer if GPU
            if x.ndim == 4:
                x = x.reshape(x.shape[0], -1)
            return x

        if self.has_embed:
            return self.embed(x).mean(dim=1)
        else:
            # Vision or direct input
            if self.task_type in ["vision", "rl"]:
                # Check for Conv models (ModernConvEqProp)
                is_conv = "Conv" in self.model.__class__.__name__
                if hasattr(self.model, 'config') and self.model.config and hasattr(self.model.config, 'name'):
                     if "Conv" in self.model.config.name:
                          is_conv = True

                # Unwrap model if compiled
                if hasattr(self.model, '_orig_mod'):
                     orig = self.model._orig_mod
                     if "Conv" in orig.__class__.__name__:
                          is_conv = True

                if is_conv:
                     return x
                elif x.dim() > 2:
                     return x.view(x.size(0), -1)
                else:
                     return x
            else:
                 return x

    def get_dynamics(self, x, return_trajectory=True):
        """
        Run the model in inference mode and return internal dynamics.
        Useful for studying convergence, fixed points, and stability.
        """
        self.model.eval()
        x = x.to(self.device)
        h = self._prepare_input(x)

        if hasattr(self.model, "forward"):
            # Try to call forward with dynamics args
            try:
                # Assuming EqPropModel signature
                result = self.model(h, return_trajectory=return_trajectory, return_dynamics=True)
                # Result could be (out, traj) or (out, dynamics_dict)
                return result
            except TypeError:
                # Fallback if model doesn't support these args
                return self.model(h)
        else:
            return self.model(h)

    def train_batch(self, x, y) -> Dict[str, float]:
        """Run a single training step."""

        # Kernel Mode Branch
        if self.use_kernel:
            x_np = self._prepare_input(x)
            y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

            metrics = self.kernel.train_step(x_np, y_np)
            return metrics # returns {'loss': ..., 'accuracy': ...}

        # PyTorch Mode Branch
        self.model.train()
        if self.opt:
            self.opt.zero_grad()

        h = self._prepare_input(x)

        # Check for custom train_step (BioModel)
        metrics = None
        if hasattr(self.model, "train_step"):
            metrics = self.model.train_step(h, y)

        if metrics is not None:
            loss = metrics.get("loss", 0.0)
            acc = metrics.get("accuracy", 0.0)
        else:
            # Standard forward/backward
            if hasattr(self.model, "eq_steps"):
                logits = self.model(h, steps=self.steps)
            else:
                logits = self.model(h)

            if logits.dim() == 3 and self.task_type == "lm":
                # logits: [B, T, V] -> [B, V] (last token)
                logits = logits[:, -1, :]

            loss = self.criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.opt:
                self.opt.step()

            # Compute accuracy (detached)
            with torch.no_grad():
                if self.task_type in ["lm", "vision"]:
                    acc = (logits.argmax(1) == y).float().mean().item()
                else:
                    acc = 0.0

            loss = loss.item()

        return {"loss": loss, "accuracy": acc}

    def evaluate(self) -> Dict[str, float]:
        """Run validation loop."""
        if not self.task:
            raise RuntimeError("Task not provided. Cannot run standard evaluation loop.")

        if not self.use_kernel:
            self.model.eval()

        val_losses = []
        val_accs = []

        # No grad context for PyTorch mode
        context = torch.no_grad() if not self.use_kernel else torch.utils.contextlib.nullcontext()

        with context:
            for _ in range(self.eval_batches):
                x, y = self.task.get_batch("val")

                if self.use_kernel:
                    x_np = self._prepare_input(x)
                    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                    metrics = self.kernel.evaluate(x_np, y_np)
                    val_losses.append(metrics["loss"])
                    val_accs.append(metrics["accuracy"])
                else:
                    h = self._prepare_input(x)

                    if hasattr(self.model, "eq_steps"):
                        logits = self.model(h, steps=self.steps)
                    else:
                        logits = self.model(h)

                    if logits.dim() == 3 and self.task_type == "lm":
                        logits = logits[:, -1, :]

                    loss = self.criterion(logits, y)
                    metrics = self.task.compute_metrics(logits, y, loss.item())

                    val_losses.append(metrics["loss"])
                    val_accs.append(metrics.get("accuracy", 0.0))

        avg_loss = np.mean(val_losses) if val_losses else 0.0
        avg_acc = np.mean(val_accs) if val_accs else 0.0

        return {
            "val_loss": avg_loss,
            "val_accuracy": avg_acc,
            "val_perplexity": np.exp(min(avg_loss, 10)) if self.task_type == "lm" else 0.0
        }

    def train_epoch(self) -> Dict[str, float]:
        """Run full training epoch (train + eval)."""
        if not self.task:
             raise RuntimeError("Task not provided. Cannot run train_epoch. Use train_batch in your own loop.")

        t0 = time.time()

        # Training
        train_losses = []
        for _ in range(self.batches_per_epoch):
            x, y = self.task.get_batch("train")
            metrics = self.train_batch(x, y)
            train_losses.append(metrics["loss"])

        # Evaluation
        eval_metrics = self.evaluate()

        epoch_time = time.time() - t0

        return {
            "loss": eval_metrics["val_loss"],
            "accuracy": eval_metrics["val_accuracy"],
            "perplexity": eval_metrics["val_perplexity"],
            "time": epoch_time,
            "iteration_time": epoch_time / self.batches_per_epoch
        }

    def evaluate_loader(self, loader) -> Dict[str, float]:
        """Evaluate on a DataLoader."""
        if not self.use_kernel:
            self.model.eval()

        losses = []
        accs = []

        context = torch.no_grad() if not self.use_kernel else torch.utils.contextlib.nullcontext()

        with context:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                if self.use_kernel:
                    x_np = self._prepare_input(x)
                    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                    metrics = self.kernel.evaluate(x_np, y_np)
                    losses.append(metrics["loss"])
                    accs.append(metrics["accuracy"])
                else:
                    h = self._prepare_input(x)
                    if hasattr(self.model, "eq_steps"):
                        logits = self.model(h, steps=self.steps)
                    else:
                        logits = self.model(h)

                    if logits.dim() == 3 and self.task_type == "lm":
                        logits = logits[:, -1, :]

                    loss = self.criterion(logits, y)

                    # Compute accuracy
                    if self.task_type in ["lm", "vision"]:
                        acc = (logits.argmax(1) == y).float().mean().item()
                    else:
                        acc = 0.0

                    losses.append(loss.item())
                    accs.append(acc)

        return {
            "loss": np.mean(losses) if losses else 0.0,
            "accuracy": np.mean(accs) if accs else 0.0
        }

    def fit(self, train_loader, val_loader=None, epochs=10, callbacks=None, progress_bar=False):
        """
        Train using a standard PyTorch DataLoader.
        Restores compatibility with sklearn wrapper and standard usage.
        """
        print(f"Starting training for {epochs} epochs...")

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(epochs):
            t0 = time.time()
            train_losses = []
            train_accs = []

            # Training Loop
            self.model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                metrics = self.train_batch(x, y)
                train_losses.append(metrics["loss"])
                train_accs.append(metrics.get("accuracy", 0.0))

            # Validation Loop
            val_loss = 0.0
            val_acc = 0.0
            if val_loader:
                val_metrics = self.evaluate_loader(val_loader)
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]

            # Logging
            avg_loss = np.mean(train_losses) if train_losses else 0.0
            avg_acc = np.mean(train_accs) if train_accs else 0.0
            epoch_time = time.time() - t0

            # Update history
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)
            if val_loader:
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            if progress_bar or (epoch + 1) % 1 == 0:
                val_str = f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}" if val_loader else ""
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}"
                      f"{val_str}, "
                      f"Time={epoch_time:.1f}s")

            if callbacks:
                for cb in callbacks:
                    cb(epoch, {"loss": avg_loss, "accuracy": avg_acc, "val_loss": val_loss, "val_accuracy": val_acc})

        return history
