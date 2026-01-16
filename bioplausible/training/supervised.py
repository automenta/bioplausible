import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Optional

from bioplausible.training.base import BaseTrainer
from bioplausible.models.hebbian_chain import DeepHebbianChain

class SupervisedTrainer(BaseTrainer):
    """
    Trainer for Supervised Learning (LM, Vision).
    """

    def __init__(
        self,
        model: nn.Module,
        task,  # BaseTask
        device: str = "cpu",
        lr: float = 0.001,
        batches_per_epoch: int = 100,
        eval_batches: int = 20,
        steps: int = 20, # EqProp steps
        **kwargs
    ):
        super().__init__(model, device)
        self.task = task
        self.batches_per_epoch = batches_per_epoch
        self.eval_batches = eval_batches
        self.steps = steps

        # Check for embeddings
        self.has_embed = getattr(model, 'has_embed', False)
        self.embed = getattr(model, 'embed', None)

        # Optimizer
        if not hasattr(self.model, 'optimizer'):
            params = list(self.model.parameters())
            if self.has_embed and self.embed:
                params.extend(list(self.embed.parameters()))
            self.opt = torch.optim.Adam(params, lr=lr)
        else:
            self.opt = None

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
        if self.has_embed:
            # Average pooling over sequence for MLP-like models on LM task
            # (If this logic is correct for the intended LM models)
            # ExperimentAlgorithm used: h = self.embed(x).mean(dim=1)
            return self.embed(x).mean(dim=1)
        else:
            # Vision or direct input
            if x.dim() > 2 and self.task.task_type in ["vision", "rl"]:
                # Check for Conv models (ModernConvEqProp)
                # We check via class name or attribute to avoid importing the class here if possible
                if "Conv" in self.model.__class__.__name__:
                     return x
                else:
                     return x.view(x.size(0), -1)
            else:
                 return x

    def train_batch(self, x, y) -> Dict[str, float]:
        """Run a single training step."""
        self.model.train()
        if self.opt:
            self.opt.zero_grad()

        h = self._prepare_input(x)

        # Check for custom train_step (BioModel)
        if hasattr(self.model, "train_step"):
            metrics = self.model.train_step(h, y)
            loss = metrics.get("loss", 0.0)
            acc = metrics.get("accuracy", 0.0)
        else:
            # Standard forward/backward
            if hasattr(self.model, "eq_steps"):
                logits = self.model(h, steps=self.steps)
            else:
                logits = self.model(h)

            if logits.dim() == 3 and self.task.task_type == "lm":
                # logits: [B, T, V] -> [B, V] (last token)
                logits = logits[:, -1, :]

            loss = self.criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.opt:
                self.opt.step()

            # Compute accuracy (detached)
            with torch.no_grad():
                if self.task.task_type in ["lm", "vision"]:
                    acc = (logits.argmax(1) == y).float().mean().item()
                else:
                    acc = 0.0

            loss = loss.item()

        return {"loss": loss, "accuracy": acc}

    def evaluate(self) -> Dict[str, float]:
        """Run validation loop."""
        self.model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for _ in range(self.eval_batches):
                x, y = self.task.get_batch("val")

                h = self._prepare_input(x)

                if hasattr(self.model, "eq_steps"):
                    logits = self.model(h, steps=self.steps)
                else:
                    logits = self.model(h)

                if logits.dim() == 3 and self.task.task_type == "lm":
                    logits = logits[:, -1, :]

                loss = self.criterion(logits, y)
                metrics = self.task.compute_metrics(logits, y, loss.item())

                val_losses.append(metrics["loss"])
                val_accs.append(metrics.get("accuracy", 0.0))

        avg_loss = np.mean(val_losses)
        avg_acc = np.mean(val_accs)

        return {
            "val_loss": avg_loss,
            "val_accuracy": avg_acc,
            "val_perplexity": np.exp(min(avg_loss, 10)) if self.task.task_type == "lm" else 0.0
        }

    def train_epoch(self) -> Dict[str, float]:
        """Run full training epoch (train + eval)."""
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
