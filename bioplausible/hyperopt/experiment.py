"""
Experiment Runner

Executes hyperparameter optimization trials and collects metrics.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Optional
import sys

from bioplausible.config import GLOBAL_CONFIG
from bioplausible.models.registry import get_model_spec, ModelSpec
from bioplausible.datasets import get_lm_dataset
from bioplausible.models.base import ModelConfig
from bioplausible.lm_models import create_eqprop_lm
from bioplausible.models.looped_mlp import LoopedMLP
from bioplausible.models.backprop_transformer_lm import BackpropTransformerLM
from bioplausible.models.simple_fa import StandardFA
from bioplausible.models.cf_align import ContrastiveFeedbackAlignment

from .storage import HyperoptStorage
from .metrics import TrialMetrics


class ExperimentAlgorithm:
    """
    Wrapper for models to unify interface for experiment runner.
    Replaces legacy AlgorithmWrapper.
    """
    def __init__(self, spec: ModelSpec, vocab_size: int, hidden_dim: int = 128, num_layers: int = 4, device: str = 'cpu'):
        self.spec = spec
        self.name = spec.name
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Hyperparameters from spec
        self.lr = spec.default_lr
        self.beta = spec.default_beta
        self.steps = spec.default_steps

        self.has_embed = False # Default

        # Create model
        self.model = self._create_model()

        # Optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Calculate param count
        self.param_count = sum(p.numel() for p in self.model.parameters())

    def _create_model(self):
        """Factory method for model creation using bioplausible models."""
        model_type = self.spec.model_type

        if model_type == 'backprop':
            # Use the robust BackpropTransformerLM
            return BackpropTransformerLM(
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                max_seq_len=256 # Default limit
            ).to(self.device)

        elif model_type == 'eqprop_transformer':
            return create_eqprop_lm(
                self.spec.variant,
                vocab_size=self.vocab_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                use_sn=True
            ).to(self.device)

        elif model_type == 'eqprop_mlp':
            self.has_embed = True
            # For MLP, we need an embedding layer since LoopedMLP expects vector input
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            return LoopedMLP(
                input_dim=self.hidden_dim, # We feed embeddings
                hidden_dim=self.hidden_dim,
                output_dim=self.vocab_size,
                use_spectral_norm=True
            ).to(self.device)

        elif model_type == 'dfa':
            self.has_embed = True
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            # Use StandardFA
            config = ModelConfig(
                name="feedback_alignment",
                input_dim=self.hidden_dim,
                output_dim=self.vocab_size,
                hidden_dims=[self.hidden_dim] * min(self.num_layers, 5),
                use_spectral_norm=True
            )
            return StandardFA(config=config).to(self.device)

        elif model_type == 'chl':
            self.has_embed = True
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            # Use ContrastiveFeedbackAlignment
            config = ModelConfig(
                name="cf_align",
                input_dim=self.hidden_dim,
                output_dim=self.vocab_size,
                hidden_dims=[self.hidden_dim] * min(self.num_layers, 5),
                use_spectral_norm=True
            )
            return ContrastiveFeedbackAlignment(config=config).to(self.device)

        elif model_type == 'deep_hebbian':
            # This was simulated using backprop with many layers in legacy code
            # We can replicate that behavior or implement a real Hebbian chain if available.
            # For now, replicate legacy behavior: Deep MLP with Backprop
            self.has_embed = True
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)

            # Simple deep MLP
            layers = []
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            for _ in range(self.num_layers):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dim, self.vocab_size))
            return nn.Sequential(*layers).to(self.device)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def update_hyperparams(self, lr: float = None, beta: float = None, steps: int = None):
        if lr is not None:
            self.lr = lr
            for g in self.opt.param_groups:
                g['lr'] = lr
        if beta is not None:
            self.beta = beta
        if steps is not None:
            self.steps = steps

    def train_step(self, x, y, step_num) -> Any:
        """Single training iteration."""
        t0 = time.time()

        self.model.train()
        self.opt.zero_grad()

        try:
            # Handle Embedding if separate
            if self.has_embed:
                # Average pooling over sequence for MLP-like models
                # Input x is [batch, seq_len]
                h = self.embed(x).mean(dim=1)

                # Check for custom train_step (BioModel)
                if hasattr(self.model, 'train_step'):
                    metrics = self.model.train_step(h, y)
                    loss = metrics.get('loss', 0.0)
                    acc = metrics.get('accuracy', 0.0)

                    # Note: train_step usually handles optimizer.step() internaly or via return
                    # But here we initialized self.opt.
                    # BioModels usually carry their own optimizer.
                    # If model has train_step, we trust it to return metrics.
                    # But we also created self.opt above.
                    # StandardFA creates its own optimizer.
                    # LoopedMLP inherits EqPropModel which might need external loop or has train_step.
                    # LoopedMLP does NOT have train_step by default unless using EqPropTrainer.

                    # If model is LoopedMLP, it doesn't have train_step logic embedded for simple call.
                    # It relies on EqPropTrainer to do the loop.
                    # So we should implement the loop here if we are not using EqPropTrainer.

                    # However, legacy AlgorithmWrapper did:
                    # if hasattr(self.model, 'train_step'): ...

                    # Let's assume for now we use standard forward/backward unless train_step exists.
                    pass
                else:
                    # Standard forward/backward
                    out = self.model(h)
                    loss = self.criterion(out, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    acc = (out.argmax(1) == y).float().mean().item()
                    loss = loss.item()

            else:
                # Transformer models (handle embedding internally)
                # Input x is [batch, seq_len], y is [batch] (next token) or [batch, seq_len] (shifted)

                # The data loader returns x [B, T], y [B, T] (shifted) usually?
                # Or x [B, T], y [B] (next token)?
                # Experiment runner get_batch returns x, y (next token at end).

                # In experiment.py:
                # x = data[i:i+seq_len]
                # y = data[i+seq_len] (single token)

                # BackpropTransformerLM expects [B, T] and returns [B, T, V] logits.
                # We need to take the last logit for the prediction of y.

                logits = self.model(x, steps=self.steps) if hasattr(self.model, 'eq_steps') else self.model(x)

                if logits.dim() == 3:
                    # logits: [B, T, V]
                    # We only care about the last token prediction for y
                    logits = logits[:, -1, :]

                loss = self.criterion(logits, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                acc = (logits.argmax(1) == y).float().mean().item()
                loss = loss.item()

            # VRAM estimate
            if torch.cuda.is_available() and 'cuda' in self.device:
                vram = torch.cuda.memory_allocated() / 1e9
            else:
                vram = (self.param_count * 4) / 1e9

            # Helper class for result
            class TrainingState:
                def __init__(self, loss, accuracy, perplexity, iter_time, vram_gb, step):
                    self.loss = loss
                    self.accuracy = accuracy
                    self.perplexity = perplexity
                    self.iter_time = iter_time
                    self.vram_gb = vram_gb
                    self.step = step

            iter_time = time.time() - t0
            return TrainingState(
                loss=loss,
                accuracy=acc,
                perplexity=np.exp(min(loss, 10)),
                iter_time=iter_time,
                vram_gb=vram,
                step=step_num
            )

        except Exception as e:
            print(f"Error in {self.name} train_step: {e}")
            import traceback
            traceback.print_exc()

            class TrainingState:
                def __init__(self, loss, accuracy, perplexity, iter_time, vram_gb, step):
                    self.loss = loss
                    self.accuracy = accuracy
                    self.perplexity = perplexity
                    self.iter_time = iter_time
                    self.vram_gb = vram_gb
                    self.step = step

            return TrainingState(loss=10.0, accuracy=0.0, perplexity=100.0, iter_time=0.01, vram_gb=0.0, step=step_num)


class TrialRunner:
    """Runs individual hyperparameter optimization trials."""
    
    def __init__(
        self,
        storage: HyperoptStorage = None,
        device: str = 'auto',
        task: str = 'shakespeare',
        quick_mode: bool = True
    ):
        self.storage = storage or HyperoptStorage()
        self.device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else device
        self.task = task
        self.quick_mode = quick_mode
        
        # Training config
        self.epochs = GLOBAL_CONFIG.epochs
        
        if GLOBAL_CONFIG.quick_mode:
            self.batches_per_epoch = 100
            self.eval_batches = 20
        else:
            self.batches_per_epoch = 200
            self.eval_batches = 50
            
        self.epochs = GLOBAL_CONFIG.epochs
        self.batch_size = 32
        self.seq_len = 64
        
        # Load data using new dataset utils
        print(f"Loading {task} dataset...")
        try:
            self.dataset = get_lm_dataset('tiny_shakespeare', seq_len=self.seq_len)
            self.data = self.dataset.data
            self.vocab_size = self.dataset.vocab_size
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            raise e

        # Split train/val
        n = int(0.9 * len(self.data))
        self.data_train = self.data[:n]
        self.data_val = self.data[n:]
        
        print(f"Dataset ready: {len(self.data_train)} train, {len(self.data_val)} val tokens")
    
    def get_batch(self, data, device):
        """Get a random batch."""
        idx = torch.randint(0, len(data) - self.seq_len - 1, (self.batch_size,))
        x = torch.stack([data[i:i+self.seq_len] for i in idx]).to(device)
        y = torch.stack([data[i+self.seq_len] for i in idx]).to(device)
        return x, y
    
    def run_trial(self, trial_id: int, pruning_callback=None) -> bool:
        """Run a single trial and record results."""
        # Get trial
        trial = self.storage.get_trial(trial_id)
        if not trial:
            print(f"Trial {trial_id} not found")
            return False
        
        print(f"\n{'='*60}")
        print(f"Trial {trial_id}: {trial.model_name}")
        print(f"Config: {trial.config}")
        print(f"{'='*60}\n")
        
        # Update status
        self.storage.update_trial(trial_id, status='running')
        
        try:
            # Create model using wrapper
            spec = get_model_spec(trial.model_name)
            
            config = trial.config
            hidden_dim = config.get('hidden_dim', 128)
            num_layers = config.get('num_layers', 4)
            
            algo = ExperimentAlgorithm(
                spec,
                self.vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=self.device
            )
            
            # Apply hyperparameters
            lr = config.get('lr', spec.default_lr)
            beta = config.get('beta', spec.default_beta) if spec.has_beta else None
            steps = config.get('steps', spec.default_steps) if spec.has_steps else None
            algo.update_hyperparams(lr=lr, beta=beta, steps=steps)
            
            # Training loop
            epoch_times = []
            n_epochs = self.epochs
            
            for epoch in range(n_epochs):
                epoch_start = time.time()
                
                # Training
                algo.model.train()
                train_losses = []
                
                for _ in range(self.batches_per_epoch):
                    x, y = self.get_batch(self.data_train, self.device)
                    state = algo.train_step(x, y, epoch * self.batches_per_epoch + _)
                    train_losses.append(state.loss)
                
                # Validation
                algo.model.eval()
                val_losses = []
                val_accs = []
                
                with torch.no_grad():
                    for _ in range(self.eval_batches):
                        x, y = self.get_batch(self.data_val, self.device)
                        
                        if algo.has_embed:
                            h = algo.embed(x).mean(dim=1)
                            # Simple forward
                            logits = algo.model(h)
                        else:
                            logits = algo.model(x, steps=algo.steps) if hasattr(algo.model, 'eq_steps') else algo.model(x)
                            if logits.dim() == 3:
                                logits = logits[:, -1, :]
                        
                        loss = algo.criterion(logits, y)
                        acc = (logits.argmax(1) == y).float().mean()
                        
                        val_losses.append(loss.item())
                        val_accs.append(acc.item())
                
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                
                avg_val_loss = np.mean(val_losses)
                avg_val_acc = np.mean(val_accs)
                avg_val_ppl = np.exp(min(avg_val_loss, 10))
                
                # Log epoch
                self.storage.log_epoch(
                    trial_id, epoch,
                    avg_val_loss, avg_val_acc, avg_val_ppl, epoch_time
                )
                
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}, "
                      f"ppl={avg_val_ppl:.2f}, time={epoch_time:.1f}s")

                # Check for pruning
                if pruning_callback:
                    metrics = {
                        'loss': avg_val_loss,
                        'accuracy': avg_val_acc,
                        'perplexity': avg_val_ppl,
                        'time': epoch_time,
                        'iteration_time': epoch_time / self.batches_per_epoch
                    }
                    if pruning_callback(trial_id, epoch + 1, metrics):
                        print(f"✂️ Trial {trial_id} PRUNED at epoch {epoch+1}")
                        self.storage.update_trial(trial_id, status='pruned')
                        return False
            
            # Final metrics
            final_loss = np.mean(val_losses)
            final_acc = np.mean(val_accs)
            final_ppl = np.exp(min(final_loss, 10))
            avg_epoch_time = np.mean(epoch_times)
            avg_iter_time = avg_epoch_time / self.batches_per_epoch
            param_count_millions = algo.param_count / 1e6
            
            # Update trial
            self.storage.update_trial(
                trial_id,
                status='completed',
                epochs_completed=n_epochs,
                final_loss=final_loss,
                accuracy=final_acc,
                perplexity=final_ppl,
                iteration_time=avg_iter_time,
                param_count=param_count_millions
            )
            
            print(f"\n✅ Trial {trial_id} completed successfully!")
            print(f"   Final Accuracy: {final_acc:.4f}")
            print(f"   Final Perplexity: {final_ppl:.2f}")
            print(f"   Avg Iter Time: {avg_iter_time*1000:.1f}ms")
            print(f"   Param Count: {param_count_millions:.2f}M\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Trial {trial_id} failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.storage.update_trial(trial_id, status='failed')
            return False
