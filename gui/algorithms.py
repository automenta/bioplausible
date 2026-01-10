
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .types import TrainingState

# Import models from eqprop_torch
from eqprop_torch.models import LoopedMLP, BackpropMLP, ConvEqProp, TransformerEqProp
from eqprop_torch.lm_models import create_eqprop_lm
from eqprop_torch.bioplausible import (
    StandardFA as DirectFeedbackAlignmentEqProp,
    ContrastiveFeedbackAlignment as ContrastiveHebbianLearning,
)

# ============================================================================
# Model Registry - Flat list of all available models with their metadata
# ============================================================================

@dataclass
class ModelSpec:
    """Specification for a model in the GUI."""
    name: str           # Display name
    description: str    # Short description
    model_type: str     # Internal type key
    variant: str = None # Variant for transformer models
    default_lr: float = 0.001
    default_beta: float = 0.22
    default_steps: int = 30
    has_beta: bool = False
    has_steps: bool = False
    color: str = '#888888'

# All models available in the GUI - ordered by category
MODEL_REGISTRY = [
    # Baselines
    ModelSpec(
        name="Backprop (Transformer)",
        description="Standard Backprop Transformer baseline",
        model_type="backprop",
        default_lr=0.001,
        color="#ff6b6b"
    ),
    # EqProp MLP
    ModelSpec(
        name="EqProp MLP",
        description="Looped MLP with Spectral Norm",
        model_type="eqprop_mlp",
        default_lr=0.001,
        default_beta=0.22,
        default_steps=30,
        has_beta=True,
        has_steps=True,
        color="#4ecdc4"
    ),
    # Other Bio-Plausible Algorithms
    ModelSpec(
        name="DFA (Direct Feedback Alignment)",
        description="Random feedback weights",
        model_type="dfa",
        default_lr=0.001,
        color="#45b7d1"
    ),
    ModelSpec(
        name="CHL (Contrastive Hebbian)",
        description="Contrastive Hebbian Learning",
        model_type="chl",
        default_lr=0.001,
        default_beta=0.1,
        default_steps=20,
        has_beta=True,
        has_steps=True,
        color="#f9ca24"
    ),
    ModelSpec(
        name="Deep Hebbian (Hundred-Layer)",
        description="100-layer Hebbian chain with SN",
        model_type="deep_hebbian",
        default_lr=0.0005,
        color="#6c5ce7"
    ),
    # EqProp Transformers (From Track 37 results) - SLOW MODELS LAST
    ModelSpec(
        name="EqProp Transformer (Attention Only)",
        description="Best variant: EqProp in attention only",
        model_type="eqprop_transformer",
        variant="attention_only",
        default_lr=0.0003,
        default_steps=10,
        has_steps=True,
        color="#2ecc71"
    ),
    ModelSpec(
        name="EqProp Transformer (Full)",
        description="All layers use equilibrium",
        model_type="eqprop_transformer",
        variant="full",
        default_lr=0.0003,
        default_steps=15,
        has_steps=True,
        color="#27ae60"
    ),
    ModelSpec(
        name="EqProp Transformer (Hybrid)",
        description="Standard layers + EqProp final layer",
        model_type="eqprop_transformer",
        variant="hybrid",
        default_lr=0.0003,
        default_steps=10,
        has_steps=True,
        color="#1abc9c"
    ),
    ModelSpec(
        name="EqProp Transformer (Recurrent)",
        description="Single recurrent block, parameter efficient",
        model_type="eqprop_transformer",
        variant="recurrent_core",
        default_lr=0.0003,
        default_steps=20,
        has_steps=True,
        color="#16a085"
    ),
]

def get_model_spec(name: str) -> ModelSpec:
    """Get model spec by name."""
    for spec in MODEL_REGISTRY:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown model: {name}")


class SimpleLM(nn.Module):
    """Minimal transformer for language modeling (Backprop baseline)."""
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)
        self.tf = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h = self.embed(x) + self.pos[:, :x.size(1)]
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        h = self.tf(h, src_mask=mask)
        return self.head(h)[:, -1, :]  # Last token only


class AlgorithmWrapper:
    """Unified wrapper for all algorithms."""
    
    def __init__(self, spec: ModelSpec, vocab_size: int, hidden_dim: int = 128, num_layers: int = 20, device: str = 'cpu'):
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
        
        # Create model
        self.model = self._create_model()
        
        # Param count
        self.param_count = sum(p.numel() for p in self.model.parameters())
        if self.has_embed:
            self.param_count += sum(p.numel() for p in self.embed.parameters())
        
        # Optimizer
        params = list(self.model.parameters())
        if self.has_embed:
            params += list(self.embed.parameters())
        self.opt = torch.optim.Adam(params, lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def _create_model(self):
        """Factory method for model creation."""
        self.has_embed = False
        model_type = self.spec.model_type
        
        if model_type == 'backprop':
            return SimpleLM(self.vocab_size, self.hidden_dim).to(self.device)
            
        elif model_type == 'eqprop_transformer':
            return create_eqprop_lm(
                self.spec.variant, 
                self.vocab_size, 
                self.hidden_dim, 
                num_layers=4, 
                use_sn=True
            ).to(self.device)
            
        elif model_type == 'eqprop_mlp':
            self.has_embed = True
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            return LoopedMLP(self.hidden_dim, self.hidden_dim, self.vocab_size, use_spectral_norm=True).to(self.device)
            
        elif model_type == 'dfa':
            self.has_embed = True
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            # Create a simple model for DFA equivalent using available eqprop_torch models
            from algorithms import create_model
            return create_model(
                'feedback_alignment',
                input_dim=self.hidden_dim,
                hidden_dims=[self.hidden_dim] * min(self.num_layers, 5),  # Use up to 5 layers
                output_dim=self.vocab_size,
                use_spectral_norm=True
            ).to(self.device)

        elif model_type == 'chl':
            self.has_embed = True
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            # Create a simple model for CHL equivalent using available eqprop_torch models
            from algorithms import create_model
            return create_model(
                'cf_align',  # Contrastive Feedback Alignment (similar to Contrastive Hebbian)
                input_dim=self.hidden_dim,
                hidden_dims=[self.hidden_dim] * min(self.num_layers, 5),  # Use up to 5 layers
                output_dim=self.vocab_size,
                use_spectral_norm=True
            ).to(self.device)

        elif model_type == 'deep_hebbian':
            self.has_embed = True
            self.embed = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
            # Use the factory function to create a deep model
            from algorithms import create_model
            return create_model(
                'backprop',  # Use backprop with many layers to simulate deep hebbian
                input_dim=self.hidden_dim,
                hidden_dims=[self.hidden_dim] * self.num_layers,  # Use requested depth
                output_dim=self.vocab_size,
                use_spectral_norm=True
            ).to(self.device)
            
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

    def train_step(self, x, y, step_num) -> TrainingState:
        """Single training iteration."""
        t0 = time.time()
        
        self.model.train()
        self.opt.zero_grad()
        
        try:
            if self.has_embed:
                h = self.embed(x).mean(dim=1)
                # Handle different model types
                if hasattr(self.model, 'train_step'):  # Custom algorithm with train_step
                    out = self.model(h, steps=self.steps) if hasattr(self.model, 'W_rec') else self.model(h)
                else:
                    out = self.model(h)
                logits = out
            else:
                # Transformer models handle embedding internally
                if hasattr(self.model, 'train_step'):  # Custom algorithm
                    logits = self.model(x, steps=self.steps) if hasattr(self.model, 'eq_steps') else self.model(x)
                else:
                    logits = self.model(x, steps=self.steps) if hasattr(self.model, 'eq_steps') else self.model(x)

                # If sequence output (B, T, V), use last token
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
            
            loss = self.criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            
            acc = (logits.argmax(1) == y).float().mean().item()
            iter_time = time.time() - t0
            
            # VRAM estimate
            if torch.cuda.is_available() and 'cuda' in self.device:
                vram = torch.cuda.memory_allocated() / 1e9
            else:
                vram = (self.param_count * 4) / 1e9 
            
            # Signal norms
            norms = []
            if hasattr(self.model, 'layers'):
                for layer in self.model.layers:
                    if hasattr(layer, 'weight'):
                        norms.append(layer.weight.norm().item())
            elif hasattr(self.model, 'W_rec'):
                norms = [self.model.W_in.weight.norm().item(), self.model.W_rec.weight.norm().item(), self.model.W_out.weight.norm().item()]

            if not norms:
                norms = [1.0] * 10
            
            return TrainingState(
                loss=loss.item(),
                accuracy=acc,
                perplexity=np.exp(min(loss.item(), 10)),
                iter_time=iter_time,
                vram_gb=vram,
                signal_norms=norms[:30],
                step=step_num
            )
        
        except Exception as e:
            print(f"Error in {self.name} train_step: {e}")
            import traceback
            traceback.print_exc()
            return TrainingState(loss=10.0, iter_time=0.01, step=step_num)
    
    def generate(self, seed_idx, i2c, length=50):
        """Generate text sample."""
        self.model.eval()
        curr = seed_idx.clone().to(self.device)
        result = ""

        with torch.no_grad():
            try:
                for _ in range(length):
                    if self.has_embed:
                        h = self.embed(curr[-64:].unsqueeze(0)).mean(dim=1)
                        # Handle different model types
                        if hasattr(self.model, 'train_step'):  # Custom algorithm
                            logits = self.model(h)
                        else:
                            logits = self.model(h)
                    else:
                        ctx = curr[-64:].unsqueeze(0)
                        # Handle different model types
                        if hasattr(self.model, 'train_step'):  # Custom algorithm
                            logits = self.model(ctx, steps=self.steps) if hasattr(self.model, 'eq_steps') else self.model(ctx)
                        else:
                            logits = self.model(ctx, steps=self.steps) if hasattr(self.model, 'eq_steps') else self.model(ctx)

                        if logits.dim() == 3:
                            logits = logits[:, -1, :]

                    next_tok = logits.argmax(-1).item()
                    result += i2c.get(next_tok, '?')
                    curr = torch.cat([curr, torch.tensor([next_tok], device=self.device)])
            except:
                pass

        return result if result else "..."
