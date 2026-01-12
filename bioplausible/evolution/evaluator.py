"""
Tiered Variation Evaluator for EqProp+SN Evolution

Implements a multi-tier evaluation pipeline for rapid architecture screening:
- Tier 1 (30 sec): Synthetic data sanity check
- Tier 2 (5 min): MNIST subset (5000 samples, 10 epochs)
- Tier 3 (30 min): Full MNIST/Shakespeare evaluation
- Tier 4 (2 hrs): CIFAR-10 breakthrough test
"""

import time
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

from .fitness import FitnessScore
from .breeder import ArchConfig


class EvalTier(IntEnum):
    """Evaluation tier levels."""
    TIER_1_SMOKE = 1      # 30 sec - synthetic data sanity
    TIER_2_QUICK = 2      # 5 min - MNIST subset
    TIER_3_FULL = 3       # 30 min - full MNIST/Shakespeare
    TIER_4_BREAKTHROUGH = 4  # 2 hrs - CIFAR-10 breakthrough


@dataclass
class TierConfig:
    """Configuration for each evaluation tier."""
    n_samples: int
    epochs: int
    n_seeds: int
    timeout_sec: int
    
    
TIER_CONFIGS = {
    EvalTier.TIER_1_SMOKE: TierConfig(n_samples=500, epochs=3, n_seeds=1, timeout_sec=30),
    EvalTier.TIER_2_QUICK: TierConfig(n_samples=5000, epochs=10, n_seeds=1, timeout_sec=300),
    EvalTier.TIER_3_FULL: TierConfig(n_samples=60000, epochs=30, n_seeds=3, timeout_sec=1800),
    EvalTier.TIER_4_BREAKTHROUGH: TierConfig(n_samples=50000, epochs=100, n_seeds=5, timeout_sec=7200),
}


class VariationEvaluator:
    """Tiered evaluation for rapid architecture screening."""
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        data_dir: str = './data',
        verbose: bool = True,
    ):
        self.device = device
        self.data_dir = data_dir
        self.verbose = verbose
        self._data_cache: Dict[str, Any] = {}
    
    def evaluate(
        self,
        config: ArchConfig,
        tier: EvalTier = EvalTier.TIER_2_QUICK,
        task: str = 'mnist',
    ) -> FitnessScore:
        """
        Evaluate a model configuration at the specified tier.
        
        Args:
            config: Architecture configuration
            tier: Evaluation tier (1-4)
            task: Task name ('mnist', 'fashion', 'shakespeare', 'cifar10', 'cartpole', etc.)
            
        Returns:
            FitnessScore with all metrics
        """
        # Check if RL task
        if task in ['cartpole', 'acrobot', 'mountaincar']:
            return self._evaluate_rl(config, tier, task)
        
        tier_cfg = TIER_CONFIGS[tier]
        
        if self.verbose:
            print(f"[Tier {tier}] Evaluating {config.model_type} "
                  f"(d={config.depth}, w={config.width}) on {task}")
        
        # Run multiple seeds for higher tiers
        seed_results = []
        for seed in range(tier_cfg.n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)
            
            try:
                result = self._single_evaluation(config, tier_cfg, task, seed)
                seed_results.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"  [Seed {seed}] Failed: {e}")
                continue
        
        if not seed_results:
            # All seeds failed
            return FitnessScore(
                accuracy=0.0,
                config=config.to_dict(),
                task=task,
            )
        
        # Aggregate results
        return self._aggregate_results(seed_results, config, task)
    
    def _evaluate_rl(
        self,
        config: ArchConfig,
        tier: EvalTier,
        task: str,
    ) -> FitnessScore:
        """Evaluate on RL environment."""
        try:
            from .rl_adapter import train_rl_model, get_env_name
        except ImportError:
            return FitnessScore(
                accuracy=0.0,
                config=config.to_dict(),
                task=task,
            )
        
        tier_cfg = TIER_CONFIGS[tier]
        env_name = get_env_name(task)
        
        # Build model
        model = self._build_model(config, task)
        model = model.to(self.device)
        
        # Train on RL task
        start_time = time.time()
        n_episodes = min(tier_cfg.epochs * 10, 200)  # Scale epochs to episodes
        
        try:
            success_rate = train_rl_model(
                model,
                env_name,
                n_episodes=n_episodes,
                device=self.device,
                lr=config.lr,
            )
        except Exception as e:
            if self.verbose:
                print(f"  RL training failed: {e}")
            success_rate = 0.0
        
        train_time = time.time() - start_time
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Compute Lipschitz
        L = self._compute_lipschitz(model)
        
        return FitnessScore(
            accuracy=success_rate,
            train_accuracy=success_rate,
            speed=n_episodes / max(train_time, 0.01),
            memory_mb=0,  # Not tracking for RL
            train_time_sec=train_time,
            parameter_count=param_count,  # NEW
            lipschitz=L,
            generalization=1.0,
            stability=1.0,
            config=config.to_dict(),
            task=task,
        )
    
    def _single_evaluation(
        self,
        config: ArchConfig,
        tier_cfg: TierConfig,
        task: str,
        seed: int,
    ) -> FitnessScore:
        """Run a single evaluation with given seed."""
        start_time = time.time()
        
        # Build model
        model = self._build_model(config, task)
        model = model.to(self.device)
        
        # Get data
        train_loader, test_loader = self._get_data(task, tier_cfg.n_samples)
        
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        iterations = 0
        lipschitz_values = []
        
        model.train()
        for epoch in range(tier_cfg.epochs):
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                # Handle different model outputs
                if task == 'shakespeare':
                    out = model(x)
                    loss = F.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
                else:
                    out = model(x)
                    loss = F.cross_entropy(out, y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                iterations += 1
                
                # Timeout check
                if time.time() - start_time > tier_cfg.timeout_sec:
                    if self.verbose:
                        print(f"  Timeout at epoch {epoch}")
                    break
            
            # Track Lipschitz
            L = self._compute_lipschitz(model)
            lipschitz_values.append(L)
        
        train_time = time.time() - start_time
        
        # Evaluate
        model.eval()
        train_acc, train_loss = self._evaluate_accuracy(model, train_loader, task)
        test_acc, test_loss = self._evaluate_accuracy(model, test_loader, task)
        
        # Memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory = 0
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Compute metrics
        gen_gap = 1.0 - abs(train_acc - test_acc) / max(train_acc, 0.01)
        
        return FitnessScore(
            accuracy=test_acc,
            perplexity=torch.exp(torch.tensor(test_loss)).item() if task == 'shakespeare' else float('inf'),
            train_accuracy=train_acc,
            speed=iterations / max(train_time, 0.01),
            memory_mb=peak_memory,
            train_time_sec=train_time,
            parameter_count=param_count,  # NEW
            lipschitz=lipschitz_values[-1] if lipschitz_values else float('inf'),
            lipschitz_trajectory=lipschitz_values,
            generalization=gen_gap,
            stability=1.0,  # Single seed
            config=config.to_dict(),
            task=task,
            seed=seed,
        )
    
    def _aggregate_results(
        self,
        results: list,
        config: ArchConfig,
        task: str,
    ) -> FitnessScore:
        """Aggregate results from multiple seeds."""
        accs = [r.accuracy for r in results]
        
        return FitnessScore(
            accuracy=np.mean(accs),
            perplexity=np.mean([r.perplexity for r in results if r.perplexity < float('inf')]) or float('inf'),
            train_accuracy=np.mean([r.train_accuracy for r in results]),
            speed=np.mean([r.speed for r in results]),
            memory_mb=np.max([r.memory_mb for r in results]),
            train_time_sec=np.sum([r.train_time_sec for r in results]),
            parameter_count=int(np.mean([r.parameter_count for r in results])),  # NEW
            lipschitz=np.mean([r.lipschitz for r in results if r.lipschitz < float('inf')]) or float('inf'),
            generalization=np.mean([r.generalization for r in results]),
            stability=1.0 / (np.std(accs) + 0.01),  # Higher variance = lower stability
            config=config.to_dict(),
            task=task,
        )
    
    def _build_model(self, config: ArchConfig, task: str) -> nn.Module:
        """Build model from configuration using model registry."""
        from .models import DefaultModelBuilder
        builder = DefaultModelBuilder()
        return builder.build(config, task)
    
    def _get_data(
        self,
        task: str,
        n_samples: int,
    ) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for the task."""
        cache_key = f"{task}_{n_samples}"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        if task in ['mnist', 'fashion']:
            from torchvision import datasets, transforms
            
            Dataset = datasets.MNIST if task == 'mnist' else datasets.FashionMNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            
            train_data = Dataset(self.data_dir, train=True, download=True, transform=transform)
            test_data = Dataset(self.data_dir, train=False, download=True, transform=transform)
            
            # Subset
            train_subset = Subset(train_data, list(range(min(n_samples, len(train_data)))))
            
            train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
            
        elif task == 'cifar10':
            from torchvision import datasets, transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            
            train_data = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=transform)
            test_data = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=transform)
            
            train_subset = Subset(train_data, list(range(min(n_samples, len(train_data)))))
            
            train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
            
        elif task == 'shakespeare':
            # Small Shakespeare dataset
            train_loader, test_loader = self._get_shakespeare_data(n_samples)
            
        else:
            # Synthetic data fallback
            X = torch.randn(n_samples, 784)
            y = torch.randint(0, 10, (n_samples,))
            train_data = TensorDataset(X, y)
            
            X_test = torch.randn(1000, 784)
            y_test = torch.randint(0, 10, (1000,))
            test_data = TensorDataset(X_test, y_test)
            
            train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        
        self._data_cache[cache_key] = (train_loader, test_loader)
        return train_loader, test_loader
    
    def _get_shakespeare_data(self, n_samples: int) -> Tuple[DataLoader, DataLoader]:
        """Load Shakespeare dataset for language modeling."""
        import requests
        import os
        
        # Download if needed
        data_path = os.path.join(self.data_dir, 'shakespeare.txt')
        if not os.path.exists(data_path):
            os.makedirs(self.data_dir, exist_ok=True)
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            try:
                text = requests.get(url, timeout=10).text
                with open(data_path, 'w') as f:
                    f.write(text)
            except:
                text = "To be or not to be, that is the question." * 1000
                with open(data_path, 'w') as f:
                    f.write(text)
        
        with open(data_path, 'r') as f:
            text = f.read()[:min(n_samples * 100, len(open(data_path).read()))]
        
        # Character-level tokenization
        chars = sorted(set(text))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        vocab_size = len(chars)
        
        # Create sequences
        seq_len = 64
        data = torch.tensor([char_to_idx.get(ch, 0) for ch in text], dtype=torch.long)
        
        # Split
        n = int(0.9 * len(data))
        train_data = data[:n]
        test_data = data[n:]
        
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(0, len(data) - seq_len - 1, seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+1:i+seq_len+1])
            return TensorDataset(torch.stack(X), torch.stack(y))
        
        train_dataset = create_sequences(train_data, seq_len)
        test_dataset = create_sequences(test_data, seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        return train_loader, test_loader
    
    def _evaluate_accuracy(
        self,
        model: nn.Module,
        loader: DataLoader,
        task: str,
    ) -> Tuple[float, float]:
        """Evaluate accuracy and loss."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                
                if task == 'shakespeare':
                    loss = F.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
                    pred = out.argmax(dim=-1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
                else:
                    loss = F.cross_entropy(out, y)
                    pred = out.argmax(dim=-1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
                
                total_loss += loss.item() * y.size(0)
        
        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        return acc, avg_loss
    
    def _compute_lipschitz(self, model: nn.Module) -> float:
        """Compute approximate Lipschitz constant of the model."""
        L = 1.0
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                try:
                    # Estimate spectral norm
                    W = module.weight.detach()
                    if W.dim() >= 2:
                        sigma = torch.linalg.svdvals(W.view(W.size(0), -1))[0].item()
                        L = max(L, sigma)
                except:
                    pass
        return L
