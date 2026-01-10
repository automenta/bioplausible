#!/usr/bin/env python3
"""
Rapid Language Modeling Experiments

Fast experimentation script to understand EqProp's 2× perplexity advantage.
Focuses on speed and iteration over comprehensive evaluation.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import csv
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import BackpropTransformerLM, get_eqprop_lm, list_eqprop_lm_variants


def load_shakespeare_tiny(max_chars=1000):
    """Load tiny Shakespeare for rapid experiments."""
    data_path = Path('data/shakespeare.txt')
    data_path.parent.mkdir(exist_ok=True)
    
    if not data_path.exists():
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading Shakespeare...")
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r') as f:
        text = f.read()[:max_chars]
    
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    n = int(0.9 * len(data))
    
    return data[:n], data[n:], vocab_size, char_to_idx


def get_batch(data, seq_len, batch_size, device):
    """Sample a batch."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


def train_and_eval(model, train_data, val_data, vocab_size, seq_len, 
                   epochs, lr, batch_size, device, name="model"):
    """Train model and return final metrics."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # Train on multiple batches per epoch
        for _ in range(10):  # 10 batches per epoch for speed
            x, y = get_batch(train_data, seq_len, batch_size, device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Final evaluation
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    n_val_batches = 20
    
    with torch.no_grad():
        for _ in range(n_val_batches):
            x, y = get_batch(val_data, seq_len, batch_size, device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()
    
    train_time = time.time() - start_time
    avg_loss = total_loss / n_val_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    return {
        'name': name,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'loss': avg_loss,
        'params': n_params,
        'time': train_time,
    }


def exp_ablation(device, output_dir):
    """Experiment 1: Ablation - which components matter?"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: ABLATION STUDY")
    print("="*70)
    
    # Config
    max_chars = 2000  # Tiny for speed
    seq_len = 32
    batch_size = 16
    epochs = 20
    lr = 3e-4
    
    train_data, val_data, vocab_size, _ = load_shakespeare_tiny(max_chars)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    print(f"Data: {len(train_data)} train, {len(val_data)} val, vocab={vocab_size}")
    print(f"Config: seq_len={seq_len}, epochs={epochs}, lr={lr}")
    
    results = []
    
    # Test configurations
    configs = [
        # (variant, hidden_dim, num_layers, eq_steps)
        ('looped_mlp', 64, 2, 10),
        ('recurrent_core', 64, 2, 10),
        ('full', 64, 2, 10),
        ('recurrent_core', 32, 2, 5),   # Smaller/faster
        ('recurrent_core', 128, 2, 10),  # Larger
        ('recurrent_core', 64, 2, 20),   # More equilibrium
    ]
    
    # Backprop baseline
    print("\n[Baseline] Backprop Transformer...")
    bp_model = BackpropTransformerLM(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
    ).to(device)
    
    bp_result = train_and_eval(bp_model, train_data, val_data, vocab_size, 
                               seq_len, epochs, lr, batch_size, device, 
                               name="backprop_64")
    results.append(bp_result)
    print(f"  → PPL: {bp_result['perplexity']:.2f}, Acc: {bp_result['accuracy']:.3f}, "
          f"Params: {bp_result['params']:,}, Time: {bp_result['time']:.1f}s")
    
    # EqProp variants
    for variant, hidden, layers, eq_steps in configs:
        name = f"eqprop_{variant}_h{hidden}_eq{eq_steps}"
        print(f"\n[{name}]...")
        
        try:
            eq_model = get_eqprop_lm(
                name=variant,
                vocab_size=vocab_size,
                hidden_dim=hidden,
                num_layers=layers,
                num_heads=4,
                max_eq_steps=eq_steps,
            ).to(device)
            
            result = train_and_eval(eq_model, train_data, val_data, vocab_size,
                                   seq_len, epochs, lr, batch_size, device,
                                   name=name)
            results.append(result)
            print(f"  → PPL: {result['perplexity']:.2f}, Acc: {result['accuracy']:.3f}, "
                  f"Params: {result['params']:,}, Time: {result['time']:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Save results
    output_path = output_dir / 'ablation_results.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'perplexity', 'accuracy', 'loss', 'params', 'time'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    print(f"{'Name':40s} | {'PPL':>8s} | {'Acc':>6s} | {'Params':>10s} | {'Time':>6s}")
    print("-" * 70)
    
    bp_ppl = bp_result['perplexity']
    for r in results:
        ratio = r['perplexity'] / bp_ppl
        marker = "✓" if ratio < 1.0 else " "
        print(f"{r['name']:40s} | {r['perplexity']:8.2f} | {r['accuracy']:6.3f} | "
              f"{r['params']:10,} | {r['time']:6.1f}s {marker}")
    
    # Find best
    best = min(results, key=lambda x: x['perplexity'])
    print(f"\n🏆 Best: {best['name']} with {best['perplexity']:.2f} perplexity")
    
    return results


def exp_efficiency(device, output_dir):
    """Experiment 2: Parameter efficiency frontier."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: PARAMETER EFFICIENCY")
    print("="*70)
    
    # Config - slightly larger for meaningful comparison
    max_chars = 5000
    seq_len = 64
    batch_size = 32
    epochs = 25
    lr = 3e-4
    
    train_data, val_data, vocab_size, _ = load_shakespeare_tiny(max_chars)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    print(f"Data: {len(train_data)} train, {len(val_data)} val, vocab={vocab_size}")
    
    results = []
    
    # Test different parameter scales
    base_hidden = 128
    param_scales = [0.25, 0.5, 0.75, 1.0]
    
    for scale in param_scales:
        hidden = int(base_hidden * (scale ** 0.5))  # Scale hidden dim by sqrt
        name = f"eqprop_recurrent_h{hidden}_scale{int(scale*100)}"
        print(f"\n[{name}]...")
        
        try:
            model = get_eqprop_lm(
                name='recurrent_core',
                vocab_size=vocab_size,
                hidden_dim=hidden,
                num_layers=2,
                num_heads=4,
                max_eq_steps=10,
            ).to(device)
            
            result = train_and_eval(model, train_data, val_data, vocab_size,
                                   seq_len, epochs, lr, batch_size, device,
                                   name=name)
            result['scale'] = scale
            results.append(result)
            print(f"  → PPL: {result['perplexity']:.2f}, Params: {result['params']:,}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Save results
    output_path = output_dir / 'efficiency_results.csv'
    with open(output_path, 'w', newline='') as f:
        fields = ['name', 'scale', 'perplexity', 'accuracy', 'params', 'time']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Rapid LM experiments")
    parser.add_argument('--exp', choices=['ablation', 'efficiency', 'all'], 
                       default='ablation', help='Which experiment to run')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=Path('results/lm_rapid'))
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    
    if args.exp in ['ablation', 'all']:
        exp_ablation(args.device, args.output)
    
    if args.exp in ['efficiency', 'all']:
        exp_efficiency(args.device, args.output)
    
    print("\n" + "="*70)
    print("✓ EXPERIMENTS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
