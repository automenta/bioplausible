#!/usr/bin/env python3
"""
Debug Track 37 Discrepancy

Track 37 showed: EqProp 9.67 vs Backprop 20.28 (2× better)
Scale Study showed: EqProp 13.05 vs Backprop 11.30 (1.16× worse)

This script systematically tests differences to find the cause.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import json
from dataclasses import dataclass, asdict
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import BackpropTransformerLM, get_eqprop_lm


def load_shakespeare(max_chars=None):
    """Load Shakespeare with exact Track 37 preprocessing."""
    data_path = Path('data/shakespeare.txt')
    data_path.parent.mkdir(exist_ok=True)
    
    if not data_path.exists():
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r') as f:
        text = f.read()
        if max_chars:
            text = text[:max_chars]
    
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    n = int(0.9 * len(data))
    
    return data[:n], data[n:], vocab_size


def get_batch(data, seq_len, batch_size, device):
    """Sample batch."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


def train_and_eval(model, train_data, val_data, vocab_size, config, device, verbose=False):
    """Train with detailed logging."""
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    
    seq_len = config['seq_len']
    batch_size = config['batch_size']
    epochs = config['epochs']
    batches_per_epoch = config.get('batches_per_epoch', 50)
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for _ in range(batches_per_epoch):
            x, y = get_batch(train_data, seq_len, batch_size, device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _ in range(20):
                x, y = get_batch(val_data, seq_len, batch_size, device)
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / 20
        ppl = torch.exp(torch.tensor(avg_val_loss)).item()
        history.append({'epoch': epoch, 'train_loss': epoch_loss/batches_per_epoch, 
                       'val_loss': avg_val_loss, 'perplexity': ppl})
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: PPL={ppl:.2f}")
    
    return history[-1]['perplexity'], history


def run_comparison(config_name, config, device, seed):
    """Run single comparison."""
    torch.manual_seed(seed)
    
    train_data, val_data, vocab_size = load_shakespeare(config['dataset_size'])
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    results = {}
    
    # Backprop
    bp_model = BackpropTransformerLM(
        vocab_size=vocab_size,
        hidden_dim=config['hidden'],
        num_layers=config['layers'],
        num_heads=config.get('heads', 4),
    ).to(device)
    
    # Determined LR safely
    bp_lr = config.get('lr_bp', config.get('lr'))
    if bp_lr is None: raise ValueError("No LR specified for Backprop")
    
    bp_ppl, bp_history = train_and_eval(
        bp_model, train_data, val_data, vocab_size,
        {**config, 'lr': bp_lr}, device
    )
    results['backprop'] = {'perplexity': bp_ppl, 'history': bp_history}
    
    # EqProp variants
    for variant in config.get('variants', ['full']):
        torch.manual_seed(seed)  # Reset for fair comparison
        
        eq_model = get_eqprop_lm(
            name=variant,
            vocab_size=vocab_size,
            hidden_dim=config['hidden'],
            num_layers=config['layers'],
            num_heads=config.get('heads', 4),
            max_eq_steps=config.get('eq_steps', 15),
        ).to(device)
        
        # Determined LR safely
        eq_lr = config.get('lr_eq', config.get('lr'))
        if eq_lr is None: raise ValueError(f"No LR specified for EqProp {variant}")
        
        eq_ppl, eq_history = train_and_eval(
            eq_model, train_data, val_data, vocab_size,
            {**config, 'lr': eq_lr}, device
        )
        results[f'eqprop_{variant}'] = {'perplexity': eq_ppl, 'history': eq_history}
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('results/track37_debug'))
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TRACK 37 DISCREPANCY DEBUG")
    print("="*70)
    print(f"Device: {args.device}, Seeds: {args.seeds}")
    
    # Configurations to test
    # 10K chars / (32 batch * 64 seq) ~= 4 batches per epoch (Low Compute)
    # Track 37 used fixed 50 batches per epoch (High Compute, 12.5x more)
    configs = {
        'high_compute_track37': {
            'dataset_size': 10000,
            'seq_len': 64,
            'hidden': 128,
            'layers': 3,
            'epochs': 30,
            'batch_size': 32,
            'batches_per_epoch': 50,  # 1500 updates total
            'lr_bp': 5e-4,
            'lr_eq': 3e-4,
            'eq_steps': 15,
            'variants': ['recurrent_core'],  # Faster than full, similar performance
        },
        'low_compute_scale_study': {
            'dataset_size': 10000,
            'seq_len': 64,
            'hidden': 128,
            'layers': 3,
            'epochs': 30,
            'batch_size': 32,
            'batches_per_epoch': 4,  # Approx 10000 // (32*64) = 4. 120 updates total
            'lr_bp': 3e-4, # Matches scale study
            'lr_eq': 3e-4,
            'eq_steps': 15,
            'variants': ['recurrent_core'],
        },
    }
    
    all_results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*70}")
        print(f"Config: {config_name}")
        print(f"  LR_BP: {config.get('lr_bp', config.get('lr'))}, "
              f"LR_EQ: {config.get('lr_eq', config.get('lr'))}, "
              f"EQ_steps: {config.get('eq_steps')}")
        print("="*70)
        
        seed_results = []
        
        for seed in range(args.seeds):
            print(f"\n  Seed {seed+1}/{args.seeds}...")
            results = run_comparison(config_name, config, args.device, seed)
            
            bp_ppl = results['backprop']['perplexity']
            for name, data in results.items():
                if name != 'backprop':
                    eq_ppl = data['perplexity']
                    ratio = eq_ppl / bp_ppl
                    marker = "✓" if ratio < 1.0 else "✗"
                    print(f"    {name}: {eq_ppl:.2f} vs BP {bp_ppl:.2f} = {ratio:.2f}× {marker}")
            
            seed_results.append(results)
        
        all_results[config_name] = seed_results
        
        # Aggregate
        bp_ppls = [r['backprop']['perplexity'] for r in seed_results]
        print(f"\n  Backprop avg: {sum(bp_ppls)/len(bp_ppls):.2f}")
        
        for variant in config.get('variants', ['full']):
            eq_ppls = [r[f'eqprop_{variant}']['perplexity'] for r in seed_results]
            avg_eq = sum(eq_ppls) / len(eq_ppls)
            avg_bp = sum(bp_ppls) / len(bp_ppls)
            print(f"  EqProp {variant} avg: {avg_eq:.2f} ({avg_eq/avg_bp:.2f}× BP)")
    
    # Save results
    with open(args.output / 'debug_results.json', 'w') as f:
        # Convert history to serializable format
        serializable = {}
        for config_name, seed_results in all_results.items():
            serializable[config_name] = []
            for seed_result in seed_results:
                seed_dict = {}
                for model_name, data in seed_result.items():
                    seed_dict[model_name] = {
                        'perplexity': data['perplexity'],
                        'history': data['history']
                    }
                serializable[config_name].append(seed_dict)
        json.dump(serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for config_name, seed_results in all_results.items():
        bp_ppls = [r['backprop']['perplexity'] for r in seed_results]
        avg_bp = sum(bp_ppls) / len(bp_ppls)
        
        print(f"\n{config_name}:")
        print(f"  Backprop: {avg_bp:.2f} ± {max(bp_ppls)-min(bp_ppls):.2f}")
        
        for key in seed_results[0].keys():
            if key != 'backprop':
                ppls = [r[key]['perplexity'] for r in seed_results]
                avg = sum(ppls) / len(ppls)
                print(f"  {key}: {avg:.2f} ± {max(ppls)-min(ppls):.2f} ({avg/avg_bp:.2f}× BP)")


if __name__ == '__main__':
    main()
