#!/usr/bin/env python
"""Quick shallow search test"""

import sys
sys.path.insert(0, '/home/me/eqprop')

from experiments.shallow_search import ShallowSearcher, load_mnist_subset
from algorithms import ALGORITHM_REGISTRY

if __name__ == '__main__':
    # Test with first 5 algorithms only  
    algorithms = ['backprop', 'eqprop', 'feedback_alignment', 'eq_align', 'ada_fa']
    
    print("Loading MNIST subset...")
    train_loader, test_loader = load_mnist_subset(n_samples=1000)  # Even smaller for quick test
    
    searcher = ShallowSearcher(
        algorithms=algorithms,
        param_budget=50_000,  # Smaller for speed
    )
    
    results = searcher.ultra_shallow_eval(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=784,
        output_dim=10,
        time_budget=20.0,  # 20 seconds per algorithm
    )
    
    searcher.print_summary()
