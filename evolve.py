#!/usr/bin/env python3
"""
EqProp+SN Evolution CLI

Command-line interface for evolving EqProp+Spectral Normalization model variants.

Examples:
    # Quick evolution on MNIST (1 hour)
    python evolve.py --task mnist --generations 10 --population 20
    
    # Full CIFAR-10 breakthrough search
    python evolve.py --task cifar10 --generations 50 --population 100 --hours 24
    
    # Resume from checkpoint
    python evolve.py --resume results/evolution/checkpoint_latest.json
    
    # Dry run to test configuration
    python evolve.py --task mnist --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from evolution import (
    EvolutionEngine, 
    VariationBreeder, 
    VariationEvaluator, 
    BreakthroughDetector,
    EvalTier,
    ArchConfig,
)
from evolution.engine import EvolutionConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evolve EqProp+SN model variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Task configuration
    parser.add_argument(
        '--task', 
        type=str, 
        default='mnist',
        choices=['mnist', 'fashion', 'cifar10', 'shakespeare'],
        help='Task to evolve for'
    )
    
    # Evolution parameters
    parser.add_argument(
        '--generations', '-g',
        type=int, 
        default=20,
        help='Number of generations'
    )
    parser.add_argument(
        '--population', '-p',
        type=int, 
        default=20,
        help='Population size'
    )
    parser.add_argument(
        '--mutation-rate', '-m',
        type=float, 
        default=0.3,
        help='Mutation rate (0-1)'
    )
    parser.add_argument(
        '--crossover-rate', '-c',
        type=float, 
        default=0.7,
        help='Crossover rate (0-1)'
    )
    parser.add_argument(
        '--elite-fraction', '-e',
        type=float, 
        default=0.2,
        help='Fraction of population to keep as elites'
    )
    
    # Evaluation tier
    parser.add_argument(
        '--tier',
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help='Evaluation tier (1=smoke, 2=quick, 3=full, 4=breakthrough)'
    )
    
    # Time limits
    parser.add_argument(
        '--hours',
        type=float,
        default=24.0,
        help='Maximum runtime in hours'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/evolution',
        help='Output directory'
    )
    
    # Checkpoint
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint file'
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=5,
        help='Save checkpoint every N generations'
    )
    
    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build configuration
    config = EvolutionConfig(
        population_size=args.population,
        n_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_fraction=args.elite_fraction,
        task=args.task,
        eval_tier=EvalTier(args.tier),
        seed=args.seed,
        timeout_hours=args.hours,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output,
    )
    
    # Dry run
    if args.dry_run:
        print("Evolution Configuration:")
        print(f"  Task: {config.task}")
        print(f"  Generations: {config.n_generations}")
        print(f"  Population: {config.population_size}")
        print(f"  Mutation Rate: {config.mutation_rate}")
        print(f"  Crossover Rate: {config.crossover_rate}")
        print(f"  Elite Fraction: {config.elite_fraction}")
        print(f"  Evaluation Tier: {config.eval_tier.name}")
        print(f"  Max Hours: {config.timeout_hours}")
        print(f"  Output: {config.output_dir}")
        print(f"  Seed: {config.seed}")
        
        # Generate sample individual
        breeder = VariationBreeder(seed=config.seed)
        sample = breeder.generate_random()
        print("\nSample Individual:")
        print(f"  Model Type: {sample.model_type}")
        print(f"  Depth: {sample.depth}")
        print(f"  Width: {sample.width}")
        print(f"  EQ Steps: {sample.eq_steps}")
        print(f"  Use SN: {sample.use_sn}")
        return
    
    # Resume or create new
    if args.resume:
        print(f"Resuming from {args.resume}")
        engine = EvolutionEngine.resume(args.resume)
        # Override some settings
        engine.config.n_generations = args.generations
        engine.config.timeout_hours = args.hours
    else:
        engine = EvolutionEngine(config=config)
    
    # Run evolution
    print("\n" + "="*60)
    print("EqProp+SN Evolution System")
    print("="*60)
    
    try:
        state = engine.run(verbose=not args.quiet)
        
        # Print summary
        print("\n" + "="*60)
        print("Evolution Complete!")
        print("="*60)
        print(f"Generations: {state.generation}")
        
        if state.best_individual:
            best = state.best_individual
            print(f"\nBest Individual:")
            print(f"  Accuracy: {best.fitness.accuracy:.4f}")
            print(f"  Lipschitz: {best.fitness.lipschitz:.3f}")
            print(f"  Model: {best.config.model_type}")
            print(f"  Depth: {best.config.depth}, Width: {best.config.width}")
            print(f"  EQ Steps: {best.config.eq_steps}")
        
        # Breakthroughs
        n_breakthroughs = len(engine.detector.breakthroughs)
        if n_breakthroughs > 0:
            print(f"\n★ {n_breakthroughs} Breakthroughs Detected!")
            summary_path = engine.detector.save_summary()
            print(f"  See: {summary_path}")
        
        print(f"\nResults saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving checkpoint...")
        engine._save_checkpoint()
        print("Checkpoint saved. Resume with --resume flag.")


if __name__ == '__main__':
    main()
