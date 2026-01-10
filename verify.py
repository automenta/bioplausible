#!/usr/bin/env python3
"""
TorEqProp Comprehensive Verification Suite

Complete validation of ALL research tracks from first principles.
Generates an undeniable evidence notebook with reproducible results.

RESEARCH TRACKS COVERED:
  - Core Validation (Tracks 1-3): Stability, Parity, Healing
  - Advanced Models (Tracks 4-9, 13-14): Ternary, FA, Conv, Transformer
  - Scaling & Efficiency (Tracks 12, 16-18, 23-26, 35): Memory, Quantization, Energy
  - Applications & Analysis (Tracks 19-22, 28-32, 36-40): Criticality, OOD, Diffusion
  - CIFAR-10 Breakthrough (Tracks 33-34)
"""

import argparse
from validation import Verifier

def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Comprehensive Verification Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (~2 min, smoke test)")
    parser.add_argument("--intermediate", "-i", action="store_true", help="Intermediate mode (~1 hr, directional validation)")
    parser.add_argument("--track", "-t", type=int, nargs="+", help="Run specific track(s)")
    parser.add_argument("--list", "-l", action="store_true", help="List all tracks")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds for robustness checks (redundancy)")
    parser.add_argument("--export", action="store_true", help="Export raw data to CSV")
    
    args = parser.parse_args()
    
    verifier = Verifier(quick_mode=args.quick, intermediate_mode=args.intermediate, seed=args.seed, n_seeds_override=args.seeds, export_data=args.export)
    
    if args.list:
        verifier.list_tracks()
    else:
        verifier.run_tracks(args.track)

if __name__ == "__main__":
    main()
