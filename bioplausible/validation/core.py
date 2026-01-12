
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Callable

from .notebook import VerificationNotebook, TrackResult
from .tracks import core_tracks, advanced_tracks, scaling_tracks, special_tracks, hardware_tracks, analysis_tracks, application_tracks, engine_validation_tracks, enhanced_validation_tracks, new_tracks, rapid_validation, framework_validation, nebc_tracks, negative_results, architecture_comparison, honest_tradeoff, evolution_tracks

class Verifier:
    """Complete verification suite for all research tracks."""
    
    def __init__(self, quick_mode: bool = False, intermediate_mode: bool = False, seed: int = 42, n_seeds_override: Optional[int] = None, export_data: bool = False):
        self.quick_mode = quick_mode
        self.intermediate_mode = intermediate_mode
        self.seed = seed
        self.export_data = export_data
        self.notebook = VerificationNotebook()
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Validation Mode Configuration
        # Quick:        ~2 min - mechanics only (smoke test)
        # Intermediate: ~1 hour - directional validation
        # Full:         ~4+ hr  - statistically significant claims
        if quick_mode:
            self.epochs = 5
            self.n_samples = 200
            self.n_seeds = 1
            self.evidence_level = "smoke"
        elif intermediate_mode:
            self.epochs = 50
            self.n_samples = 5000
            self.n_seeds = 3
            self.evidence_level = "intermediate"
        else:
            self.epochs = 100
            self.n_samples = 10000
            self.n_seeds = 5
            self.evidence_level = "full"
        
        # Allow override of seeds
        if n_seeds_override is not None:
            self.n_seeds = n_seeds_override
            
        self.data_records = [] # For CSV export
        self.current_seed = seed # Track current seed for logging
        
        # Track definitions
        self.tracks = {
            0: ("Framework Validation", framework_validation.track_0_framework_validation),
            1: ("Spectral Normalization Stability", core_tracks.track_1_spectral_norm),
            2: ("EqProp vs Backprop Parity", core_tracks.track_2_backprop_parity),
            3: ("Adversarial Self-Healing", core_tracks.track_3_adversarial_healing),
            4: ("Ternary Weights", advanced_tracks.track_4_ternary_weights),
            5: ("Neural Cube 3D Topology", scaling_tracks.track_5_neural_cube),
            6: ("Feedback Alignment", advanced_tracks.track_6_feedback_alignment),
            7: ("Temporal Resonance", advanced_tracks.track_7_temporal_resonance),
            8: ("Homeostatic Stability", advanced_tracks.track_8_homeostatic),
            9: ("Gradient Alignment", advanced_tracks.track_9_gradient_alignment),
            # Track 10 removed - superseded by Track 26 (O(1) Memory Reality)
            # Track 11 removed - consolidated into Track 23
            12: ("Lazy Event-Driven Updates", scaling_tracks.track_12_lazy_updates),
            13: ("Convolutional EqProp", special_tracks.track_13_conv_eqprop),
            14: ("Transformer EqProp", special_tracks.track_14_transformer),
            15: ("PyTorch vs Kernel", special_tracks.track_15_kernel_comparison),
            16: ("FPGA Bit Precision", hardware_tracks.track_16_fpga_quantization),
            17: ("Analog/Photonics Noise", hardware_tracks.track_17_analog_photonics),
            18: ("DNA/Thermodynamic", hardware_tracks.track_18_thermodynamic_dna),
            19: ("Criticality Analysis", analysis_tracks.track_19_criticality),
            20: ("Transfer Learning", application_tracks.track_20_transfer_learning),
            21: ("Continual Learning", application_tracks.track_21_continual_learning),
            22: ("Golden Reference Harness", engine_validation_tracks.track_22_golden_reference),
            23: ("Comprehensive Depth Scaling", engine_validation_tracks.track_23_comprehensive_depth),
            24: ("Lazy Updates Wall-Clock", engine_validation_tracks.track_24_lazy_wallclock),
            25: ("Real Dataset Benchmark", enhanced_validation_tracks.track_25_real_dataset),
            26: ("O(1) Memory Reality", enhanced_validation_tracks.track_26_memory_reality),
            # Track 27 removed - consolidated into Track 23
            28: ("Robustness Suite", enhanced_validation_tracks.track_28_robustness_suite),
            29: ("Energy Dynamics", enhanced_validation_tracks.track_29_energy_dynamics),
            30: ("Damage Tolerance", enhanced_validation_tracks.track_30_damage_tolerance),
            31: ("Residual EqProp", enhanced_validation_tracks.track_31_residual_eqprop),
            32: ("Bidirectional Generation", enhanced_validation_tracks.track_32_bidirectional_generation),
            33: ("CIFAR-10 Benchmark", enhanced_validation_tracks.track_33_cifar10_benchmark),
            34: ("CIFAR-10 Breakthrough", new_tracks.track_34_cifar10_breakthrough),
            35: ("O(1) Memory Scaling", new_tracks.track_35_memory_scaling),
            36: ("Energy OOD Detection", new_tracks.track_36_energy_ood),
            37: ("Character LM", new_tracks.track_37_language_modeling),
            38: ("Adaptive Compute", new_tracks.track_38_adaptive_compute),
            39: ("EqProp Diffusion", new_tracks.track_39_eqprop_diffusion),
            40: ("Hardware Analysis", new_tracks.track_40_hardware_analysis),
            41: ("Rapid Rigorous Validation", rapid_validation.track_41_rapid_rigorous_validation),
            # NEBC (Nobody Ever Bothered Club) - Bio-plausible algorithms + Spectral Norm
            50: ("NEBC EqProp Variants", nebc_tracks.track_50_nebc_eqprop_variants),
            51: ("NEBC Feedback Alignment", nebc_tracks.track_51_nebc_feedback_alignment),
            52: ("NEBC Direct Feedback Alignment", nebc_tracks.track_52_nebc_direct_feedback_alignment),
            53: ("NEBC Contrastive Hebbian", nebc_tracks.track_53_nebc_contrastive_hebbian),
            54: ("NEBC Deep Hebbian Chain", nebc_tracks.track_54_nebc_deep_hebbian_chain),
            # Negative Results - Scientific Completeness
            55: ("Negative Result: Linear Chain", negative_results.track_55_negative_linear_chain),
            56: ("Depth Architecture Comparison", architecture_comparison.track_56_depth_architecture_comparison),
            # Reality Check - Should we continue?
            57: ("Honest Trade-off Analysis", honest_tradeoff.track_57_honest_tradeoff_analysis),
            # Evolution System
            60: ("Evolution vs Random Search", evolution_tracks.track_60_evolution_validation),
        }
    
    def print_header(self):
        evidence_labels = {
            "smoke": "🧪 Smoke Test (mechanics only)",
            "intermediate": "📊 Intermediate (directional)",
            "full": "✅ Full Validation (statistically significant)"
        }
        mode_name = "Quick" if self.quick_mode else ("Intermediate" if self.intermediate_mode else "Full")
        mode_icon = "⚡" if self.quick_mode else ("📊" if self.intermediate_mode else "🔬")
        
        print("=" * 70)
        print("       TOREQPROP COMPREHENSIVE VERIFICATION SUITE")
        print("       Undeniable Evidence for All Research Claims")
        print("=" * 70)
        print(f"\\n📋 Configuration:")
        print(f"   Seed: {self.seed}")
        print(f"   Mode: {mode_icon} {mode_name}")
        print(f"   Evidence: {evidence_labels[self.evidence_level]}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Samples: {self.n_samples}")
        print(f"   Seeds: {self.n_seeds}")
        print(f"   Tracks: {len(self.tracks)}")
        if self.export_data:
            print(f"   Export: Enabled (results/data.csv)")
        print("=" * 70)

    def record_metric(self, track_id: int, seed: int, step: int, metric_name: str, value: float):
        """Record a data point for export."""
        if self.export_data:
            self.data_records.append({
                "track_id": track_id,
                "seed": seed,
                "step": step,
                "metric": metric_name,
                "value": value,
                "timestamp": datetime.now().isoformat()
            })

    def evaluate_robustness(self, track_fn, n_seeds: int = 3) -> Dict:
        """Run a track logic multiple times with different seeds."""
        scores = []
        metrics_list = []
        
        # Determine number of seeds to run
        # override rules:
        # 1. if quick_mode -> 1
        # 2. if --seeds X provided -> X
        # 3. if default (3) -> use track-specific n_seeds (arg)
        
        run_count = self.n_seeds
        if self.n_seeds == 3 and not self.quick_mode:
             run_count = n_seeds
        
        print(f"      Running robustness check ({run_count} seeds)...")
        
        for i in range(run_count):
            seed = self.seed + i*100
            self.current_seed = seed # Update state for loggers
            
            # Temporarily set seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            try:
                score, metrics = track_fn()
                scores.append(score)
                metrics_list.append(metrics)
                
                # Record aggregations for export
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.record_metric(0, seed, 0, k, v) # Track ID 0 is generic/unknown here
                        
            except Exception as e:
                print(f"        Seed {seed}: Failed ({e})")
                import traceback
                traceback.print_exc()
                scores.append(0)
                metrics_list.append({})
                
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        
        # Calculate 95% Confidence Interval
        n = len(scores)
        se = std_score / np.sqrt(n) if n > 1 else 0.0
        ci_95 = 1.96 * se
        
        # Aggregate metrics with confidence intervals
        agg_metrics = {}
        if metrics_list:
            keys = metrics_list[0].keys()
            for k in keys:
                vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
                if vals:
                    m_mean = np.mean(vals)
                    m_std = np.std(vals) if len(vals) > 1 else 0.0
                    m_se = m_std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                    m_ci = 1.96 * m_se
                    agg_metrics[f"{k}_mean"] = m_mean
                    agg_metrics[f"{k}_std"] = m_std
                    agg_metrics[f"{k}_ci95"] = m_ci  # New: CI for each metric
                    
        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "ci_95": ci_95,  # New: 95% CI half-width
            "metrics": agg_metrics,
            "all_scores": scores
        }
    
    def run_tracks(self, track_ids: Optional[List[int]] = None) -> Dict:
        """Run specified tracks (or all if None)."""
        self.print_header()
        self.notebook.add_header(self.seed)
        
        if track_ids is None:
            track_ids = list(self.tracks.keys())
        
        # Auto-run Track 0 (Framework Validation) in intermediate/full modes
        if (self.intermediate_mode or (not self.quick_mode)) and 0 not in track_ids:
            print("\n⚙️  Running Track 0 (Framework Validation) automatically...")
            track_ids = [0] + track_ids
        
        results = {}
        start_time = time.time()
        
        for i, track_id in enumerate(track_ids):
            if track_id not in self.tracks:
                print(f"⚠️ Unknown track: {track_id}")
                continue
            
            name, method = self.tracks[track_id]
            
            try:
                # Pass self (Verifier) to the track method
                result = method(self)
                results[track_id] = result
                self.notebook.add_track_result(result)
                
                icon = {"pass": "✅", "fail": "❌", "partial": "⚠️", "stub": "🔧"}[result.status]
                print(f"\n{icon} Track {track_id}: {name} - {result.status.upper()} ({result.score:.0f}/100)")
                
            except Exception as e:
                print(f"\n❌ Track {track_id} failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Progress
            elapsed = time.time() - start_time
            completed = i + 1
            remaining = len(track_ids) - completed
            if remaining > 0:
                eta = (elapsed / completed) * remaining
                print(f"   Progress: {completed}/{len(track_ids)} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
        
        # Save
        total_time = time.time() - start_time
        # output path relative to original verify.py location
        output_path = Path(__file__).parent.parent / "results" / "verification_notebook.md"
        self.notebook.save(output_path)
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 VERIFICATION COMPLETE")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📓 Output: {output_path}")
        
        passed = sum(1 for r in results.values() if r.status == "pass")
        total = len(results)
        print(f"\n📊 Results: {passed}/{total} tracks passed")
        
        if self.export_data and self.data_records:
            import csv
            csv_path = Path(__file__).parent.parent / "results" / "data.csv"
            keys = self.data_records[0].keys()
            with open(csv_path, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.data_records)
            print(f"💾 Data exported to: {csv_path}")
        
        return results

    def list_tracks(self):
        """Print all available tracks."""
        print("\nAvailable Verification Tracks:")
        print("-" * 60)
        for tid, (name, _) in self.tracks.items():
            print(f"  {tid:2d}. {name}")
        print("-" * 60)
