#!/usr/bin/env python3
"""
FastLMEquiTile Demo Script
===========================

High-performance language modeling demo showcasing EquiTile's unique advantages.

Features:
- Real-time metrics dashboard
- Automatic checkpointing and resume
- Multiple dataset support (Shakespeare, TinyStories, Python)
- Comparison mode for benchmarking
- Export trained models for inference

Usage
-----
Basic training:
    python -m bioplausible.equitile.lm.demo --task shakespeare --epochs 5

With custom settings:
    python -m bioplausible.equitile.lm.demo \
        --task shakespeare \
        --epochs 10 \
        --batch-size 64 \
        --learning-rate 3e-4 \
        --device cuda

Comparison mode:
    python -m bioplausible.equitile.lm.demo \
        --task shakespeare \
        --compare \
        --epochs 5

Resume training:
    python -m bioplausible.equitile.lm.demo \
        --task shakespeare \
        --resume checkpoints/checkpoint_500.pt
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from bioplausible.core.logging import get_logger
from bioplausible.core.utils.device import get_device
from bioplausible.equitile.lm.data import (
    CharacterTokenizer,
    Tokenizer,
    create_custom_dataset,
    create_shakespeare_dataset,
)
from bioplausible.equitile.lm.fast_lm import (
    FastLMConfig,
    FastLMEquiTile,
    create_fast_lm_shakespeare,
    create_fast_lm_small,
    create_fast_lm_tiny,
)
from bioplausible.equitile.lm.training import (
    LMTrainer,
    TrainingConfig,
    TrainingMetrics,
)

logger = get_logger()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# =============================================================================
# Real-time Metrics Dashboard
# =============================================================================


class MetricsDashboard:
    """Real-time training metrics dashboard.

    Displays training progress with:
    - Loss and perplexity curves
    - Learning rate schedule
    - Throughput metrics
    - Generated samples
    - Tile importance visualization
    """

    def __init__(
        self,
        log_dir: str = "logs",
        enable_console: bool = True,
        enable_json: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.enable_console = enable_console
        self.enable_json = enable_json

        # Metrics storage
        self.history: dict[str, list[float]] = {
            "step": [],
            "train_loss": [],
            "val_loss": [],
            "train_ppl": [],
            "val_ppl": [],
            "learning_rate": [],
            "tokens_per_sec": [],
            "samples_per_sec": [],
        }

        # Generated samples
        self.generations: list[dict[str, object]] = []

        # Tile importance
        self.tile_importance: list[list[float]] = []

        # Timing
        self.start_time = time.time()
        self.last_log_time = time.time()

        # Log file
        self.log_file = self.log_dir / "training.log"

    def log(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        learning_rate: float = 0.0,
        tokens_per_sec: float = 0.0,
        samples_per_sec: float = 0.0,
        generated_text: str | None = None,
        tile_importance: list[float] | None = None,
    ) -> None:
        """Log metrics for current step."""
        # Update history
        self.history["step"].append(step)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss if val_loss else train_loss)
        self.history["train_ppl"].append(torch.exp(torch.tensor(train_loss)).item())
        self.history["val_ppl"].append(
            torch.exp(torch.tensor(val_loss)).item() if val_loss else float("inf")
        )
        self.history["learning_rate"].append(learning_rate)
        self.history["tokens_per_sec"].append(tokens_per_sec)
        self.history["samples_per_sec"].append(samples_per_sec)

        # Store generation
        if generated_text:
            self.generations.append({
                "step": step,
                "text": generated_text,
                "timestamp": time.time(),
            })

        # Store tile importance
        if tile_importance:
            self.tile_importance.append(tile_importance)

        # Console output
        if self.enable_console:
            self._print_console(
                step,
                epoch,
                train_loss,
                val_loss,
                learning_rate,
                tokens_per_sec,
                generated_text,
            )

        # JSON log
        if self.enable_json:
            self._write_json()

    def _print_console(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        learning_rate: float,
        tokens_per_sec: float,
        generated_text: str | None,
    ) -> None:
        """Print formatted console output."""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed / 60:.1f}m"

        # Format loss display
        loss_str = f"{train_loss:.4f}"
        ppl_str = f"{torch.exp(torch.tensor(train_loss)).item():.2f}"

        if val_loss is not None:
            loss_str += f" (val: {val_loss:.4f})"
            ppl_str += f" ({torch.exp(torch.tensor(val_loss)).item():.2f})"

        # Print status line
        logger.info(
            "[%s] Epoch %d | Step %d | Loss: %s | PPL: %s | LR: %.2e | Throughput: %.0f tok/s",
            elapsed_str,
            epoch,
            step,
            loss_str,
            ppl_str,
            learning_rate,
            tokens_per_sec,
        )

        # Print generation
        if generated_text:
            # Clean up text for display
            clean_text = generated_text.replace("\n", "\\n")[:80]
            logger.info("  Generated: %s...", clean_text)

        # Flush
        with Path(self.log_file).open("a", encoding="utf-8") as f:
            f.write(
                f"[{elapsed_str}] Step {step}: Loss={train_loss:.4f}, PPL={ppl_str}\n"
            )

    def _write_json(self) -> None:
        """Write metrics to JSON file."""
        metrics_path = self.log_dir / "metrics.json"

        data = {
            "history": self.history,
            "generations": self.generations[-10:],  # Last 10 generations
            "tile_importance": self.tile_importance[-100:],  # Last 100 steps
            "summary": self.get_summary(),
        }

        with Path(metrics_path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> dict[str, object]:
        """Get summary statistics."""
        if not self.history["step"]:
            return {}

        return {
            "total_steps": len(self.history["step"]),
            "best_train_loss": min(self.history["train_loss"]),
            "best_val_loss": (
                min(self.history["val_loss"]) if any(self.history["val_loss"]) else None
            ),
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": (
                self.history["val_loss"][-1]
                if self.history["val_loss"][-1] != float("inf")
                else None
            ),
            "avg_throughput": sum(self.history["tokens_per_sec"])
            / len(self.history["tokens_per_sec"]),
            "total_time": time.time() - self.start_time,
        }

    def plot(self, save_path: str | None = None) -> None:
        """Generate plots (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "Matplotlib not available. Install with: pip install matplotlib"
            )
            return

        _fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Loss curve
        ax = axes[0, 0]
        ax.plot(self.history["step"], self.history["train_loss"], label="Train")
        if any(v != float("inf") for v in self.history["val_loss"]):
            ax.plot(self.history["step"], self.history["val_loss"], label="Val")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Perplexity curve
        ax = axes[0, 1]
        ax.plot(self.history["step"], self.history["train_ppl"], label="Train")
        if any(v != float("inf") for v in self.history["val_ppl"]):
            ax.plot(self.history["step"], self.history["val_ppl"], label="Val")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[0, 2]
        ax.plot(self.history["step"], self.history["learning_rate"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

        # Throughput
        ax = axes[1, 0]
        ax.plot(self.history["step"], self.history["tokens_per_sec"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Training Throughput")
        ax.grid(True, alpha=0.3)

        # Tile importance (if available)
        ax = axes[1, 1]
        if self.tile_importance:
            importance_array = torch.tensor(self.tile_importance)
            ax.imshow(importance_array.T, aspect="auto", cmap="viridis")
            ax.set_xlabel("Step")
            ax.set_ylabel("Tile")
            ax.set_title("Tile Importance Over Time")

        # Generation quality (sample length over time)
        ax = axes[1, 2]
        if self.generations:
            lengths = [len(g["text"]) for g in self.generations]
            steps = [g["step"] for g in self.generations]
            ax.plot(steps, lengths)
            ax.set_xlabel("Step")
            ax.set_ylabel("Generated Length")
            ax.set_title("Generated Sample Length")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info("Plots saved to %s", save_path)
        else:
            plt.savefig(self.log_dir / "training_plots.png", dpi=150)

        plt.close()


# =============================================================================
# Demo Configuration
# =============================================================================


@dataclass
class DemoConfig:
    """Configuration for the demo script."""

    # Task
    task: str = "shakespeare"
    data_path: str | None = None

    # Model
    model_size: str = "small"  # tiny, small, medium
    embed_dim: int = 192
    num_layers: int = 6
    neurons_per_tile: int = 48
    tiles_per_layer: int = 4
    mot_k: int = 2

    # Attention
    attention_type: str = "auto"
    sliding_window: int = 0
    num_heads: int = 6
    num_kv_heads: int = 2

    # Training
    epochs: int = 10
    batch_size: int = 32
    seq_length: int = 256
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1

    # Optimization
    use_amp: bool = True
    use_compile: bool = False
    compile_mode: str = "max-autotune"
    gradient_clip: float = 1.0

    # Hardware
    device: str = "auto"
    num_workers: int = 4

    # Output
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    generate_samples: bool = True

    # Comparison mode
    compare: bool = False


def create_demo_config_from_args(args: argparse.Namespace) -> DemoConfig:
    """Create demo config from command-line arguments."""
    return DemoConfig(
        task=args.task,
        data_path=args.data_path,
        model_size=args.model_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        neurons_per_tile=args.neurons_per_tile,
        tiles_per_layer=args.tiles_per_layer,
        mot_k=args.mot_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=not args.no_amp,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        gradient_clip=args.gradient_clip,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        generate_samples=not args.no_generate,
        compare=args.compare,
        # Attention config
        attention_type=args.attention_type,
        sliding_window=args.sliding_window,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
    )


# =============================================================================
# Model Creation
# =============================================================================


def create_model(config: DemoConfig, vocab_size: int) -> FastLMEquiTile:
    """Create FastLMEquiTile model based on config."""
    if config.model_size == "tiny":
        return create_fast_lm_tiny(vocab_size=vocab_size)
    elif config.model_size == "small":
        return create_fast_lm_small(vocab_size=vocab_size)
    elif config.model_size == "medium":
        return create_fast_lm_shakespeare(vocab_size=vocab_size)
    else:
        # Custom configuration with all options
        model_config = FastLMConfig(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            neurons_per_tile=config.neurons_per_tile,
            tiles_per_layer=config.tiles_per_layer,
            mot_k=config.mot_k,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            attention_type=config.attention_type,
            sliding_window=config.sliding_window,
            use_compile=config.use_compile,
            compile_mode=config.compile_mode,
        )
        model = FastLMEquiTile(model_config)
        model._init_weights()
        return model


# =============================================================================
# Dataset Creation
# =============================================================================


def create_dataset(config: DemoConfig):
    """Create dataset based on task."""
    if config.task == "shakespeare":
        return create_shakespeare_dataset(
            batch_size=config.batch_size,
            seq_length=config.seq_length,
            num_workers=config.num_workers,
            cache_dir=config.data_path,
        )
    elif config.task == "custom" and config.data_path:
        # Load custom text file
        with Path(config.data_path).open(encoding="utf-8") as f:
            text = f.read()
        return create_custom_dataset(
            text,
            batch_size=config.batch_size,
            seq_length=config.seq_length,
            num_workers=config.num_workers,
            cache_dir=config.log_dir,
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")


# =============================================================================
# Training Function
# =============================================================================


def run_training(
    config: DemoConfig,
) -> tuple[FastLMEquiTile, TrainingMetrics, Tokenizer]:
    """Run training with the given configuration."""
    logger.info("=" * 60)
    logger.info("FastLMEquiTile Training Demo")
    logger.info("=" * 60)

    # Auto-detect device
    if config.device == "auto":
        config.device = str(get_device())
    logger.info("Device: %s", config.device)

    # Create dataset
    logger.info("\nLoading dataset: %s", config.task)
    train_loader, val_loader, tokenizer = create_dataset(config)
    logger.info("Vocabulary size: %d", tokenizer.vocab_size)
    logger.info("Train batches: %d", len(train_loader))
    logger.info("Val batches: %d", len(val_loader))

    # Create model
    logger.info("\nCreating model...")
    model = create_model(config, tokenizer.vocab_size)
    param_count = model.get_parameter_count()
    logger.info("Parameters: %s (%sM)", f"{param_count:,}", f"{param_count / 1e6:.2f}")

    # Print model architecture summary
    logger.info("\nModel Architecture:")
    logger.info("  Embed dim: %d", config.embed_dim)
    logger.info("  Layers: %d", config.num_layers)
    logger.info("  Tiles per layer: %d", config.tiles_per_layer)
    logger.info("  Active tiles (MoT-k): %d", config.mot_k)
    logger.info("  Attention heads: %d", model.config.num_heads)
    logger.info("  KV heads (GQA): %d", model.config.num_kv_heads)

    # Create training config
    training_config = TrainingConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        use_amp=config.use_amp,
        gradient_clip=config.gradient_clip,
        checkpoint_dir=config.checkpoint_dir,
        device=config.device,
        num_workers=config.num_workers,
        log_every=10,
        eval_every=50,
        generate_every=100 if config.generate_samples else 0,
    )

    # Create trainer
    trainer = LMTrainer(model, training_config)
    trainer.set_tokenizer(tokenizer)

    # Set generation prompt based on task
    if config.task == "shakespeare":
        trainer.set_generation_prompt("First Citizen:")
    else:
        trainer.set_generation_prompt("The ")

    # Create dashboard
    dashboard = MetricsDashboard(log_dir=config.log_dir)

    # Add callback for dashboard logging
    def on_step_callback(trainer, metrics):
        # Get latest generation
        generated = None
        if (
            config.generate_samples
            and metrics.global_step % training_config.generate_every == 0
        ):
            generated = trainer.generate_sample(max_length=100)

        dashboard.log(
            step=metrics.global_step,
            epoch=metrics.epoch,
            train_loss=metrics.train_loss[-1] if metrics.train_loss else 0,
            val_loss=metrics.val_loss[-1] if metrics.val_loss else None,
            learning_rate=metrics.learning_rates[-1] if metrics.learning_rates else 0,
            tokens_per_sec=(
                metrics.tokens_per_second[-1] if metrics.tokens_per_second else 0
            ),
            generated_text=generated,
        )

    trainer.add_on_step_callback(on_step_callback)

    # Resume if checkpoint exists
    resume_from = None
    checkpoint_path = Path(config.checkpoint_dir) / "final_model.pt"
    if checkpoint_path.exists():
        logger.info("\nFound existing checkpoint: %s", checkpoint_path)
        response_input = input("Resume training? (y/n): ")
        if response_input.lower() == "y":
            resume_from = str(checkpoint_path)

    # Train
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60 + "\n")

    start_time = time.time()
    metrics = trainer.train(train_loader, val_loader, resume_from=resume_from)
    total_time = time.time() - start_time

    # Generate final plots
    dashboard.plot()

    # Save final sample
    if config.generate_samples:
        logger.info("\n" + "=" * 60)
        logger.info("Final Generation Sample")
        logger.info("=" * 60)
        sample = trainer.generate_sample(max_length=300, temperature=0.8)
        logger.info(sample)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info("Total time: %.1f minutes", total_time / 60)
    logger.info("Best validation loss: %.4f", metrics.best_val_loss)
    logger.info("Best validation step: %d", metrics.best_val_step)
    logger.info("Final training loss: %.4f", metrics.train_loss[-1])
    logger.info("Final training perplexity: %.2f", metrics.train_perplexity[-1])
    logger.info(
        "Average throughput: %.0f tok/s",
        sum(metrics.tokens_per_second) / len(metrics.tokens_per_second),
    )
    logger.info("\nOutputs saved to:")
    logger.info("  Checkpoints: %s/", config.checkpoint_dir)
    logger.info("  Logs: %s/", config.log_dir)
    logger.info("  Plots: %s/training_plots.png", config.log_dir)

    return model, metrics, tokenizer


# =============================================================================
# Comparison Mode
# =============================================================================


def run_comparison(config: DemoConfig) -> None:
    """Run comparison between EquiTile and baseline."""
    logger.info("=" * 60)
    logger.info("EquiTile vs Baseline Comparison")
    logger.info("=" * 60)

    # For now, just run standard training
    # Full comparison would include NanoGPT baseline
    logger.info(
        "\nNote: Full comparison mode requires NanoGPT baseline implementation."
    )
    logger.info("Running standard training for benchmarking...\n")

    run_training(config)


# =============================================================================
# Inference Mode
# =============================================================================


def run_inference(
    checkpoint_path: str,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "auto",
) -> None:
    """Run inference with a trained model."""
    if device == "auto":
        device = str(get_device())

    logger.info("Loading model from %s...", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate model from config
    model_config = FastLMConfig(**checkpoint.get("config", {}))
    model = FastLMEquiTile(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create dummy tokenizer for inference
    tokenizer = CharacterTokenizer()

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Generate
    logger.info("\nPrompt: %s", prompt)
    logger.info("-" * 40)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

    generated = tokenizer.decode(output_ids[0].tolist())
    logger.info(generated)


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FastLMEquiTile Training Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task
    parser.add_argument(
        "--task",
        type=str,
        default="shakespeare",
        choices=["shakespeare", "custom"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to custom data file",
    )

    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium"],
        help="Model size preset",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=192,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--neurons-per-tile",
        type=int,
        default=48,
        help="Neurons per tile",
    )
    parser.add_argument(
        "--tiles-per-layer",
        type=int,
        default=4,
        help="Tiles per layer",
    )
    parser.add_argument(
        "--mot-k",
        type=int,
        default=2,
        help="Number of active tiles in MoT",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Sequence length",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Attention
    parser.add_argument(
        "--attention-type",
        type=str,
        default="auto",
        choices=["auto", "flash", "sdpa", "manual"],
        help="Attention backend",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=0,
        help="Sliding window size (0 = global)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=6,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=2,
        help="Number of K/V heads (for GQA)",
    )

    # Optimization
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Enable torch.compile",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Disable sample generation during training",
    )

    # Modes
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison mode",
    )
    parser.add_argument(
        "--inference",
        type=str,
        default=None,
        help="Run inference with checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The ",
        help="Prompt for inference",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Inference mode
    if args.inference:
        run_inference(
            args.inference,
            args.prompt,
            device=args.device,
        )
        return

    # Create config
    config = create_demo_config_from_args(args)

    # Run training or comparison
    if args.compare:
        run_comparison(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
