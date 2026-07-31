#!/usr/bin/env python3
"""
TinyStories Training Script
===========================

Large-scale validation of EquiTile on a proper language modeling task.

TinyStories: Synthetic stories dataset demonstrating compositional generalization.
- ~10M tokens training set
- Proper vocabulary (not character-level)
- Tests real language understanding

Usage
-----
# Train EquiTile
python -m bioplausible.equitile.lm.train_tinystories \
    --model equitile \
    --epochs 10 \
    --batch-size 64 \
    --device cuda

# Train NanoGPT baseline
python -m bioplausible.equitile.lm.train_tinystories \
    --model nanogpt \
    --epochs 10 \
    --batch-size 64 \
    --device cuda
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bioplausible.equitile.benchmarks.compare_nanoGPT import NanoGPTConfig, NanoGPTModel
from bioplausible.equitile.lm import (
    BPETokenizer,
    FastLMConfig,
    FastLMEquiTile,
)
from bioplausible.equitile.lm.training import (
    create_training_config,
    train_model as train_lm_model,
)

logger = logging.getLogger(__name__)

# =============================================================================
# TinyStories Dataset
# =============================================================================


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for language modeling.

    Parameters
    ----------
    data_path : str
        Path to TinyStories JSONL file
    tokenizer : BPETokenizer
        Tokenizer to use
    seq_length : int
        Sequence length for training
    max_samples : int, optional
        Maximum samples to load (for testing)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        seq_length: int = 256,
        max_samples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Load and tokenize stories
        logger.info("Loading TinyStories from %s...", data_path)
        self.tokens = []

        count = 0
        with Path(data_path).open() as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break

                data = json.loads(line)
                story = data.get("story", "")

                if len(story) > 50:  # Filter very short stories
                    tokens = tokenizer.encode(story)
                    self.tokens.extend(tokens)
                    count += 1

        logger.info("Loaded %d stories, %s tokens", count, f"{len(self.tokens):,}")

        # Calculate number of sequences
        self.n_sequences = max(0, len(self.tokens) // seq_length)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_length
        end = start + self.seq_length + 1

        chunk = self.tokens[start:end]

        # Pad if necessary
        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (
                self.seq_length + 1 - len(chunk)
            )

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids


def download_tinystories(data_dir: str = "data") -> str:
    """Download TinyStories dataset if not present."""
    data_path = Path(data_dir) / "TinyStories"
    train_file = data_path / "TinyStories_all_data.txt"

    if train_file.exists():
        return str(train_file)

    logger.warning("TinyStories not found at %s", train_file)
    logger.warning(
        "Please download from: https://huggingface.co/datasets/roneneldan/TinyStories"
    )
    logger.warning("Then extract to %s", data_path)

    # Create dummy data for testing
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating dummy dataset for testing...")
    dummy_stories = [
        {"story": "Once upon a time, there was a little cat named Whiskers. " * 100},
        {"story": "A brave knight fought a dragon and saved the princess. " * 100},
        {"story": "The sun was shining and the birds were singing. " * 100},
    ]

    with Path(train_file).open("w") as f:
        for story in dummy_stories * 100:  # Repeat for more data
            f.write(json.dumps(story) + "\n")

    return str(train_file)


# =============================================================================
# Training
# =============================================================================


def create_equitile_model(vocab_size: int, device: torch.device) -> nn.Module:
    """Create EquiTile model with optimized config."""
    config = FastLMConfig(
        vocab_size=vocab_size,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        num_kv_heads=4,
        hidden_dim=512,
        neurons_per_tile=64,
        tiles_per_layer=4,
        mot_k=4,  # Use all tiles for best quality
        max_seq_len=256,
        use_gradient_checkpointing=True,
        use_compile=False,
    )

    model = FastLMEquiTile(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("EquiTile parameters: %s", f"{param_count:,}")
    return model


def create_nanogpt_model(vocab_size: int, device: torch.device) -> nn.Module:
    """Create NanoGPT model with matched parameters."""
    # Match EquiTile parameters (~3.5M)
    config = NanoGPTConfig(
        vocab_size=vocab_size,
        block_size=256,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1,
    )

    model = NanoGPTModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("NanoGPT parameters: %s", f"{param_count:,}")

    return model


def train_model(
    _model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[dict[str, float]]:
    """Train model using production LMTrainer."""
    config = create_training_config(
        epochs=epochs,
        learning_rate=learning_rate,
        use_amp=False,
        log_every=50,
    )
    config.device = (
        device.type
        if device.type != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    metrics = train_lm_model(model, train_loader, val_loader, config=config)

    history = []
    for i in range(len(metrics.train_loss)):
        history.append({
            "epoch": i + 1,
            "train_loss": metrics.train_loss[i],
            "val_loss": metrics.val_loss[i] if i < len(metrics.val_loss) else 0.0,
            "val_ppl": metrics.val_perplexity[i]
            if i < len(metrics.val_perplexity)
            else 0.0,
            "tokens_per_sec": metrics.tokens_per_second[i]
            if i < len(metrics.tokens_per_second)
            else 0.0,
            "epoch_time": 0.0,
        })
    return history


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train on TinyStories")
    parser.add_argument(
        "--model",
        type=str,
        choices=["equitile", "nanogpt", "both"],
        default="both",
        help="Model to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
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
        help="Learning rate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples for testing",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="Vocabulary size",
    )

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Download/load dataset
    data_file = download_tinystories(args.data_dir)

    # Train tokenizer
    logger.info("\nTraining BPE tokenizer (vocab_size=%d)...", args.vocab_size)
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    # Load a subset for tokenizer training
    sample_texts = []
    with Path(data_file).open() as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            data = json.loads(line)
            sample_texts.append(data.get("story", ""))

    tokenizer.train(sample_texts)
    vocab_size = len(tokenizer.vocab)
    logger.info("Actual vocab size: %d", vocab_size)

    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = TinyStoriesDataset(
        data_file, tokenizer, args.seq_length, args.max_samples
    )

    # Split for validation
    val_size = max(1, len(train_dataset) // 10)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info("Train batches: %d", len(train_loader))
    logger.info("Val batches: %d", len(val_loader))

    # Train models
    results = {}

    if args.model in ["equitile", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("Training EquiTile")
        logger.info("=" * 70)

        model = create_equitile_model(vocab_size, device)
        history = train_model(
            "EquiTile",
            model,
            train_loader,
            val_loader,
            args.epochs,
            args.learning_rate,
            device,
        )
        results["equitile"] = history

    if args.model in ["nanogpt", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("Training NanoGPT")
        logger.info("=" * 70)

        model = create_nanogpt_model(vocab_size, device)
        history = train_model(
            "NanoGPT",
            model,
            train_loader,
            val_loader,
            args.epochs,
            args.learning_rate,
            device,
        )
        results["nanogpt"] = history

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    results_file = (
        output_dir / f"tinystories_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with Path(results_file).open("w") as f:
        json.dump(
            {
                "config": vars(args),
                "vocab_size": vocab_size,
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info("\nResults saved to %s", results_file)

    # Summary
    if "equitile" in results and "nanogpt" in results:
        logger.info("\n" + "=" * 70)
        logger.info("FINAL COMPARISON")
        logger.info("=" * 70)

        eq_final = results["equitile"][-1]
        ng_final = results["nanogpt"][-1]

        logger.info("EquiTile Val PPL: %.2f", eq_final["val_ppl"])
        logger.info("NanoGPT Val PPL:  %.2f", ng_final["val_ppl"])
        logger.info("EquiTile Tok/s:   %s", f"{eq_final['tokens_per_sec']:,.0f}")
        logger.info("NanoGPT Tok/s:    %s", f"{ng_final['tokens_per_sec']:,.0f}")

        ppl_ratio = ng_final["val_ppl"] / eq_final["val_ppl"]
        speed_ratio = eq_final["tokens_per_sec"] / ng_final["tokens_per_sec"]

        logger.info(
            "\nPPL Improvement: %.2fx %s",
            ppl_ratio,
            "(EquiTile better)" if ppl_ratio > 1 else "(NanoGPT better)",
        )
        logger.info("Speedup: %.2fx", speed_ratio)


if __name__ == "__main__":
    main()
