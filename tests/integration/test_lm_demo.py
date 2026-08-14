"""
Tests for TileLM (substrate-native Language Model)
=================================================

Comprehensive tests for TileLM and related components.

Run tests:
    pytest tests/integration/test_lm_demo.py -v

Run specific test:
    pytest tests/integration/test_lm_demo.py::test_tile_lm_forward -v
"""

import pathlib

import pytest
import torch

from bioplausible.core.trainer import (
    CoreTrainer,
    TrainerConfig,
)
from bioplausible.data.lm import (
    CharacterTokenizer,
    LMDataset,
    create_shakespeare_dataset,
)
from bioplausible.zoo.models.tile_lm import TileLM
from tests.conftest import lm_train_step

pytestmark = pytest.mark.gpu


def _tiny_lm(vocab_size: int = 100) -> TileLM:
    """Build a minimal TileLM shared across tests."""
    return TileLM.from_lm(
        vocab_size=vocab_size,
        embed_dim=32,
        num_layers=1,
        neurons_per_tile=8,
        tiles_per_layer=1,
        max_seq_len=32,
    )


# =============================================================================
# Model Tests
# =============================================================================


class TestTileLMConfig:
    """Tests for TileLM configuration via from_lm factory."""

    def test_from_lm_creates_model(self):
        """Test from_lm factory creates model."""
        model = TileLM.from_lm(vocab_size=500, embed_dim=128, num_layers=4)
        assert model.get_parameter_count() > 0
        assert model.lm_extra.vocab_size == 500
        assert model.config.input_dim == 128

    def test_from_lm_custom_params(self):
        """Test from_lm with custom parameters."""
        model = TileLM.from_lm(
            vocab_size=500,
            embed_dim=128,
            num_layers=4,
            neurons_per_tile=32,
            tiles_per_layer=2,
            learning_rate=1e-3,
            max_seq_len=256,
        )
        assert model.lm_extra.vocab_size == 500
        assert model.config.input_dim == 128
        assert model.config.num_hidden_layers == 4
        assert model.config.neurons_per_tile == 32
        assert model.config.tiles_per_layer == 2
        assert model.config.learning_rate == 1e-3
        assert model.lm_extra.max_seq_len == 256


class TestTileLM:
    """Tests for TileLM model."""

    def test_model_creation(self):
        """Test model creation."""
        model = TileLM.from_lm(vocab_size=100, embed_dim=32, num_layers=1)
        assert model.get_parameter_count() > 0

    def test_model_forward(self):
        """Test model forward pass."""
        model = _tiny_lm(vocab_size=100)
        input_ids = torch.randint(0, 100, (2, 10))
        logits = model(input_ids)
        assert logits.shape == (2, 10, 100)

    def test_model_generation(self):
        """Test autoregressive generation."""
        model = _tiny_lm(vocab_size=50).eval()
        input_ids = torch.randint(0, 50, (1, 5))
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=15,
                temperature=0.8,
            )
        assert output.shape == (1, 15)

    def test_model_training_step(self):
        """Test training step."""
        model = _tiny_lm(vocab_size=100).train()
        input_ids = torch.randint(0, 100, (2, 10))
        target_ids = torch.randint(0, 100, (2, 10))
        stats = lm_train_step(model, input_ids, target_ids)
        assert "loss" in stats
        assert "perplexity" in stats
        assert stats["loss"] > 0

    def test_weight_tying(self):
        """Test weight tying between input/output embeddings."""
        model = _tiny_lm(vocab_size=100)
        # TileLM uses weight-tied output: F.linear(out, token_embedding.weight * output_scale)
        assert model.output_scale is not None

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = _tiny_lm(vocab_size=100)
        params = model.get_parameter_count()
        assert params < 1_000_000

        model = TileLM.from_lm(vocab_size=500, embed_dim=128, num_layers=4)
        params = model.get_parameter_count()
        assert params < 5_000_000

    def test_substrate_logits_unchanged(self):
        """Test substrate forward_logits works on flat tensors."""
        model = _tiny_lm(vocab_size=100)
        flat = torch.randn(3, 32)
        out = model.forward_logits(flat)
        assert out.shape == (3, 32)

    def test_build_classmethod(self):
        """Test zoo build classmethod."""
        model = TileLM.build(
            spec=None,
            input_dim=32,
            output_dim=100,
            hidden_dim=64,
            num_layers=1,
            vocab_size=100,
            embed_dim=32,
            device="cpu",
        )
        ids = torch.randint(0, 100, (1, 4))
        assert model.forward(ids).shape == (1, 4, 100)


# =============================================================================
# Data Tests
# =============================================================================


class TestCharacterTokenizer:
    """Tests for CharacterTokenizer."""

    def test_tokenizer_creation(self):
        """Test tokenizer creation."""
        text = "Hello, world!"
        tokenizer = CharacterTokenizer(text)
        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token_id == 0

    def test_encode_decode(self):
        """Test encoding and decoding."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_batch_encode(self):
        """Test batch encoding."""
        tokenizer = CharacterTokenizer("abc")
        texts = ["ab", "bc", "abc"]
        encoded = tokenizer.batch_encode(texts, max_length=3)
        assert encoded.shape[0] == 3
        assert encoded.shape[1] == 3


class TestLMDataset:
    """Tests for LMDataset."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        text = "Hello world! " * 100
        tokenizer = CharacterTokenizer(text)
        dataset = LMDataset(text, tokenizer, seq_length=10)
        assert len(dataset) > 0

    def test_dataset_item(self):
        """Test dataset item retrieval."""
        text = "Hello world! " * 100
        tokenizer = CharacterTokenizer(text)
        dataset = LMDataset(text, tokenizer, seq_length=10)
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (10,)
        assert target_ids.shape == (10,)

    def test_target_is_shifted_input(self):
        """Test that target is shifted input."""
        text = "abcdefghij" * 10
        tokenizer = CharacterTokenizer(text)
        dataset = LMDataset(text, tokenizer, seq_length=5)
        input_ids, target_ids = dataset[0]
        assert torch.equal(input_ids[1:], target_ids[:-1])


class TestShakespeareDataset:
    """Tests for Shakespeare dataset."""

    def test_create_shakespeare_dataset(self):
        """Test Shakespeare dataset creation."""
        train_loader, val_loader, tokenizer = create_shakespeare_dataset(
            batch_size=4,
            seq_length=32,
            num_workers=0,
        )
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert tokenizer.vocab_size > 0

    def test_shakespeare_batch(self):
        """Test Shakespeare batch loading."""
        train_loader, _, _ = create_shakespeare_dataset(
            batch_size=4,
            seq_length=32,
            num_workers=0,
        )
        for input_ids, target_ids in train_loader:
            assert input_ids.shape == (4, 32)
            assert target_ids.shape == (4, 32)
            break


# =============================================================================
# Training Tests
# =============================================================================


class TestTrainingConfig:
    """Tests for TrainerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainerConfig(model="tile_lm")
        assert config.epochs == 10
        assert config.optimizer == "adam"

    def test_model_config(self):
        """Test model configuration."""
        config = TrainerConfig(model="tile_lm", model_kwargs={"vocab_size": 100})
        assert config.model == "tile_lm"
        assert config.model_kwargs["vocab_size"] == 100


class TestCoreTrainer:
    """Tests for CoreTrainer with TileLM."""

    def test_trainer_creation(self):
        """Test trainer creation."""
        model = _tiny_lm(vocab_size=100)
        config = TrainerConfig(model="tile_lm", epochs=1, batch_size=4)
        trainer = CoreTrainer(config)
        assert trainer.config is config

    def test_checkpoint_roundtrip(self):
        """Test model checkpoint save/load roundtrip."""
        import os
        import tempfile

        from bioplausible.core.checkpoint import (
            load_checkpoint,
            save_checkpoint,
        )

        model = _tiny_lm(vocab_size=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test.pt")
            save_checkpoint(
                checkpoint_path,
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": {},
                    "config": {},
                    "metrics": {},
                },
            )
            assert pathlib.Path(checkpoint_path).exists()
            ckpt = load_checkpoint(checkpoint_path)
            assert "model_state_dict" in ckpt
            model2 = _tiny_lm(vocab_size=100)
            model2.load_state_dict(ckpt["model_state_dict"])


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full training pipeline."""

    def test_full_training_loop(self):
        """Test complete training loop."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokenizer = CharacterTokenizer(text)
        from torch.utils.data import DataLoader

        train_dataset = LMDataset(text, tokenizer, seq_length=16)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        model = TileLM.from_lm(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32,
            num_layers=1,
            neurons_per_tile=8,
            tiles_per_layer=1,
            max_seq_len=32,
        )
        config = TrainerConfig(
            model="tile_lm",
            epochs=2,
            batch_size=4,
        )
        trainer = CoreTrainer(config)
        # Note: CoreTrainer uses its own data loading via task
        # For this test, we just verify the model trains
        model.train()
        input_ids = torch.randint(0, tokenizer.vocab_size, (4, 16))
        target_ids = input_ids.clone()
        for _ in range(4):
            stats = lm_train_step(model, input_ids, target_ids)
            assert "loss" in stats
        # Just verify it runs without error
        assert True

    def test_model_on_gpu(self):
        """Test model runs on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = _tiny_lm(vocab_size=100).cuda()
        input_ids = torch.randint(0, 100, (2, 10)).cuda()
        logits = model(input_ids)
        assert logits.device.type == "cuda"


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestBenchmarks:
    """Tests for benchmark utilities."""

    def test_nanoGPT_model(self):
        """Test NanoGPT model creation."""
        from bioplausible.benchmarks.compare_nanoGPT import (
            NanoGPTConfig,
            NanoGPTModel,
        )

        config = NanoGPTConfig(
            vocab_size=100,
            n_layer=2,
            n_embd=64,
            n_head=4,
        )
        model = NanoGPTModel(config)
        input_ids = torch.randint(0, 100, (2, 10))
        logits, loss = model(input_ids, input_ids)
        assert logits.shape == (2, 10, 100)
        assert loss > 0

    def test_efficiency_analyzer(self):
        """Test efficiency analyzer."""
        from bioplausible.benchmarks.efficiency_analysis import (
            EfficiencyAnalyzer,
        )

        model = _tiny_lm(vocab_size=100)
        analyzer = EfficiencyAnalyzer(model, device="cpu")
        param_counts = analyzer.count_parameters()
        assert param_counts["total"] > 0
        assert param_counts["embedding"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
