#!/usr/bin/env python3
"""
Domain-Specific Tests for EquiTile

Tests for:
- Vision (ConvEquiTile)
- Language Modeling (TileLM)
- Reinforcement Learning (RLEquiTile)

Usage:
    python -m pytest tests/test_equitile_domains.py -v
"""

import pytest
import torch

from bioplausible.data.lm import CharacterTokenizer
from bioplausible.equitile import (
    ConvEquiTile,  # Vision; RL
    ConvEquiTileConfig,
    RecurrentRLEquiTile,
    RLEquiTile,
    RLEquiTileConfig,
    RolloutBuffer,
    VisionAugmentation,
    compute_gae,
    create_cifar_model,
    create_mnist_model,
    create_rl_model,
)
from bioplausible.zoo.models.tile_lm import TileLM

# =============================================================================
# Vision Tests
# =============================================================================


class TestVision:
    """Tests for ConvEquiTile vision module."""

    def test_conv_equitile_config(self) -> None:
        """Test ConvEquiTileConfig."""
        config = ConvEquiTileConfig(
            input_channels=3,
            input_size=32,
            num_classes=10,
        )

        assert config.input_channels == 3
        assert config.input_size == 32
        assert config.num_classes == 10
        assert len(config.conv_channels) == 3

    def test_conv_equitile_creation(self) -> None:
        """Test ConvEquiTile creation."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=[16, 32],
        )
        model = ConvEquiTile(config)

        assert model is not None
        assert model.config.num_classes == 10

    def test_conv_equitile_forward(self) -> None:
        """Test ConvEquiTile forward pass."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=[4, 8],
            neurons_per_tile=16,
            tiles_per_layer=1,
            num_fc_layers=1,
        )
        model = ConvEquiTile(config)
        model.eval()

        # Create batch of images
        images = torch.randn(4, 1, 28, 28)

        with torch.no_grad():
            logits = model(images)

        assert logits.shape == (4, 10)

    def test_conv_equitile_train_step(self) -> None:
        """Test ConvEquiTile training step."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=[4, 8],
            neurons_per_tile=16,
            tiles_per_layer=1,
            num_fc_layers=1,
        )
        model = ConvEquiTile(config)

        images = torch.randn(4, 1, 28, 28)
        labels = torch.randint(0, 10, (4,))

        stats = model.train_step(images, labels)

        assert "loss" in stats
        assert "accuracy" in stats
        assert stats["loss"] > 0

    def test_create_mnist_model(self) -> None:
        """Test MNIST model factory."""
        model = create_mnist_model(neurons_per_tile=32)

        assert model is not None
        assert model.config.num_classes == 10

    def test_create_cifar_model(self) -> None:
        """Test CIFAR model factory."""
        model = create_cifar_model(neurons_per_tile=64)

        assert model is not None
        assert model.config.input_channels == 3
        assert model.config.input_size == 32

    def test_vision_augmentation(self) -> None:
        """Test VisionAugmentation."""
        # Test without crop (preserves shape)
        aug = VisionAugmentation(
            random_crop=False,
            random_flip=True,
            normalize=False,
        )

        images = torch.rand(4, 3, 32, 32)
        augmented = aug(images)

        assert augmented.shape == images.shape
        assert augmented.min() >= 0
        assert augmented.max() <= 1

        # Test with crop (changes shape)
        aug_crop = VisionAugmentation(
            random_crop=True,
            crop_size=28,
            random_flip=False,
            normalize=False,
        )

        augmented_crop = aug_crop(images)
        assert augmented_crop.shape == (4, 3, 28, 28)


# =============================================================================
# Language Tests
# =============================================================================


class TestLanguage:
    """Tests for the substrate-native LM module (TileLM)."""

    def test_lm_config(self) -> None:
        """Test from_lm config mapping."""
        model = TileLM.from_lm(
            vocab_size=1000,
            embed_dim=128,
            num_layers=2,
        )
        config = model.get_config()
        assert config.input_dim == 128
        assert config.output_dim == 128
        assert config.num_hidden_layers == 2
        assert config.extra["vocab_size"] == 1000

    def test_lm_creation(self) -> None:
        """Test TileLM creation."""
        model = TileLM.from_lm(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            max_seq_len=16,
        )
        assert model is not None
        assert model.lm_extra.vocab_size == 100

    def test_lm_equitile_forward(self) -> None:
        """Test TileLM forward pass."""
        model = TileLM.from_lm(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            max_seq_len=16,
        )
        model.eval()

        # Create batch of token IDs
        input_ids = torch.randint(0, 100, (4, 16))

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (4, 16, 100)

    def test_lm_equitile_train_step(self) -> None:
        """Test TileLM training step."""
        torch.manual_seed(42)
        model = TileLM.from_lm(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            max_seq_len=16,
        )

        input_ids = torch.randint(0, 100, (4, 16))
        target_ids = torch.randint(0, 100, (4, 16))

        # Verify forward pass is numerically stable first
        with torch.no_grad():
            logits = model(input_ids)
        assert not logits.isnan().any(), "Forward pass produced NaN"

        stats = model.train_step(input_ids, target_ids)

        assert "loss" in stats
        assert "perplexity" in stats
        loss = stats["loss"]
        assert loss > 0, f"Expected positive loss, got {loss}"

    def test_lm_tokenizer(self) -> None:
        """Test the canonical character tokenizer."""
        tokenizer = CharacterTokenizer()

        # Test encoding
        text = "hello world"
        encoded = tokenizer.encode(text)
        assert len(encoded) > 0

        # Test decoding
        decoded = tokenizer.decode(encoded)
        assert len(decoded) > 0

        # Test batch encoding
        texts = ["hello", "world"]
        batch = tokenizer.batch_encode(texts, max_length=10)
        assert batch.shape == (2, 10)

    def test_from_lm_factory(self) -> None:
        """Test from_lm factory."""
        model = TileLM.from_lm(vocab_size=500, embed_dim=128, num_layers=2)

        assert model is not None
        assert model.get_config().input_dim == 128
        assert model.get_config().num_hidden_layers == 2

    def test_lm_generate(self) -> None:
        """Test TileLM generation."""
        model = TileLM.from_lm(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            max_seq_len=16,
        )
        model.eval()

        input_ids = torch.randint(1, 50, (1, 8))  # Start with some tokens

        with torch.no_grad():
            generated = model.generate(input_ids, max_length=16)

        assert generated.shape[1] == 16
        assert generated.shape[0] == 1


# =============================================================================
# RL Tests
# =============================================================================


class TestRL:
    """Tests for RLEquiTile RL module."""

    def test_rl_equitile_config(self) -> None:
        """Test RLEquiTileConfig."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )

        assert config.obs_dim == 8
        assert config.action_dim == 4
        assert config.action_type == "discrete"

    def test_rl_equitile_discrete_creation(self) -> None:
        """Test RLEquiTile with discrete actions."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)

        assert model is not None
        assert model.config.action_type == "discrete"

    def test_rl_equitile_continuous_creation(self) -> None:
        """Test RLEquiTile with continuous actions."""
        config = RLEquiTileConfig(
            obs_dim=12,
            action_dim=6,
            action_type="continuous",
        )
        model = RLEquiTile(config)

        assert model is not None
        assert model.config.action_type == "continuous"

    def test_rl_equitile_act_discrete(self) -> None:
        """Test RLEquiTile action selection (discrete)."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)
        model.eval()

        obs = torch.randn(1, 8)

        with torch.no_grad():
            action, value, log_prob = model.act(obs)

        assert action.shape == (1,)
        assert value.shape == (1,)
        # Log prob can be [1] or [1, 1] depending on torch version/distribution
        assert log_prob.reshape(-1).shape == (1,)
        assert action.item() in range(4)

    def test_rl_equitile_act_continuous(self) -> None:
        """Test RLEquiTile action selection (continuous)."""
        config = RLEquiTileConfig(
            obs_dim=12,
            action_dim=6,
            action_type="continuous",
        )
        model = RLEquiTile(config)
        model.eval()

        obs = torch.randn(1, 12)

        with torch.no_grad():
            action, value, log_prob = model.act(obs)

        assert action.shape == (1, 6)
        assert value.shape == (1,)
        assert log_prob.shape == (
            1,
            1,
        )  # continuous: 2D action -> 2D log_prob (keepdim)

    def test_rl_equitile_evaluate_actions(self) -> None:
        """Test RLEquiTile action evaluation."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)

        obs = torch.randn(4, 8)
        actions = torch.randint(0, 4, (4,))

        log_prob, entropy, value = model.evaluate_actions(obs, actions)

        # For discrete: log_prob and entropy are per-sample (no sum)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_rl_equitile_train_step(self) -> None:
        """Test RLEquiTile training step."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RLEquiTile(config)

        obs = torch.randn(4, 8)
        actions = torch.randint(0, 4, (4,))
        advantages = torch.randn(4, 1)
        returns = torch.randn(4)
        old_log_probs = torch.randn(4, 1)

        stats = model.train_step(obs, actions, advantages, returns, old_log_probs)

        assert "total_loss" in stats
        assert "policy_loss" in stats
        assert "value_loss" in stats

    def test_recurrent_rl_equitile(self) -> None:
        """Test RecurrentRLEquiTile."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )
        model = RecurrentRLEquiTile(config, rnn_hidden_dim=64)

        assert model is not None
        assert model.rnn_hidden_dim == 64

        # Test hidden state reset
        model.reset_hidden(batch_size=4, device=torch.device("cpu"))
        assert model._hidden_state is not None

    def test_rollout_buffer(self) -> None:
        """Test RolloutBuffer."""
        buffer = RolloutBuffer(obs_dim=8, action_dim=4)

        # Add some transitions
        for _ in range(10):
            buffer.add(
                obs=torch.randn(8),
                action=torch.tensor(1),
                reward=torch.tensor(1.0),
                done=torch.tensor(0),
                value=torch.tensor(0.5),
                log_prob=torch.tensor(-1.0),
            )

        assert len(buffer) == 10

        # Get data with GAE
        obs, actions, advantages, returns, log_probs = buffer.get(
            gamma=0.99,
            lam=0.95,
            last_value=0.0,
        )

        assert obs.shape[0] == 10
        assert advantages.shape[0] == 10
        assert returns.shape[0] == 10

        # Buffer should be cleared
        assert len(buffer) == 0

    def test_compute_gae(self) -> None:
        """Test GAE computation."""
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        dones = torch.tensor([0, 0, 0, 0])

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=0.99,
            lam=0.95,
            last_value=0.0,
        )

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert advantages.mean() > 0  # Should be positive for positive rewards

    def test_create_rl_model_discrete(self) -> None:
        """Test RL model factory (discrete)."""
        model = create_rl_model(
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
        )

        assert model is not None
        assert model.config.action_type == "discrete"

    def test_create_rl_model_continuous(self) -> None:
        """Test RL model factory (continuous)."""
        model = create_rl_model(
            obs_dim=12,
            action_dim=6,
            action_type="continuous",
        )

        assert model is not None
        assert model.config.action_type == "continuous"


# =============================================================================
# Integration Tests
# =============================================================================


class TestDomainIntegration:
    """Integration tests across domains."""

    def test_vision_to_rl_pipeline(self) -> None:
        """Test vision features can feed into RL."""
        # Create vision model
        vision_config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=[4, 8],
            neurons_per_tile=16,
            tiles_per_layer=1,
            num_fc_layers=1,
        )
        vision_model = ConvEquiTile(vision_config)

        # Get feature dimension from vision model
        feature_dim = vision_model.feature_extractor.output_size

        # Create RL model with matching obs_dim
        rl_config = RLEquiTileConfig(
            obs_dim=feature_dim,
            action_dim=4,
            action_type="discrete",
        )
        rl_model = RLEquiTile(rl_config)

        # Process image through vision model
        images = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            vision_output = vision_model.extract_features(images)

        # Use vision output as RL observation
        obs = vision_output
        with torch.no_grad():
            action, value, log_prob = rl_model.act(obs)

        assert action.shape[0] == 4

    def test_language_to_rl_pipeline(self) -> None:
        """Test language features can feed into RL."""
        # Create language model
        lm_model = TileLM.from_lm(
            vocab_size=100,
            embed_dim=32,
            num_layers=1,
            max_seq_len=16,
        )

        # Create RL model
        rl_config = RLEquiTileConfig(
            obs_dim=32,  # Use LM embed dim
            action_dim=4,
            action_type="discrete",
        )
        rl_model = RLEquiTile(rl_config)

        # Process text through language model
        input_ids = torch.randint(0, 100, (4, 16))
        with torch.no_grad():
            hidden = lm_model.get_hidden_states(input_ids)

        # Use last hidden state as RL observation
        obs = hidden[:, -1, :]  # Last token
        with torch.no_grad():
            action, value, log_prob = rl_model.act(obs)

        assert action.shape[0] == 4


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
