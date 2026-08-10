"""
EquiTile RL: EquiTile for Reinforcement Learning
=================================================

Extends EquiTile with reinforcement learning capabilities:
- RLEquiTile: Policy and value networks for RL
- Actor-Critic architecture with tile-based learning
- Support for discrete and continuous action spaces
- Integration with Gymnasium environments

The configuration and feature extractor now inherit from the unified
``DeploymentConfig`` hierarchy in ``deployments/base``; this module adds the
RL-specific pieces (actor/critic heads, rollout buffer, GAE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

from bioplausible.config.unified import ModelConfig
from bioplausible.core.model import BioModel
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import Domain, LocalityLevel, register_model
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer
from bioplausible.equitile.deployments import _feature_extractors as _fe
from bioplausible.equitile.deployments.base import RLDeploymentConfig

# Re-export the shared feature extractor under its historical usage.
RLFeatureExtractor = _fe.RLFeatureExtractor

__all__ = [
    "RLEquiTile",
    "RLEquiTileConfig",
    "RecurrentRLEquiTile",
    "RolloutBuffer",
    "compute_gae",
    "create_atari_model",
    "create_mujoco_model",
    "create_recurrent_rl_model",
    "create_rl_model",
]
if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class RLEquiTileConfig(RLDeploymentConfig):
    """Configuration for RL EquiTile.

    Inherits the shared deployment fields from ``RLDeploymentConfig`` and
    keeps the historical RL defaults (backprop mode, larger lr, 32 tiles).
    """

    neurons_per_tile: int = 32
    learning_rate: float = 3e-4
    mode: Literal["pc", "ep", "backprop"] = "backprop"
    inference_steps: int = 5


# =============================================================================
# RL EquiTile Network
# =============================================================================


@register_model(
    "rl_equitile",
    domains=[Domain.RL],
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.75,
    requires_backward=False,
    credit_assignment_type="hebbian",
    family="equitile",
    tags=[status_tag("experimental")],
)
class RLEquiTile(BioModel):
    """EquiTile for Reinforcement Learning.

    Implements actor-critic architecture with tile-based local learning
    for both policy and value functions.

    Parameters
    ----------
    config : RLEquiTileConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters
    """

    algorithm_name = "RLEquiTile"

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        """Build RLEquiTile from factory arguments."""
        config_kwargs = {
            "obs_dim": input_dim,
            "action_dim": output_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "learning_rate": kwargs.get("lr", spec.default_lr),
            "neurons_per_tile": kwargs.get("neurons_per_tile", 32),
            "tiles_per_layer": kwargs.get("tiles_per_layer", 4),
        }

        valid_keys = RLEquiTileConfig.__annotations__.keys()
        for k, v in kwargs.items():
            if k in valid_keys:
                config_kwargs[k] = v

        for k, v in spec.custom_hyperparams.items():
            if k in valid_keys:
                config_kwargs[k] = v

        config = RLEquiTileConfig(**config_kwargs)

        model = cls(config=config)
        return model.to(device)

    def __init__(
        self,
        config: RLEquiTileConfig | None = None,
        **kwargs: object,
    ) -> None:
        if config is None:
            config = RLEquiTileConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="rl_equitile",
                input_dim=config.obs_dim,
                output_dim=config.action_dim,
            )
        )

        self.config = config

        # Shared feature extractor (EquiTile-based, from the unified module)
        self.feature_extractor = RLFeatureExtractor(config)

        # Actor (policy) head
        self.actor = self._build_actor(config)

        # Critic (value) head
        self.critic = self._build_critic(config)

        # Optimizer
        self.optimizer = create_optimizer(
            self, OptimizerConfig(name="adam", lr=config.learning_rate)
        )

        self._init_weights()

    def _build_actor(self, config: RLEquiTileConfig) -> nn.Module:
        """Build actor (policy) head."""
        tile_dim = config.neurons_per_tile * config.tiles_per_layer

        if config.action_type == "discrete":
            return nn.Linear(tile_dim, config.action_dim)
        else:
            actor = nn.Linear(tile_dim, config.action_dim)
            nn.init.orthogonal_(actor.weight, gain=0.01)
            nn.init.zeros_(actor.bias)
            return actor

    def _build_critic(self, config: RLEquiTileConfig) -> nn.Module:
        """Build critic (value) head."""
        tile_dim = config.neurons_per_tile * config.tiles_per_layer
        critic = nn.Linear(tile_dim, 1)
        nn.init.orthogonal_(critic.weight, gain=1.0)
        nn.init.zeros_(critic.bias)
        return critic

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(
                        module.weight, gain=nn.init.calculate_gain("relu")
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def extract_features(self, obs: Tensor) -> Tensor:
        """Extract features from observation."""
        return self.feature_extractor(obs)

    def act(
        self,
        obs: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select action."""
        features = self.extract_features(obs)

        if self.config.action_type == "discrete":
            action_logits = self.actor(features)
            dist = Categorical(logits=action_logits)
        else:
            action_mean = self.actor(features)
            action_log_std = torch.clamp(
                torch.ones_like(action_mean) * self.config.log_std_init,
                self.config.log_std_min,
                self.config.log_std_max,
            )
            action_std = torch.exp(action_log_std)
            dist = Normal(action_mean, action_std)

        if deterministic:
            if self.config.action_type == "discrete":
                action = action_logits.argmax(dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()

        value = self.critic(features).squeeze(-1)

        if self.config.action_type == "discrete":
            log_prob = dist.log_prob(action)
            if log_prob.dim() == 1:
                log_prob = log_prob.unsqueeze(-1)
        else:
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, value, log_prob

    def evaluate_actions(
        self,
        obs: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions for PPO-style updates."""
        features = self.extract_features(obs)

        if self.config.action_type == "discrete":
            action_logits = self.actor(features)
            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            action_mean = self.actor(features)
            action_log_std = torch.clamp(
                torch.ones_like(action_mean) * self.config.log_std_init,
                self.config.log_std_min,
                self.config.log_std_max,
            )
            action_std = torch.exp(action_log_std)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        value = self.critic(features).squeeze(-1)

        return log_prob, entropy, value

    def get_value(self, obs: Tensor) -> Tensor:
        """Get value estimate."""
        features = self.extract_features(obs)
        return self.critic(features).squeeze(-1)

    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass (return logits).

        Compatible with generic RLTrainer (REINFORCE).
        """
        features = self.extract_features(obs)
        return self.actor(features)

    def compute_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_log_probs: Tensor,
    ) -> dict[str, Tensor]:
        """Compute PPO-style loss."""
        log_prob, entropy, value = self.evaluate_actions(obs, actions)

        ratio = torch.exp(log_prob - old_log_probs)
        clip_ratio = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(value, returns)

        entropy_loss = -entropy.mean()

        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy.mean(),
            "ratio": ratio.mean(),
        }

    def train_step(
        self,
        obs: Tensor,
        actions: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_log_probs: Tensor,
    ) -> dict[str, float]:
        """Perform one training step."""
        loss_dict = self.compute_loss(obs, actions, advantages, returns, old_log_probs)

        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()

        nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)

        self.optimizer.step()

        return {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in loss_dict.items()
        }


# =============================================================================
# Recurrent RL EquiTile (for Partial Observability)
# =============================================================================


class RecurrentRLEquiTile(RLEquiTile):
    """Recurrent EquiTile for partially observable environments.

    Adds LSTM/GRU layers for temporal memory.

    Parameters
    ----------
    config : RLEquiTileConfig
        Configuration
    rnn_type : str
        RNN type: 'lstm' or 'gru'
    rnn_hidden_dim : int
        RNN hidden dimension
    """

    def __init__(
        self,
        config: RLEquiTileConfig,
        rnn_type: Literal["lstm", "gru"] = "lstm",
        rnn_hidden_dim: int = 128,
    ) -> None:
        super().__init__(config)

        self.rnn_hidden_dim = rnn_hidden_dim

        tile_dim = config.neurons_per_tile * config.tiles_per_layer

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(tile_dim, rnn_hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(tile_dim, rnn_hidden_dim, batch_first=True)

        # Update actor and critic for rnn output
        self.actor = nn.Linear(rnn_hidden_dim, config.action_dim)
        self.critic = nn.Linear(rnn_hidden_dim, 1)

        self._hidden_state = None

    def reset_hidden(self, batch_size: int, device: torch.device) -> None:
        """Reset hidden state."""
        self._hidden_state = (
            torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device),
            torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device),
        )

    def extract_features(self, obs: Tensor) -> Tensor:
        """Extract features with recurrence."""
        base_features = self.feature_extractor(obs)

        if base_features.dim() == 2:
            base_features = base_features.unsqueeze(1)

        if self._hidden_state is not None:
            output, self._hidden_state = self.rnn(base_features, self._hidden_state)
        else:
            output, _ = self.rnn(base_features)

        return output.squeeze(1)


# =============================================================================
# RL Utilities
# =============================================================================


class RolloutBuffer:
    """Rollout buffer for on-policy RL algorithms.

    Stores trajectories for PPO-style updates.

    Parameters
    ----------
    obs_dim : int
        Observation dimension
    action_dim : int
        Action dimension
    device : torch.device
        Device
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.dones: list[Tensor] = []
        self.values: list[Tensor] = []
        self.log_probs: list[Tensor] = []

    def add(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        value: Tensor,
        log_prob: Tensor,
    ) -> None:
        """Add transition to buffer."""
        self.obs.append(obs.clone())
        self.actions.append(action.clone())
        self.rewards.append(reward.clone())
        self.dones.append(done.clone())
        self.values.append(value.clone())
        self.log_probs.append(log_prob.clone())

    def get(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        last_value: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get buffered data with GAE advantages."""
        obs = torch.stack(self.obs)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            lam=lam,
            last_value=last_value,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.clear()

        return obs, actions, advantages, returns, log_probs

    def clear(self) -> None:
        """Clear buffer."""
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.obs)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : torch.Tensor
        Rewards
    values : torch.Tensor
        Value estimates
    dones : torch.Tensor
        Done flags
    gamma : float
        Discount factor
    lam : float
        GAE lambda
    last_value : float
        Value of final state

    Returns
    -------
    tuple
        (advantages, returns)
    """
    advantages = []
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.stack(advantages)
    returns = advantages + values

    return advantages, returns


# =============================================================================
# Factory Functions
# =============================================================================


def create_rl_model(
    obs_dim: int,
    action_dim: int,
    action_type: Literal["discrete", "continuous"] = "discrete",
    hidden_dim: int = 128,
    **kwargs: object,
) -> RLEquiTile:
    """Create RLEquiTile model."""
    config = RLEquiTileConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_dim=hidden_dim,
        **kwargs,
    )
    return RLEquiTile(config)


def create_recurrent_rl_model(
    obs_dim: int,
    action_dim: int,
    action_type: Literal["discrete", "continuous"] = "discrete",
    rnn_hidden_dim: int = 128,
    **kwargs: object,
) -> RecurrentRLEquiTile:
    """Create RecurrentRLEquiTile model."""
    config = RLEquiTileConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        **kwargs,
    )
    return RecurrentRLEquiTile(config, rnn_hidden_dim=rnn_hidden_dim)


def create_atari_model(
    obs_shape: tuple[int, int, int] = (4, 84, 84),
    action_dim: int = 4,
    **kwargs: object,
) -> RLEquiTile:
    """Create RLEquiTile for Atari games.

    Note: Flattens the image observation to a 1D vector.
    """
    obs_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]

    return create_rl_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=512,
        **kwargs,
    )


def create_mujoco_model(
    obs_dim: int,
    action_dim: int,
    **kwargs: object,
) -> RLEquiTile:
    """Create RLEquiTile for MuJoCo environments (continuous action space)."""
    return create_rl_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type="continuous",
        hidden_dim=256,
        **kwargs,
    )
