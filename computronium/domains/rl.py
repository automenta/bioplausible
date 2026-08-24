"""
Reinforcement Learning Domain Tasks

Standard RL environments (CartPole, Pendulum, etc.)
"""

import torch
from torch import nn

from computronium.domains.base import (
    DomainSpec,
    DomainTask,
    DomainType,
    Metrics,
    TaskSplit,
)

__all__ = [
    "RLTask",
]


class RLTask(DomainTask):
    """Reinforcement learning domain tasks."""

    def __init__(
        self,
        name: str = "cartpole",
        env_id: str = "CartPole-v1",
        max_steps: int = 1000,
        gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.env_id = env_id
        self.max_steps = max_steps
        self.gamma = gamma
        self._env = None

    @property
    def domain_type(self) -> DomainType:
        return DomainType.RL

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.RL,
            description=f"RL task: {self.env_id}",
            default_metrics=["reward", "episode_length"],
            supported_tasks=["reinforcement_learning"],
            default_batch_size=1,  # Episodes
            default_lr=3e-4,
            tags=["rl", "control"],
        )

    def setup(self) -> None:
        try:
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "gymnasium required for RL tasks. Install with: pip install gymnasium"
            )

        self._env = gym.make(self.env_id)
        obs_space = self._env.observation_space
        act_space = self._env.action_space

        self._input_dim = (
            obs_space.shape[0] if hasattr(obs_space, "shape") else obs_space.n
        )
        if hasattr(act_space, "n"):
            self._output_dim = act_space.n
        else:
            self._output_dim = act_space.shape[0]

        self._setup_done = True

    def get_dataloader(self, split: TaskSplit) -> None:
        # RL doesn't use traditional dataloaders
        return None

    def get_batch(
        self, split: str | TaskSplit = "train", batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "RL Task does not support get_batch directly, use RLTrainer"
        )

    def create_trainer(self, model: nn.Module, **kwargs) -> object:
        from computronium.training.rl import RLTrainer

        rl_args = {}
        if "batches_per_epoch" in kwargs and "episodes_per_epoch" not in kwargs:
            kwargs["episodes_per_epoch"] = kwargs["batches_per_epoch"]
        valid_keys = [
            "episodes",
            "lr",
            "gamma",
            "max_steps",
            "tracker",
            "episodes_per_epoch",
        ]
        for k in valid_keys:
            if k in kwargs:
                rl_args[k] = kwargs[k]

        return RLTrainer(model, self.env_id, device=str(self.device), **rl_args)

    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: int | None = None,
        n_episodes: int = 10,
    ) -> Metrics:
        if self._env is None:
            self.setup()

        model.eval()
        total_rewards = []
        total_lengths = []

        for _ in range(n_episodes):
            obs, _ = self._env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done or episode_length < self.max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_logits = model(obs_tensor)
                    action = action_logits.argmax(1).item()

                obs, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)

        import numpy as np

        return Metrics(
            loss=-np.mean(total_rewards),  # Negative reward as loss
            custom={
                "mean_reward": np.mean(total_rewards),
                "std_reward": np.std(total_rewards),
                "mean_length": np.mean(total_lengths),
            },
        )
