"""
RL Environment Adapter for EqProp Evolution

Wraps OpenAI Gym environments for evaluation with EqProp models.
Provides standardized interface for RL task evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        gym = None


logger = logging.getLogger(__name__)


class RLEvaluator:
    """Evaluate EqProp models on RL environments."""
    
    def __init__(self, env_name: str, device: str = 'cpu'):
        if gym is None:
            raise ImportError("gym/gymnasium not available")
        
        self.env_name = env_name
        self.device = device
        
        # Create environment
        try:
            self.env = gym.make(env_name)
        except:
            # Try old gym API
            self.env = gym.make(env_name)
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def evaluate_policy(
        self,
        model: nn.Module,
        n_episodes: int = 10,
        max_steps: int = 500,
        deterministic: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate model as RL policy.
        
        Args:
            model: EqProp model to evaluate
            n_episodes: Number of episodes
            max_steps: Max steps per episode
            deterministic: Use argmax instead of sampling
            
        Returns:
            (mean_reward, success_rate)
        """
        model.eval()
        episode_rewards = []
        
        for ep in range(n_episodes):
            obs, _ = self.env.reset() if hasattr(self.env, 'reset') else (self.env.reset(), None)
            episode_reward = 0
            
            for step in range(max_steps):
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Get action from model
                with torch.no_grad():
                    action_logits = model(obs_tensor)
                    
                    if deterministic:
                        action = action_logits.argmax(dim=-1).item()
                    else:
                        # Sample from softmax
                        probs = torch.softmax(action_logits, dim=-1)
                        action = torch.multinomial(probs, 1).item()
                
                # Step environment
                result = self.env.step(action)
                if len(result) == 5:  # New gym API
                    obs, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:  # Old gym API
                    obs, reward, done, _ = result
                
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        
        # Normalize to [0, 1] based on env
        success_rate = self._normalize_reward(mean_reward)
        
        return mean_reward, success_rate
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to [0, 1] based on environment."""
        # Environment-specific normalization
        if 'CartPole' in self.env_name:
            # Max reward is 500 (or 200 for v0)
            return min(reward / 200.0, 1.0)
        elif 'Acrobot' in self.env_name:
            # Reward is in [-500, 0], success is -100
            return max((reward + 500) / 500.0, 0.0)
        elif 'MountainCar' in self.env_name:
            # Reward is in [-200, 0], success is reaching goal
            return max((reward + 200) / 200.0, 0.0)
        else:
            # Generic: assume positive is better
            return max(min(reward / 100.0, 1.0), 0.0)
    
    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


def train_rl_model(
    model: nn.Module,
    env_name: str,
    n_episodes: int = 100,
    device: str = 'cpu',
    lr: float = 1e-3,
) -> float:
    """
    Quick training for RL model using basic policy gradient.
    
    This is a simple baseline trainer for smoke testing.
    Full RL training would be more sophisticated.
    """
    evaluator = RLEvaluator(env_name, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_reward = -float('inf')
    
    for episode in range(n_episodes):
        obs, _ = evaluator.env.reset() if hasattr(evaluator.env, 'reset') else (evaluator.env.reset(), None)
        
        log_probs = []
        rewards = []
        
        for step in range(500):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Forward
            action_logits = model(obs_tensor)
            probs = torch.softmax(action_logits, dim=-1)
            
            # Sample action
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Step
            result = evaluator.env.step(action.item())
            if len(result) == 5:
                obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                obs, reward, done, _ = result
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done:
                break
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss = loss - log_prob * G
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode_reward = sum(rewards)
        best_reward = max(best_reward, episode_reward)
    
    evaluator.close()
    
    # Return normalized success rate
    return evaluator._normalize_reward(best_reward)


# Environment name mapping
ENV_NAMES = {
    'cartpole': 'CartPole-v1',
    'acrobot': 'Acrobot-v1',
    'mountaincar': 'MountainCar-v0',
}


def get_env_name(task: str) -> str:
    """Get gym environment name from task."""
    return ENV_NAMES.get(task, task)
