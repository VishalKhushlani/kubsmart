"""
Script to train a PPO agent on a Kubernetes environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from kubernetes_environment_rl import K8SEnv


class PolicyNetwork(nn.Module):
    """
    Policy network mapping states to action probabilities.
    """

    def __init__(self, n_inputs: int, n_actions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPO:
    """
    Proximal Policy Optimization algorithm.
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-2,
        clip_epsilon: float = 0.2,
    ) -> None:
        self.policy = PolicyNetwork(n_inputs, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def get_action(self, state) -> tuple[int, torch.Tensor]:
        """
        Sample an action and its log probability from the policy.
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(
        self,
        states: list,
        actions: list,
        old_log_probs: list,
        returns: torch.Tensor,
    ) -> None:
        """
        Perform PPO update given trajectories and returns.
        """
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64)
        old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32)

        # Compute new log probabilities
        probs = self.policy(states_t[:, -1, :])
        dist = Categorical(probs)
        new_lp_t = dist.log_prob(actions_t)

        # Probability ratio for clipping
        ratio = (new_lp_t - old_lp_t).exp()
        obj = ratio * returns
        clipped = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        ) * returns

        loss = -torch.min(obj, clipped).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_ppo(
    env: K8SEnv,
    agent: PPO,
    episodes: int,
) -> None:
    """
    Train PPO agent for a given number of episodes.
    """
    for ep in range(1, episodes + 1):
        states, actions, log_probs, rewards = [], [], [], []
        state = env.reset()

        for _ in range(100):
            action, log_prob = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)

            state = new_state
            if done:
                break

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + agent.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)

        agent.update(states, actions, log_probs, returns_t)
        print(f"Episode {ep}, Total Reward: {sum(rewards)}")


def main() -> None:
    """
    Entry point for PPO training.
    """
    n_inputs = 24
    n_actions = 8
    episodes = 500
    namespace = 'default'

    env = K8SEnv(namespace)
    agent = PPO(n_inputs, n_actions)
    train_ppo(env, agent, episodes)


if __name__ == '__main__':
    main()
