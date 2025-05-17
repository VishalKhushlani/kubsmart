"""
Script to train a DQN agent on a Kubernetes environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from kubernetes_environment_rl import K8SEnv


class QNetwork(nn.Module):
    """
    Deep Q-Network mapping states to action-value estimates.
    """

    def __init__(self, n_inputs: int, n_actions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQN:
    """
    DQN algorithm with target network and epsilon-greedy exploration.
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-2,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.q_network = QNetwork(n_inputs, n_actions)
        self.target_network = QNetwork(n_inputs, n_actions)
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=lr
        )
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def get_action(self, state) -> int:
        """
        Select action using epsilon-greedy strategy.
        """
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(
                0, self.n_actions, (1,)
            ).item()

        state_t = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return torch.argmax(q_values, dim=1).item()

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ) -> None:
        """
        Update Q-network parameters using MSE loss.
        """
        state_t = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)
        next_t = torch.tensor(
            next_state, dtype=torch.float32
        ).unsqueeze(0)
        reward_t = torch.tensor(
            reward, dtype=torch.float32
        )
        action_t = torch.tensor(
            action, dtype=torch.int64
        )

        current_q = self.q_network(state_t)[0, action_t]
        with torch.no_grad():
            max_next_q = self.target_network(next_t).max(
                dim=1
            )[0]
        target_q = (
            reward_t + (1 - done) * self.gamma * max_next_q
        )

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def update_target_network(self) -> None:
        """
        Sync target network with the main Q-network.
        """
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )


def train_dqn(
    env,
    agent: DQN,
    episodes: int,
    update_interval: int = 10,
) -> DQN:
    """
    Train DQN agent over specified episodes.
    """
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for _ in range(100):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )

            total_reward += reward
            state = next_state

            if done:
                break

        if ep % update_interval == 0:
            agent.update_target_network()

        print(f"Episode {ep}, Total Reward: {total_reward}")

    return agent


def main() -> None:
    """
    Entry point for DQN training.
    """
    # Hyperparameters
    n_inputs = 24
    n_actions = 8
    episodes = 500
    namespace = 'default'

    env = K8SEnv(namespace)
    agent = DQN(
        n_inputs,
        n_actions,
    )
    train_dqn(
        env,
        agent,
        episodes,
    )


if __name__ == '__main__':
    main()
