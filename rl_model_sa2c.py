"""
Script to train a Soft Actor-Critic (SAC) agent on a Kubernetes environment.
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from kubernetes_environment_rl import K8SEnv


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network that outputs mean and log std for each action.
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_actions),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.network(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    """
    Critic network mapping state-action pairs to Q-values.
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs + n_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SoftActorCritic:
    """
    Soft Actor-Critic (SAC) algorithm for continuous action spaces.
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        policy_update_interval: int = 2,
    ) -> None:
        self.actor = GaussianPolicy(n_inputs, n_actions)
        self.critic1 = QNetwork(n_inputs, n_actions)
        self.critic2 = QNetwork(n_inputs, n_actions)
        self.critic1_target = QNetwork(n_inputs, n_actions)
        self.critic2_target = QNetwork(n_inputs, n_actions)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=critic_lr
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=critic_lr
        )

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_update_interval = policy_update_interval
        self.update_step = 0
        self.n_actions = n_actions

    def select_action(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        state_t = torch.tensor(state, dtype=torch.float32)
        action, _ = self.actor.sample(state_t)
        return action

    def update(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) -> None:
        # Convert batches to tensors
        state_t = torch.tensor(states, dtype=torch.float32)
        next_t = torch.tensor(next_states, dtype=torch.float32)
        action_t = torch.tensor(actions, dtype=torch.float32)
        reward_t = torch.tensor(rewards, dtype=torch.float32)
        done_t = torch.tensor(dones, dtype=torch.float32)

        # Compute target Q-value
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_t)
            q1_next = self.critic1_target(next_t, next_action)
            q2_next = self.critic2_target(next_t, next_action)
            q_next = torch.min(q1_next, q2_next)
            value_target = (
                reward_t + (1 - done_t) * self.gamma *
                (q_next - self.alpha * next_log_prob)
            )

        # Compute current Q estimates
        q1 = self.critic1(state_t, action_t)
        q2 = self.critic2(state_t, action_t)

        # Critic losses
        q1_loss = nn.functional.mse_loss(q1, value_target)
        q2_loss = nn.functional.mse_loss(q2, value_target)

        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy update
        if self.update_step % self.policy_update_interval == 0:
            actions_pred, log_prob_pred = self.actor.sample(state_t)
            q1_pred = self.critic1(state_t, actions_pred)
            q2_pred = self.critic2(state_t, actions_pred)
            q_pred = torch.min(q1_pred, q2_pred)
            actor_loss = (
                self.alpha * log_prob_pred - q_pred
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target critics
            for target, source in zip(
                self.critic1_target.parameters(),
                self.critic1.parameters(),
            ):
                target.data.copy_(
                    self.tau * source.data + (1 - self.tau) * target.data
                )
            for target, source in zip(
                self.critic2_target.parameters(),
                self.critic2.parameters(),
            ):
                target.data.copy_(
                    self.tau * source.data + (1 - self.tau) * target.data
                )

        self.update_step += 1

    def train(
        self,
        env: K8SEnv,
        episodes: int,
        batch_size: int = 64,
        replay_buffer_size: int = 1_000_000,
    ) -> None:
        """
        Train SAC agent using a simple replay buffer.
        """
        replay_buffer = []

        for ep in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0

            for _ in range(100):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                replay_buffer.append((
                    state, action, reward, next_state, done
                ))
                if len(replay_buffer) > replay_buffer_size:
                    replay_buffer.pop(0)

                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    s_b, a_b, r_b, ns_b, d_b = zip(*batch)
                    self.update(s_b, a_b, r_b, ns_b, d_b)

                state = next_state
                total_reward += reward
                if done:
                    break

            print(f"Episode {ep}, Total Reward: {total_reward}")


def main() -> None:
    """
    Entry point for SAC training.
    """
    n_inputs = 6
    n_actions = 8
    episodes = 500
    namespace = "default"

    env = K8SEnv(namespace)
    agent = SoftActorCritic(n_inputs, n_actions)
    agent.train(env, episodes)


if __name__ == "__main__":
    main()
