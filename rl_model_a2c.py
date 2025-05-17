"""
Script to train an Actor-Critic (PPO-style) agent on a Kubernetes environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from kubernetes_environment_rl import K8SEnv


class ActorNetwork(nn.Module):
    """
    Policy (actor) network mapping states to action probabilities.
    """

    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)


class CriticNetwork(nn.Module):
    """
    Value (critic) network mapping states to state-value estimates.
    """

    def __init__(self, n_inputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.network(state)


class ActorCritic:
    """
    Actor-Critic algorithm combining policy gradient and value-based updates.
    """

    def __init__(
        self,
        n_inputs,
        n_actions,
        gamma=0.99,
        actor_lr=1e-2,
        critic_lr=1e-2,
    ):
        self.actor = ActorNetwork(n_inputs, n_actions)
        self.critic = CriticNetwork(n_inputs)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        self.gamma = gamma

    def get_action(self, state):
        """
        Sample an action from the policy given a state.

        Returns:
            action (int): chosen action index
            log_prob (Tensor): log probability of the action
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Update actor and critic networks based on a single transition.
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward_t = torch.tensor(reward, dtype=torch.float32)
        action_t = torch.tensor(action, dtype=torch.int64)

        # Critic estimates
        current_value = self.critic(state_t)
        with torch.no_grad():
            next_value = self.critic(next_t)

        target = reward_t + (1 - done) * self.gamma * next_value
        advantage = target - current_value

        # Critic loss (MSE)
        critic_loss = advantage.pow(2).mean()

        # Actor loss (policy gradient)
        probs = self.actor(state_t)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action_t)
        actor_loss = -(log_prob * advantage.detach()).mean()

        # Backpropagate losses
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train(self, env, episodes, max_steps=100):
        """
        Run training loop over specified episodes.
        """
        for ep in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                action, _ = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    break

            print(f"Episode {ep}, Total Reward: {total_reward}")

        return self


def main():
    """
    Entry point for training the Actor-Critic agent.
    """
    # Configuration
    n_inputs = 24
    n_actions = 8
    episodes = 500
    namespace = "default"

    # Initialize environment and agent
    env = K8SEnv(namespace)
    agent = ActorCritic(n_inputs, n_actions)

    # Start training
    agent.train(env, episodes)


if __name__ == "__main__":
    main()
