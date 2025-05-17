import random

import torch
import torch.nn as nn
import torch.optim as optim

# Define our policy network
from kubernetes_environment_rl import K8SEnv


# Gaussian Policy (Actor) Network
# Updated Gaussian Policy (Actor) Network
class GaussianPolicy(nn.Module):
    def __init__(self, n_inputs, n_actions, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_actions)  # Output both mean and log_std
        )

    def forward(self, state):
        x = self.network(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal_dist = torch.distributions.Normal(mean, std)
        action = normal_dist.rsample()
        log_prob = normal_dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob


# Q-value (Critic) Network
class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs + n_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        print(state, action)
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SoftActorCritic:
    def __init__(self, n_inputs, n_actions, gamma=0.99, tau=0.005, alpha=0.2,
                 actor_lr=0.001, critic_lr=0.001, policy_update_interval=2):
        self.actor = GaussianPolicy(n_inputs, n_actions)
        self.critic1 = QNetwork(n_inputs, n_actions)
        self.critic2 = QNetwork(n_inputs, n_actions)
        self.critic1_target = QNetwork(n_inputs, n_actions)
        self.critic2_target = QNetwork(n_inputs, n_actions)

        # Initialize the target critic networks with the critic network weights
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_update_interval = policy_update_interval
        self.update_step = 0
        self.n_actions = n_actions

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action, _ = self.actor.sample(state)
        return action[0].argmax().item()

    def update(self, state, action, reward, next_state, done):
        print(state, action, reward, next_state)
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        with torch.no_grad():
            next_action, next_action_log_prob = self.actor.sample(next_state)
            q1_next_target = self.critic1_target(next_state, next_action)
            q2_next_target = self.critic2_target(next_state, next_action)
            q_next_target = torch.min(q1_next_target, q2_next_target)
            value_target = reward + (1 - done) * self.gamma * (q_next_target - self.alpha * next_action_log_prob)

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        q1_loss = nn.functional.mse_loss(q1, value_target)
        q2_loss = nn.functional.mse_loss(q2, value_target)

        # Update the critic networks
        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.update_step % self.policy_update_interval == 0:
            actions_pred, log_probs_pred = self.actor.sample(state)
            q1_pred = self.critic1(state, actions_pred)
            q2_pred = self.critic2(state, actions_pred)
            q_pred = torch.min(q1_pred, q2_pred)
            actor_loss = (self.alpha * log_probs_pred - q_pred).mean()

            # Update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Softly update the target critic networks
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.update_step += 1

    def train(self, env, episodes, batch_size=64, replay_buffer_size=1000000):
        # Implementing a simple replay buffer for off-policy training
        replay_buffer = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            for t in range(100):  # T_MAX
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                # Store transition in replay buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > replay_buffer_size:
                    replay_buffer.pop(0)

                if len(replay_buffer) > batch_size:
                    minibatch = random.sample(replay_buffer, batch_size)
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*minibatch)
                    self.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)

                episode_reward += reward
                state = next_state

                if done:
                    break

            print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

        return self


# Run the training
N_INPUTS = 6  # Number of inputs to the policy network. Change this according to your state representation
N_ACTIONS = 8  # Number of possible actions. Change this according to your action space
EPISODES = 500  # Number of episodes to train for
NAMESPACE = 'default'  # Kubernetes namespace

env = K8SEnv(NAMESPACE)
sa2c = SoftActorCritic(N_INPUTS, N_ACTIONS)
sa2c.train(env, EPISODES)
