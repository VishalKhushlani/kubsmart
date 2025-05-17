import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define our policy network
from kubernetes_environment_rl import K8SEnv


class ActorNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

# Define the Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, n_inputs):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)


# Define the PPO algorithm
class ActorCritic:
    def __init__(self, n_inputs, n_actions, gamma=0.99, actor_lr=0.01, critic_lr=0.01):
        self.actor = ActorNetwork(n_inputs, n_actions)
        self.critic = CriticNetwork(n_inputs)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.n_actions = n_actions

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)  # Reshape the state
        with torch.no_grad():
            action_probs = self.actor(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)

        # Get predicted value for current state and next state
        current_value = self.critic(state)
        with torch.no_grad():
            next_value = self.critic(next_state)

        # Compute the target value
        target_value = reward + (1 - done) * self.gamma * next_value

        # Compute advantage
        advantage = target_value - current_value

        # Critic loss (Mean Squared Error)
        critic_loss = advantage.pow(2).mean()

        # Compute actor loss
        action_probs = self.actor(state)
        action_distribution = Categorical(action_probs)
        log_prob = action_distribution.log_prob(action)
        actor_loss = -(log_prob * advantage.detach()).mean()

        # Update the actor and critic networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            for t in range(100):  # T_MAX is the maximum number of steps in an episode
                action, _ = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                # Update the networks
                self.update(state, action, reward, next_state, done)

                total_reward += reward

                if done:
                    break

                state = next_state

            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        return self

# Run the training
N_INPUTS = 24  # Number of inputs to the policy network. Change this according to your state representation
N_ACTIONS = 8  # Number of possible actions. Change this according to your action space
EPISODES = 500  # Number of episodes to train for
NAMESPACE = 'default'  # Kubernetes namespace

env = K8SEnv(NAMESPACE)
a2c = ActorCritic(N_INPUTS, N_ACTIONS)
a2c.train(env, EPISODES)
