import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define our policy network
from kubernetes_environment_rl import K8SEnv


class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(QNetwork, self).__init__()
        # Define the network layers
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, state):
        return self.network(state)


# Define the PPO algorithm
class DQN:
    def __init__(self, n_inputs, n_actions, gamma=0.99, lr=0.01, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995):
        self.q_network = QNetwork(n_inputs, n_actions)
        self.target_q_network = QNetwork(n_inputs, n_actions)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def get_action(self, state):
        # Epsilon-greedy action selection
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            state = torch.tensor(state, dtype=torch.float32).view(1, -1)  # Reshape the state
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)

        # Compute the Q-value of the current state-action pair
        current_q = self.q_network(state)[0][action]

        # Compute the maximum Q-value for the next state
        with torch.no_grad():
            max_next_q = self.target_q_network(next_state).max(1)[0]

        # Compute the target Q-value
        target_q = reward + (1 - done) * self.gamma * max_next_q

        # Compute the loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


def train_dqn(env, dqn, episodes, update_target_interval=10):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(100):  # T_MAX is the maximum number of steps in an episode
            action = dqn.get_action(state)
            next_state, reward, done, _ = env.step(action)

            # Update the Q-network
            dqn.update(state, action, reward, next_state, done)

            total_reward += reward

            if done:
                break

            state = next_state

        # Periodically update the target Q-network
        if episode % update_target_interval == 0:
            dqn.update_target_network()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return dqn

# Since we haven't instantiated the environment, the train function isn't executed yet.
# This is just the function definition.


# Run the training
N_INPUTS = 24  # Number of inputs to the policy network. Change this according to your state representation
N_ACTIONS = 8  # Number of possible actions. Change this according to your action space
EPISODES = 500  # Number of episodes to train for
NAMESPACE = 'default'  # Kubernetes namespace

env = K8SEnv(NAMESPACE)
dqn = DQN(N_INPUTS, N_ACTIONS)
train_dqn(env, dqn, EPISODES)
