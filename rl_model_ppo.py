import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define our policy network
from kubernetes_environment_rl import K8SEnv


class PolicyNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(PolicyNetwork, self).__init__()
        # Define the network layers
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


# Define the PPO algorithm
class PPO:
    def __init__(self, n_inputs, n_actions, gamma=0.99, lr=0.01, clip_epsilon=0.2):
        self.policy = PolicyNetwork(n_inputs, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)  # Reshape the state
        action_probs = self.policy(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def update(self, states, actions, log_probs, returns):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        action_probs = self.policy(states[:, -1, :])  # Only consider the last state in each sequence
        action_distribution = Categorical(action_probs)
        new_log_probs = action_distribution.log_prob(actions)

        ratio = (new_log_probs - log_probs).exp()

        surrogate_objective = ratio * returns
        clipped_surrogate_objective = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * returns

        loss = -torch.min(surrogate_objective, clipped_surrogate_objective).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_ppo(env, ppo, episodes):
    for episode in range(episodes):
        states, actions, log_probs, rewards = [], [], [], []
        state = env.reset()

        for t in range(100):  # T_MAX is the maximum number of steps in an episode
            action, log_prob = ppo.get_action(state)

            new_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)

            if done:
                break

            state = new_state

        returns = []
        Gt = 0
        pw = 0

        for reward in reversed(rewards):
            Gt += ppo.gamma ** pw * reward
            pw += 1
            returns.insert(0, Gt)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # normalize returns

        ppo.update(states, actions, log_probs, returns)


# Run the training
N_INPUTS = 24  # Number of inputs to the policy network. Change this according to your state representation
N_ACTIONS = 8  # Number of possible actions. Change this according to your action space
EPISODES = 500  # Number of episodes to train for
NAMESPACE = 'default'  # Kubernetes namespace

env = K8SEnv(NAMESPACE)
ppo = PPO(N_INPUTS, N_ACTIONS)
train_ppo(env, ppo, EPISODES)
