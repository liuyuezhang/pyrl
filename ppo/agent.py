import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=64):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(state_dim, n_latent_var)
        self.linear2 = nn.Linear(n_latent_var, n_latent_var)
        self.actor = nn.Linear(n_latent_var, action_dim)
        self.critic = nn.Linear(n_latent_var, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x),


class Memory:
    def __init__(self):
        self.actions_v = []
        self.states_v = []
        self.log_prob_actions_v = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.actions_v[:]
        del self.states_v[:]
        del self.log_prob_actions_v[:]
        del self.rewards[:]
        del self.dones[:]


class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, device):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.memory = Memory()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, state):
        with torch.no_grad():
            state_v = torch.from_numpy(state).float().to(self.device)
            action_probs_v = self.policy.forward(state_v)[0]
            dist = Categorical(action_probs_v)
            action_v = dist.sample()

            self.memory.states_v.append(state_v)
            self.memory.actions_v.append(action_v)
            self.memory.log_prob_actions_v.append(dist.log_prob(action_v))

        return action_v.item()

    def step(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)

    def update(self):
        # Monte Carlo estimate of state rewards:
        Rs = []
        R = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            R = reward + (1-done) * self.gamma * R
            Rs.insert(0, R)

        # Normalizing the rewards:
        Rs_v = torch.Tensor(Rs).to(self.device)
        Rs_v = (Rs_v - Rs_v.mean()) / (Rs_v.std() + 1e-8)

        # convert list to tensor
        old_states_v = torch.stack(self.memory.states_v).to(self.device).detach()
        old_actions_v = torch.stack(self.memory.actions_v).to(self.device).detach()
        old_log_prob_actions_v = torch.stack(self.memory.log_prob_actions_v).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            action_probs_v, values_v = self.policy.forward(old_states_v)
            dist = Categorical(action_probs_v)
            log_prob_actions_v = dist.log_prob(old_actions_v)
            entropy_v = dist.entropy()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios_v = torch.exp(log_prob_actions_v - old_log_prob_actions_v.detach())

            # Finding Surrogate Loss:
            advs_v = Rs_v - values_v.detach()
            surr1_v = ratios_v * advs_v
            surr2_v = torch.clamp(ratios_v, 1 - self.eps_clip, 1 + self.eps_clip) * advs_v
            loss_v = -torch.min(surr1_v, surr2_v) + 0.5 * F.mse_loss(values_v, Rs_v) - 0.01 * entropy_v

            # take gradient step
            self.optimizer.zero_grad()
            loss_v.mean().backward()
            self.optimizer.step()

        self.memory.clear()
