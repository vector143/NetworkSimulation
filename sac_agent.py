import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, List
from dataclasses import dataclass
import copy

@dataclass
class SACConfig:
    actor_hidden: int = 256
    critic_hidden: int = 256

    # 训练超参
    lr_actor: float = 3e-4  # Actor 学习率
    lr_critic: float = 1e-3  # Critic 学习率（通常比 Actor 大）
    gamma: float = 0.99  # 折扣因子
    pi: float = 0.005 #软更新系数
    alpha: float = 0.2 #温度系数
    buffer_capacity: int = 1000000
    batch_size: int = 256
    start_steps: int = 1000 #随机探索步数

class Actor(nn.Module):
    def __init__(self, state_dim: int, config: SACConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, config.actor_hidden),
            nn.Tanh(),
            nn.Linear(config.actor_hidden, config.actor_hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(config.actor_hidden, 1)
        self.log_std_head = nn.Linear(config.actor_hidden, 1)
    def forward(self, state):
        hidden = self.net(state)
        mean = self.mean_head(hidden)
        mean = torch.tanh(mean)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std,-20,2)
        std = torch.exp(log_std)
        return mean, std
    def get_action(self,state,epsilon):
        mean,std = self.forward(state)
        alpha = mean + std * epsilon
        log_prob = -0.5*(torch.sum(epsilon**2,dim=-1)+2*torch.log(std).sum(dim=-1)+torch.log(torch.tensor(2 * torch.pi, device=epsilon.device)))
        return alpha,log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, config: SACConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, config.critic_hidden),
            nn.Tanh(),
            nn.Linear(config.critic_hidden, config.critic_hidden),
            nn.Tanh(),
            nn.Linear(config.critic_hidden, 1),
        )
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.net(x).squeeze(-1)


class SACBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_updated = []
        self.dones = []

    def add(self, state, action, reward, states_updated, done):
        self.states.append(state)
        self.actions.append(action)  # 保持为 float
        self.rewards.append(reward)  # 保持为 float
        self.states_updated.append(states_updated)  # 保持为 ndarray
        self.dones.append(done)  # 保持为 bool/float
    def get_all(self):
        return{
            "states": torch.FloatTensor(np.array(self.states)),
            "actions": torch.FloatTensor(self.actions).unsqueeze(-1),
            "rewards": torch.FloatTensor(self.rewards),
            "states_updated": torch.tensor(np.array(self.states_updated), dtype=torch.float32),
            "dones": torch.FloatTensor(self.dones),
        }


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, config: SACConfig, device:str = 'cpu'):
        super().__init__()
        self.config = config
        self.actor = Actor(state_dim, config)
        self.Q_net1 = QNetwork(state_dim, action_dim, config)
        self.Q_net2 = QNetwork(state_dim, action_dim, config)
        self.T_net1 = copy.deepcopy(self.Q_net1)
        self.T_net2 = copy.deepcopy(self.Q_net2)
        self.replay_buffer = SACBuffer()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.Q_optimizer1 = optim.Adam(self.Q_net1.parameters(), lr=config.lr_critic)
        self.Q_optimizer2 = optim.Adam(self.Q_net2.parameters(), lr=config.lr_critic)
        self.device = torch.device(device)
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        epsilon = torch.randn(1,1).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor,epsilon)
        return action, log_prob
    def update(self, buffer: SACBuffer):
        data = buffer.get_all()
        states = data["states"].to(self.device)
        rewards = data["rewards"].to(self.device)
        dones = data["dones"].to(self.device)
        actions = data["actions"].to(self.device)
        states_updated = data["states_updated"].to(self.device)

        epsilon_next = torch.randn_like(actions).to(self.device)
        a_next, log_prob_next = self.actor.get_action(states_updated, epsilon_next)

        with torch.no_grad():
            q1_next = self.T_net1(states_updated,a_next)
            q2_next = self.T_net2(states_updated,a_next)
            q_next = torch.min(q1_next, q2_next)
            target = rewards + self.config.gamma * (1-dones)*(q_next-self.config.alpha*log_prob_next)

        q1 = self.Q_net1(states,actions)
        q2 = self.Q_net2(states,actions)
        q1_loss = nn.MSELoss()(q1,target)
        q2_loss = nn.MSELoss()(q2,target)

        self.Q_optimizer1.zero_grad()
        q1_loss.backward()
        self.Q_optimizer1.step()

        self.Q_optimizer2.zero_grad()
        q2_loss.backward()
        self.Q_optimizer2.step()

        epsilon_new = torch.randn_like(actions).to(self.device)
        a_new, log_prob_new = self.actor.get_action(states, epsilon_new)

        q1_new = self.Q_net1(states,a_new)
        q2_new = self.Q_net2(states,a_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.config.alpha*log_prob_new - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.Q_net1.parameters(), self.T_net1.parameters()):
                target_param.data.copy_(self.config.pi * param.data + (1 - self.config.pi) * target_param.data)
            for param, target_param in zip(self.Q_net2.parameters(), self.T_net2.parameters()):
                target_param.data.copy_(self.config.pi * param.data + (1 - self.config.pi) * target_param.data)
    def save(self, path: str):
        """保存模型"""
        torch.save({
            "actor": self.actor.state_dict(),
            "Q_net1": self.Q_net1.state_dict(),
            "Q_net2": self.Q_net2.state_dict(),
            "T_net1": self.T_net1.state_dict(),
            "T_net2": self.T_net2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "Q_optimizer1": self.Q_optimizer1.state_dict(),
            "Q_optimizer2": self.Q_optimizer2.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.Q_net1.load_state_dict(checkpoint["Q_net1"])
        self.Q_net2.load_state_dict(checkpoint["Q_net2"])
        self.T_net1.load_state_dict(checkpoint["T_net1"])
        self.T_net2.load_state_dict(checkpoint["T_net2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.Q_optimizer1.load_state_dict(checkpoint["Q_optimizer1"])
        self.Q_optimizer2.load_state_dict(checkpoint["Q_optimizer2"])