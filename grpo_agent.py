"""
手写 GRPO (Group Relative Policy Optimization) Agent
=====================================================
实现 Actor 网络 + 组内标准化优势估计 + PPO-Clip 目标函数

GRPO 核心公式：
- 优势函数: A_i = (r_i - mean(r_1,...,r_G)) / (std(r_1,...,r_G) + ε)
- PPO-Clip 目标:
  L = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
  其中 ratio = π_new(a|s) / π_old(a|s)

设计原则：
- 不依赖 Critic 网络，消除自举偏差
- 通过组内多动作采样，用环境真实奖励做横向对比
- 保留 PPO 的 Clip 机制和熵正则化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, List
from dataclasses import dataclass


# ===========================================================================
# 配置类
# ===========================================================================

@dataclass
class GRPOConfig:
    """GRPO 超参数配置"""
    # 网络结构
    actor_hidden: int = 256

    # 训练超参
    lr_actor: float = 3e-4
    clip_epsilon: float = 0.2
    k_epochs: int = 3
    entropy_coef: float = 0.01

    # GRPO 特有
    group_size: int = 4       # 每个状态采样 G 个动作
    epsilon: float = 1e-8     # 组内标准化防除零

    # 数据收集
    rollout_steps: int = 2048
    batch_size: int = 64
    max_grad_norm: float = 0.5


# ===========================================================================
# 神经网络：Actor（与 PPO 完全相同）
# ===========================================================================

class Actor(nn.Module):
    """策略网络（Actor）：输出连续动作的均值和标准差，以及离散动作的 logits"""

    def __init__(self, state_dim: int, config: GRPOConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, config.actor_hidden),
            nn.Tanh(),
            nn.Linear(config.actor_hidden, config.actor_hidden),
            nn.Tanh(),
        )
        # 连续动作：1维（downtilt）
        self.mean_head = nn.Linear(config.actor_hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

        # 离散动作：drx_cycle 3档
        self.discrete_heads = nn.ModuleDict({
            "drx_cycle": nn.Linear(config.actor_hidden, 3),
        })

    def forward(self, state):
        hidden = self.net(state)
        mean = torch.tanh(self.mean_head(hidden))
        std = torch.exp(self.log_std.clamp(-20, 2))

        discrete_logits = {}
        for name, head in self.discrete_heads.items():
            discrete_logits[name] = head(hidden)

        return mean, std, discrete_logits

    def get_action(self, state, deterministic=False):
        """采样动作并返回 log_prob"""
        mean, std, discrete_logits = self.forward(state)

        # 连续动作
        if deterministic:
            cont_action = mean
        else:
            dist = Normal(mean, std)
            cont_action = dist.rsample()
        dist = Normal(mean, std)
        cont_log_prob = dist.log_prob(cont_action).sum(dim=-1)

        # 离散动作
        disc_actions = {}
        disc_log_prob = 0.0
        for name, logits in discrete_logits.items():
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            disc_actions[name] = action
            disc_log_prob += dist.log_prob(action)

        batch_size = cont_action.shape[0]
        default_csi = torch.full((batch_size,), 3, dtype=torch.long, device=cont_action.device)

        return {
            "continuous": cont_action,
            "drx_cycle": disc_actions["drx_cycle"],
            "csi_rs_period": default_csi,
        }, cont_log_prob + disc_log_prob

    def get_g_actions(self, state, G: int):
        """对同一个状态采样 G 个动作，返回动作列表、log_probs、熵"""
        mean, std, discrete_logits = self.forward(state)

        # 扩展batch维度到G
        mean_expanded = mean.repeat(G, 1)
        std_expanded = std.repeat(G, 1)

        # 连续动作：采样G次
        dist_cont = Normal(mean_expanded, std_expanded)
        cont_actions = dist_cont.rsample()
        cont_log_probs = dist_cont.log_prob(cont_actions).sum(dim=-1)
        cont_entropy = dist_cont.entropy().sum(dim=-1).mean()

        # 离散动作：采样G次
        disc_actions_list = {}
        disc_log_probs = torch.zeros(G, device=state.device)
        disc_entropy = 0.0
        for name, logits in discrete_logits.items():
            logits_expanded = logits.repeat(G, 1)
            dist_disc = Categorical(logits=logits_expanded)
            disc_actions_list[name] = dist_disc.sample()
            disc_log_probs += dist_disc.log_prob(disc_actions_list[name])
            disc_entropy += dist_disc.entropy().mean()

        # 组装G个动作字典
        actions = []
        for i in range(G):
            actions.append({
                "continuous": cont_actions[i:i+1],
                "drx_cycle": disc_actions_list["drx_cycle"][i:i+1],
                "csi_rs_period": torch.full((1,), 3, dtype=torch.long, device=state.device),
            })

        return actions, cont_log_probs + disc_log_probs, cont_entropy + disc_entropy

    def evaluate_action(self, state, action_dict):
        """给定状态和旧动作，重新计算 log_prob 和 entropy"""
        mean, std, discrete_logits = self.forward(state)

        dist = Normal(mean, std)
        cont_log_prob = dist.log_prob(action_dict["continuous"]).sum(dim=-1)
        cont_entropy = dist.entropy().sum(dim=-1)

        disc_log_prob = 0.0
        disc_entropy = 0.0
        for name, logits in discrete_logits.items():
            dist = Categorical(logits=logits)
            disc_log_prob += dist.log_prob(action_dict[name])
            disc_entropy += dist.entropy()

        return cont_log_prob + disc_log_prob, cont_entropy + disc_entropy


# ===========================================================================
# 经验缓冲区（只存状态和动作，不存value）
# ===========================================================================

class RolloutBuffer:
    """GRPO 的 on-policy 缓冲区"""

    def __init__(self):
        self.states = []
        self.actions_cont = []
        self.actions_drx = []
        self.log_probs = []
        self.advantages = []
        self.returns = []

    def add(self, state, action_dict, log_prob, advantage, return_):
        self.states.append(state)
        self.actions_cont.append(action_dict["continuous"])
        self.actions_drx.append(action_dict["drx_cycle"])
        self.log_probs.append(torch.tensor(log_prob))
        self.advantages.append(torch.tensor(advantage))
        self.returns.append(torch.tensor(return_))

    def get_all(self):
        return {
            "states": torch.FloatTensor(np.array(self.states)),
            "actions_cont": torch.stack(self.actions_cont),
            "actions_drx": torch.stack(self.actions_drx),
            "log_probs": torch.stack(self.log_probs),
            "advantages": torch.stack(self.advantages),
            "returns": torch.FloatTensor(self.returns),
        }

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ===========================================================================
# GRPO Agent 主类
# ===========================================================================

class GRPOAgent:
    """手写 GRPO Agent

    使用示例：
        agent = GRPOAgent(state_dim=8, config=GRPOConfig())
        action_dict, log_prob = agent.get_action(state_tensor)
        agent.update(buffer)
    """

    def __init__(self, state_dim: int, config: GRPOConfig = None, device: str = "cpu"):
        self.config = config if config is not None else GRPOConfig()
        self.device = torch.device(device)

        self.actor = Actor(state_dim, self.config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """输入状态 numpy 数组，返回动作字典 + log_prob"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_dict, log_prob = self.actor.get_action(state_tensor, deterministic)
        return action_dict, log_prob.item()

    def compute_group_advantages(self, state_tensor, env, state_vec):
        G = self.config.group_size
        actions, log_probs, entropy = self.actor.get_g_actions(state_tensor, G)

        rewards = []
        for action in actions:
            r = env.evaluate_action(state_vec, action)
            rewards.append(r)
        rewards = torch.FloatTensor(rewards).to(self.device)

        mean_r = rewards.mean()
        std_r = rewards.std() + self.config.epsilon
        advantages = (rewards - mean_r) / std_r

        return actions, log_probs, advantages, rewards, entropy

    def update(self, buffer: RolloutBuffer):
        """GRPO 核心更新逻辑"""
        data = buffer.get_all()
        states = data["states"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)
        advantages = data["advantages"].to(self.device)
        returns = data["returns"].to(self.device)

        old_actions = {
            "continuous": data["actions_cont"].to(self.device),
            "drx_cycle": data["actions_drx"].to(self.device),
        }

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = len(states)
        batch_size = min(self.config.batch_size, total_steps)

        for epoch in range(self.config.k_epochs):
            indices = torch.randperm(total_steps)
            for start in range(0, total_steps, batch_size):
                batch_idx = indices[start:start + batch_size]

                batch_states = states[batch_idx]
                batch_actions = {
                    "continuous": old_actions["continuous"][batch_idx],
                    "drx_cycle": old_actions["drx_cycle"][batch_idx],
                }
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]

                new_log_probs, entropy = self.actor.evaluate_action(batch_states, batch_actions)

                # PPO-Clip Actor Loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                    1 + self.config.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= self.config.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])