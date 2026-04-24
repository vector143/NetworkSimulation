"""
手写 PPO (Proximal Policy Optimization) Agent
==============================================
实现 Actor-Critic 架构 + GAE 优势估计 + PPO-Clip 目标函数

PPO 核心公式回顾：
- 优势函数 (GAE): A_t = Σ (γλ)^k · δ_{t+k}
  其中 δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
- PPO-Clip 目标:
  L = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
  其中 ratio = π_new(a|s) / π_old(a|s)

设计原则：
- 只依赖 PyTorch，不依赖 Stable-Baselines
- 通过 flatten_obs/flatten_action 与环境对接
- 所有超参数集中在 PPOConfig 里管理
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, Tuple, List
from dataclasses import dataclass
import copy


# ===========================================================================
# 配置类
# ===========================================================================

@dataclass
class PPOConfig:
    """PPO 超参数配置"""
    # 网络结构
    actor_hidden: int = 256  # Actor 隐藏层维度
    critic_hidden: int = 256  # Critic 隐藏层维度

    # 训练超参
    lr_actor: float = 3e-4  # Actor 学习率
    lr_critic: float = 1e-3  # Critic 学习率（通常比 Actor 大）
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE 的 λ 参数
    clip_epsilon: float = 0.2  # PPO Clip 范围 [1-ε, 1+ε]
    k_epochs: int = 10  # 同一批数据更新多少轮
    entropy_coef: float = 0.01  # 熵正则化系数（鼓励探索）

    # 数据收集
    rollout_steps: int = 2048  # 收集多少步做一次更新
    batch_size: int = 64  # mini-batch 大小
    max_grad_norm: float = 0.5  # 梯度裁剪阈值


# ===========================================================================
# 神经网络：Actor + Critic
# ===========================================================================

class Actor(nn.Module):
    """策略网络（Actor）：输出连续动作的均值和标准差，以及离散动作的 logits

    输入：状态向量（8 维 KPI 展平后的向量）
    输出：
      - 连续动作：高斯分布的均值（3维）和 log_std（3维，可学习参数）
      - 离散动作：Categorical 分布的 logits（两个 head，各 3 类）
    """

    def __init__(self, state_dim: int, config: PPOConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, config.actor_hidden),
            nn.Tanh(),
            nn.Linear(config.actor_hidden, config.actor_hidden),
            nn.Tanh(),
        )
        # 连续动作：3 维（downtilt, tx_power_offset, p0_nominal_pusch）,暂时改为1维
        self.mean_head = nn.Linear(config.actor_hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # 可学习的 log 标准差

        # 离散动作：各 3 类（drx_cycle 3 档, csi_rs_period 3 档），暂时设为drx_cycle一个
        self.discrete_heads = nn.ModuleDict({
            "drx_cycle": nn.Linear(config.actor_hidden, 3),
        })

    def forward(self, state):
        """返回动作分布的参数（不采样）"""
        hidden = self.net(state)

        # 连续动作：输出维度由 mean_head 自动决定（现在是1维）
        mean = torch.tanh(self.mean_head(hidden))
        std = torch.exp(self.log_std.clamp(-20, 2))

        # 离散动作
        discrete_logits = {}
        for name, head in self.discrete_heads.items():
            discrete_logits[name] = head(hidden)

        return mean, std, discrete_logits

    # def get_action(self, state, deterministic=False):
    #     """采样动作并返回 log_prob
    #
    #     参数:
    #         state: 状态张量
    #         deterministic: True 时用均值（评估模式），False 时采样（训练模式）
    #     返回:
    #         action_dict: 包含 continuous 和离散动作的字典
    #         log_prob: 整个动作的联合 log 概率（连续+离散）
    #     """
    #     mean, std, discrete_logits = self.forward(state)
    #
    #     # ---- 连续动作 ----
    #     if deterministic:
    #         cont_action = mean  # 直接用均值
    #     else:
    #         dist = Normal(mean, std)
    #         cont_action = dist.rsample()  # rsample 支持重参数化
    #
    #     # 连续动作的 log_prob（对每个维度求和，得到联合 log 概率）
    #     dist = Normal(mean, std)
    #     cont_log_prob = dist.log_prob(cont_action).sum(dim=-1)
    #
    #     # ---- 离散动作 ----
    #     disc_actions = {}
    #     disc_log_prob = 0.0
    #     for name, logits in discrete_logits.items():
    #         dist = Categorical(logits=logits)
    #         if deterministic:
    #             action = torch.argmax(logits, dim=-1)
    #         else:
    #             action = dist.sample()
    #         disc_actions[name] = action
    #         disc_log_prob += dist.log_prob(action)
    #
    #     return {
    #         "continuous": cont_action,
    #         "drx_cycle": disc_actions["drx_cycle"],
    #         "csi_rs_period": disc_actions["csi_rs_period"],
    #     }, cont_log_prob + disc_log_prob

    def get_action(self, state, deterministic=False):
        """采样动作并返回 log_prob

        返回:
            action_dict: 包含 continuous 和离散动作的字典
            log_prob: 联合 log 概率
        """
        mean, std, discrete_logits = self.forward(state)

        # ---- 连续动作（1维：downtilt）----
        if deterministic:
            cont_action = mean
        else:
            dist = Normal(mean, std)
            cont_action = dist.rsample()

        dist = Normal(mean, std)
        cont_log_prob = dist.log_prob(cont_action).sum(dim=-1)

        # ---- 离散动作（只有 drx_cycle）----
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

        # 组装动作字典：被屏蔽的维度填充默认值
        return {
            "continuous": cont_action,  # shape (batch, 1) 只含 downtilt
            "drx_cycle": disc_actions["drx_cycle"],
            "csi_rs_period": torch.zeros_like(disc_actions["drx_cycle"]),  # 默认值
        }, cont_log_prob + disc_log_prob

    # def get_action(self, state, deterministic=False):
    #     """采样动作并返回 log_prob
    #
    #     返回:
    #         action_dict: 包含 continuous 和离散动作的字典
    #         log_prob: 联合 log 概率（这里只有连续部分）
    #     """
    #     mean, std, discrete_logits = self.forward(state)
    #
    #     # ---- 连续动作（1维：downtilt）----
    #     if deterministic:
    #         cont_action = mean
    #     else:
    #         dist = Normal(mean, std)
    #         cont_action = dist.rsample()
    #
    #     dist = Normal(mean, std)
    #     cont_log_prob = dist.log_prob(cont_action).sum(dim=-1)
    #
    #     # ---- 离散动作：全部用默认值 ----
    #     # drx_cycle 默认 2 (1280ms), csi_rs_period 默认 3 (160ms)
    #     batch_size = cont_action.shape[0]
    #     default_drx = torch.full((batch_size,), 2, dtype=torch.long, device=cont_action.device)
    #     default_csi = torch.full((batch_size,), 3, dtype=torch.long, device=cont_action.device)
    #
    #     return {
    #         "continuous": cont_action,
    #         "drx_cycle": default_drx,
    #         "csi_rs_period": default_csi,
    #     }, cont_log_prob  # 只返回连续动作的 log_prob

    def evaluate_action(self, state, action_dict):
        """给定状态和旧动作，重新计算 log_prob（用于 PPO 更新时算 ratio）

        参数:
            state: 状态张量
            action_dict: 包含 "continuous", "drx_cycle", "csi_rs_period" 的字典
        返回:
            log_prob: 在当前策略下，这些动作的 log 概率
            entropy: 当前策略的熵（用于熵正则化）
        """
        mean, std, discrete_logits = self.forward(state)

        # 连续动作
        dist = Normal(mean, std)
        cont_log_prob = dist.log_prob(action_dict["continuous"]).sum(dim=-1)
        cont_entropy = dist.entropy().sum(dim=-1)

        # 离散动作
        disc_log_prob = 0.0
        disc_entropy = 0.0
        for name, logits in discrete_logits.items():
            dist = Categorical(logits=logits)
            disc_log_prob += dist.log_prob(action_dict[name])
            disc_entropy += dist.entropy()

        return cont_log_prob + disc_log_prob, cont_entropy + disc_entropy


class Critic(nn.Module):
    """价值网络（Critic）：估计状态价值 V(s)"""

    def __init__(self, state_dim: int, config: PPOConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, config.critic_hidden),
            nn.Tanh(),
            nn.Linear(config.critic_hidden, config.critic_hidden),
            nn.Tanh(),
            nn.Linear(config.critic_hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)  # 输出标量


# ===========================================================================
# 经验缓冲区
# ===========================================================================

class RolloutBuffer:
    """PPO 的 on-policy 缓冲区：存储一段轨迹，用完即弃"""

    def __init__(self):
        self.states = []
        self.actions_cont = []  # 连续动作
        self.actions_drx = []  # 离散动作
        #self.actions_csi = []  # 离散动作
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action_dict, log_prob, reward, done, value):
        self.states.append(state)
        self.actions_cont.append(action_dict["continuous"])
        self.actions_drx.append(action_dict["drx_cycle"])
        #self.actions_csi.append(action_dict["csi_rs_period"])
        self.log_probs.append(torch.tensor(log_prob))
        self.rewards.append(torch.tensor(reward))
        self.dones.append(torch.tensor(done))
        self.values.append(torch.tensor(value))

    def get_all(self):
        """返回缓冲区中所有数据（作为 batch 用于 PPO 更新）"""
        return {
            "states": torch.FloatTensor(np.array(self.states)),
            "actions_cont": torch.stack(self.actions_cont),
            "actions_drx": torch.stack(self.actions_drx),
            #"actions_csi": torch.stack(self.actions_csi),
            "log_probs": torch.stack(self.log_probs),
            "rewards": torch.FloatTensor(self.rewards),
            "dones": torch.FloatTensor(self.dones),
            "values": torch.stack(self.values),
        }

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ===========================================================================
# PPO Agent 主类
# ===========================================================================

class PPOAgent:
    """手写 PPO Agent

    使用示例：
        agent = PPOAgent(state_dim=8, config=PPOConfig())
        # 训练时
        action_dict, log_prob = agent.get_action(state_tensor)
        # 更新时
        agent.update(buffer)
    """

    def __init__(self, state_dim: int, config: PPOConfig = None, device: str = "cpu"):
        self.config = config if config is not None else PPOConfig()
        self.device = torch.device(device)

        # 网络初始化
        self.actor = Actor(state_dim, self.config).to(self.device)
        self.critic = Critic(state_dim, self.config).to(self.device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """输入状态 numpy 数组，返回动作字典 + log_prob + value

        参数:
            state: 形状 (state_dim,) 的 numpy 数组
            deterministic: 是否确定性推理
        返回:
            action_dict, log_prob, value
            其中 value 是 V(s) 的标量值，用于 GAE 计算
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_dict, log_prob = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)

        return action_dict, log_prob.item(), value.item()

    def update(self, buffer: RolloutBuffer):
        """PPO 核心更新逻辑

        1. 从 buffer 取出所有数据
        2. 计算 GAE 优势函数
        3. K 轮小批量更新（PPO-Clip）
        """
        data = buffer.get_all()
        states = data["states"].to(self.device)
        rewards = data["rewards"].to(self.device)
        dones = data["dones"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)
        old_values = data["values"].to(self.device)

        # 组装动作字典
        old_actions = {
            "continuous": data["actions_cont"].to(self.device),
            "drx_cycle": data["actions_drx"].to(self.device),
            #"csi_rs_period": data["actions_csi"].to(self.device),
        }

        # ---- 步骤 1：计算 GAE 优势函数 ----
        advantages = self._compute_gae(rewards, old_values, dones)
        returns = advantages + old_values  # 目标价值 = A + V
        # 归一化优势（稳定训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = len(states)
        batch_size = min(self.config.batch_size, total_steps)

        # ---- 步骤 2：K 轮更新 ----
        for epoch in range(self.config.k_epochs):
            # 随机打乱索引
            indices = torch.randperm(total_steps)

            for start in range(0, total_steps, batch_size):
                batch_idx = indices[start:start + batch_size]

                batch_states = states[batch_idx]
                batch_actions = {
                    "continuous": old_actions["continuous"][batch_idx],
                    "drx_cycle": old_actions["drx_cycle"][batch_idx],
                    #"csi_rs_period": old_actions["csi_rs_period"][batch_idx],
                }
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # ---- 计算当前策略下的 log_prob 和 entropy ----
                new_log_probs, entropy = self.actor.evaluate_action(batch_states, batch_actions)
                new_values = self.critic(batch_states)

                # ---- PPO-Clip Actor Loss ----
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                    1 + self.config.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # 熵正则化（鼓励探索，防止过早收敛）
                actor_loss -= self.config.entropy_coef * entropy.mean()

                # ---- Critic Loss (MSE) ----
                critic_loss = nn.MSELoss()(new_values, batch_returns)

                # ---- 反向传播 ----
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

    # def _compute_gae(self, rewards, values, dones):
    #     """计算 GAE (Generalized Advantage Estimation) 优势函数
    #
    #     从轨迹末尾向前递推：
    #     δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)
    #     A_t = δ_t + γ·λ·(1-done_t)·A_{t+1}
    #
    #     参数:
    #         rewards: shape (T,)
    #         values: shape (T,)
    #         dones: shape (T,)
    #     返回:
    #         advantages: shape (T,)
    #     """
    #     T = len(rewards)
    #     advantages = torch.zeros(T, device=self.device)
    #     gae = 0.0
    #
    #     for t in reversed(range(T)):
    #         if t == T - 1:
    #             next_value = 0  # 终止状态的价值为 0
    #         else:
    #             next_value = values[t + 1] * (1 - dones[t])
    #
    #         delta = rewards[t] + self.config.gamma * next_value - values[t]
    #         gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
    #         advantages[t] = gae
    #
    #     return advantages

    def _compute_gae(self, rewards, values, dones):
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
                next_done = 1  # 最后一步之后视为终止
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]  # 改这里：用下一步的done

            delta = rewards[t] + self.config.gamma * next_value * (1 - next_done) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        return advantages

    def save(self, path: str):
        """保存模型"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])