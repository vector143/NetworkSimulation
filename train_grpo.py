"""
GRPO 训练脚本 (修正版)
=====================
核心修正：选组内优势最大的动作执行，而非固定取第一个
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import time
import torch

from wireless_env import SimplifiedWirelessEnv, WirelessEnvConfig, flatten_obs, ActionNormalizer
from wrappers import EvaluateActionWrapper
from grpo_agent import GRPOAgent, GRPOConfig, RolloutBuffer

# ============================================================
# 配置
# ============================================================
ENV_CONFIG = WirelessEnvConfig(max_steps=200)

GRPO_CFG = GRPOConfig(
    rollout_steps=2048,
    batch_size=64,
    k_epochs=3,
    lr_actor=3e-4,
    group_size=4,           # 每个状态采样4个动作
    entropy_coef=0.01,
)

TOTAL_STEPS = 100_000
SAVE_INTERVAL = 10_000
LOG_INTERVAL = 1_000
LOG_DIR = "logs_grpo_fixed"
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 初始化
# ============================================================
env = EvaluateActionWrapper(ActionNormalizer(SimplifiedWirelessEnv(ENV_CONFIG)))
agent = GRPOAgent(state_dim=8, config=GRPO_CFG, device="cpu")
buffer = RolloutBuffer()

episode_rewards = []
recent_rewards = deque(maxlen=10)
step_reward_log = []

state, _ = env.reset()
state_vec = flatten_obs(state)

total_steps = 0
episode_count = 0
episode_reward = 0

print(f"开始 GRPO 训练，总步数: {TOTAL_STEPS}")
start_time = time.time()

# ============================================================
# 训练主循环 (修正版)
# ============================================================
while total_steps < TOTAL_STEPS:
    # ---- 1. 采样G个动作，计算组内优势 ----
    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)

    actions, log_probs, advantages, rewards, entropy = agent.compute_group_advantages(
        state_tensor, env, state_vec
    )
    # actions: list of G action dicts
    # advantages: tensor shape (G,), 已做组内标准化
    # rewards: tensor shape (G,), 真实奖励

    # ---- 2. 核心修正：选组内优势最大的动作执行 ----
    best_idx = torch.argmax(advantages).item()  # 或 torch.argmax(rewards)
    action_dict = actions[best_idx]
    log_prob = log_probs[best_idx].item()
    advantage = advantages[best_idx].item()
    best_reward = rewards[best_idx].item()  # 可选，用于日志

    # ---- 3. 环境执行选定的动作 ----
    next_state, reward, terminated, truncated, info = env.step(action_dict)
    next_state_vec = flatten_obs(next_state)
    done = terminated or truncated

    # ---- 4. 存入 buffer ----
    buffer.add(state_vec, action_dict, log_prob, advantage, reward)

    step_reward_log.append(reward)
    episode_reward += reward
    total_steps += 1

    # ---- 5. 更新状态 ----
    state_vec = next_state_vec

    # ---- 6. buffer 满了就更新 ----
    if len(buffer) >= GRPO_CFG.rollout_steps:
        agent.update(buffer)
        buffer.clear()

    # ---- 7. episode 结束 ----
    if done:
        episode_count += 1
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)

        state, _ = env.reset()
        state_vec = flatten_obs(state)
        episode_reward = 0

    # ---- 8. 打印日志 ----
    if total_steps % LOG_INTERVAL == 0 and episode_count > 0:
        avg_r = np.mean(recent_rewards) if recent_rewards else 0
        elapsed = time.time() - start_time
        print(f"Step {total_steps:6d} | Episode {episode_count:4d} | "
              f"Avg Reward (10 ep): {avg_r:7.2f} | Time: {elapsed:.0f}s")

    # ---- 9. 保存模型 ----
    if total_steps % SAVE_INTERVAL == 0:
        agent.save(f"{LOG_DIR}/grpo_step_{total_steps}.pt")

# ============================================================
# 训练结束：保存 & 画图
# ============================================================
agent.save(f"{LOG_DIR}/grpo_final.pt")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, alpha=0.4, label="Episode Reward")
if len(episode_rewards) > 10:
    smoothed = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
    plt.plot(range(9, len(episode_rewards)), smoothed, label="Smoothed (10 ep)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("GRPO on Wireless Network Optimization (Fixed)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
if len(step_reward_log) > 100:
    step_smoothed = np.convolve(step_reward_log, np.ones(100)/100, mode='valid')
    plt.plot(step_smoothed)
plt.xlabel("Step (x100)")
plt.ylabel("Avg Reward (100-step window)")
plt.title("Step-level Reward (Smoothed)")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{LOG_DIR}/training_curve_grpo_fixed.png", dpi=150)
plt.show()

print(f"训练完成！共 {episode_count} 个 Episode, {total_steps} 步")
print(f"模型已保存至 {LOG_DIR}/grpo_final.pt")