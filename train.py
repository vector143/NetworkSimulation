"""
PPO 训练脚本
============
接 wireless_env + ppo_agent，跑完整的训练循环并保存结果
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import time

from wireless_env import SimplifiedWirelessEnv, WirelessEnvConfig, flatten_obs
from ppo_agent import PPOAgent, PPOConfig, RolloutBuffer

# ============================================================
# 配置
# ============================================================

# 环境
ENV_CONFIG = WirelessEnvConfig(max_steps=200)

# PPO
PPO_CFG = PPOConfig(
    rollout_steps=2048,
    batch_size=64,
    k_epochs=10,
    lr_actor=3e-4,
    lr_critic=1e-3,
)

# 训练
TOTAL_STEPS = 100_000        # 总训练步数
SAVE_INTERVAL = 10_000       # 每多少步保存一次
LOG_INTERVAL = 1_000         # 每多少步打印一次
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 初始化
# ============================================================

env = SimplifiedWirelessEnv(ENV_CONFIG)
agent = PPOAgent(state_dim=8, config=PPO_CFG, device="cpu")
buffer = RolloutBuffer()

# 日志
episode_rewards = []       # 每个 episode 的总 reward
recent_rewards = deque(maxlen=10)  # 最近 10 个 episode 的平均
step_reward_log = []       # 所有步的 reward（画曲线用）

state, _ = env.reset()
state_vec = flatten_obs(state)

total_steps = 0
episode_count = 0
episode_reward = 0

print(f"开始训练，总步数: {TOTAL_STEPS}")
start_time = time.time()

# ============================================================
# 训练主循环
# ============================================================

while total_steps < TOTAL_STEPS:
    # ---- 1. 选动作 ----
    action_dict, log_prob, value = agent.get_action(state_vec)

    # ---- 2. 环境执行 ----
    next_state, reward, terminated, truncated, info = env.step(action_dict)
    next_state_vec = flatten_obs(next_state)
    done = terminated or truncated

    # ---- 3. 存入 buffer ----
    buffer.add(state_vec, action_dict, log_prob, reward, done, value)

    step_reward_log.append(reward)
    episode_reward += reward
    total_steps += 1

    # ---- 4. 更新状态 ----
    state_vec = next_state_vec

    # ---- 5. buffer 满了就更新 ----
    if len(buffer) >= PPO_CFG.rollout_steps:
        agent.update(buffer)
        buffer.clear()

    # ---- 6. episode 结束 ----
    if done:
        episode_count += 1
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)

        state, _ = env.reset()
        state_vec = flatten_obs(state)
        episode_reward = 0

    # ---- 7. 打印日志 ----
    if total_steps % LOG_INTERVAL == 0 and episode_count > 0:
        avg_r = np.mean(recent_rewards) if recent_rewards else 0
        elapsed = time.time() - start_time
        print(f"Step {total_steps:6d} | Episode {episode_count:4d} | "
              f"Avg Reward (10 ep): {avg_r:7.2f} | Time: {elapsed:.0f}s")

    # ---- 8. 保存模型 ----
    if total_steps % SAVE_INTERVAL == 0:
        agent.save(f"{LOG_DIR}/ppo_step_{total_steps}.pt")

# ============================================================
# 训练结束：保存 & 画图
# ============================================================

agent.save(f"{LOG_DIR}/ppo_final.pt")

# 画奖励曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards, alpha=0.4, label="Episode Reward")
# 滑动平均
if len(episode_rewards) > 10:
    smoothed = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
    plt.plot(range(9, len(episode_rewards)), smoothed, label="Smoothed (10 ep)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO on Wireless Network Optimization")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
# 步级 reward 滑动平均
if len(step_reward_log) > 100:
    step_smoothed = np.convolve(step_reward_log, np.ones(100)/100, mode='valid')
    plt.plot(step_smoothed)
plt.xlabel("Step (x100)")
plt.ylabel("Avg Reward (100-step window)")
plt.title("Step-level Reward (Smoothed)")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{LOG_DIR}/training_curve.png", dpi=150)
plt.show()

print(f"训练完成！共 {episode_count} 个 Episode, {total_steps} 步")
print(f"模型已保存至 {LOG_DIR}/ppo_final.pt")
print(f"训练曲线已保存至 {LOG_DIR}/training_curve.png")