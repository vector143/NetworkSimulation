"""
Dyna-PPO 训练脚本
=================
基于 train_ppo.py 改造，加入"想象预训 → 真实精调"两阶段设计。
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import time

import torch

from wireless_env import SimplifiedWirelessEnv, WirelessEnvConfig, flatten_obs, ActionNormalizer
from ppo_agent import PPOAgent, PPOConfig, RolloutBuffer
from model_wrapper import WirelessModel, WirelessModelConfig  # === NEW ===

# ============================================================
# 配置
# ============================================================

# 环境
ENV_CONFIG = WirelessEnvConfig(max_steps=200)

# 模型配置（与环境一致）
MODEL_CONFIG = WirelessModelConfig()  # === NEW ===

# PPO
PPO_CFG = PPOConfig(
    rollout_steps=2048,
    batch_size=64,
    k_epochs=3,
    lr_actor=3e-4,
    lr_critic=3e-4,
)

# Dyna 参数  # === NEW ===
NUM_IMAGINED = 5          # 每次想象生成几条轨迹
IMAGINE_LEN = 10          # 每条想象轨迹多少步
PRETRAIN_EPOCHS = 3       # 想象数据预训几轮

# 训练
TOTAL_STEPS = 100_000
SAVE_INTERVAL = 10_000
LOG_INTERVAL = 1_000
LOG_DIR = "logs_dyna"  # === NEW === 单独目录，方便和纯PPO对比
os.makedirs(LOG_DIR, exist_ok=True)


def imagination_phase(agent, model, real_trajectories,
                      num_imagined=5, imagine_len=10, explore_noise_scale=1.5):
    """
    从真实轨迹终点出发，用模型推演多条想象轨迹。
    """
    imagined_buffer = RolloutBuffer()

    default_action = {
        "downtilt": np.array([0.0], dtype=np.float32),
        "tx_power_offset": np.array([0.0], dtype=np.float32),
        "p0_nominal_pusch": np.array([-111.0], dtype=np.float32),
        "drx_cycle": 1,
        "csi_rs_period": 1,
    }

    for _ in range(num_imagined):
        # 1. 获取想象起点
        state = real_trajectories[-1]['final_state']  # 注意键名是 'final_state'

        # 2. 推演 imagine_len 步
        for step in range(imagine_len):
            # a. 状态 → 观测
            obs = model.compute_obs_from_state(state, default_action)
            obs_vector = flatten_obs(obs)

            # b. 用当前策略选动作
            action_dict, log_prob, value = agent.get_action(obs_vector)

            # c. 用模型推演一步
            next_state, next_obs, reward = model.step(state, action_dict)

            # d. 存入缓冲区
            imagined_buffer.add(
                state=obs_vector,
                action_dict=action_dict,
                log_prob=log_prob,
                reward=reward,
                done=False,
                value=value
            )

            # e. 更新状态
            state = next_state

    return imagined_buffer


# ============================================================
# 初始化
# ============================================================

env = ActionNormalizer(SimplifiedWirelessEnv(ENV_CONFIG))
agent = PPOAgent(state_dim=8, config=PPO_CFG, device="cpu")
model = WirelessModel(MODEL_CONFIG, seed=42)  # === NEW ===
buffer = RolloutBuffer()

# 日志
episode_rewards = []
recent_rewards = deque(maxlen=10)
step_reward_log = []

state, _ = env.reset()
state_vec = flatten_obs(state)

total_steps = 0
episode_count = 0
episode_reward = 0

# === NEW === 用于存储当前轨迹的最终状态
current_final_state = None
real_trajectories = []  # 存每个 episode 的 final_state

print(f"开始训练 (Dyna-PPO)，总步数: {TOTAL_STEPS}")
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

    # === NEW === 每步保存 env 的内部状态（用于 episode 结束时记录 final_state）
    current_final_state = {
        'user_positions': env.unwrapped.user_positions.copy(),
        'user_velocities': env.unwrapped.user_velocities.copy(),
    }

    # ---- 5. buffer 满了就更新（Dyna-PPO 版本）----
    if len(buffer) >= PPO_CFG.rollout_steps:
        # === NEW === 阶段2：想象预训
        if len(real_trajectories) > 0:
            imagined_buffer = imagination_phase(
                agent, model, real_trajectories,
                num_imagined=NUM_IMAGINED, imagine_len=IMAGINE_LEN
            )
            for _ in range(PRETRAIN_EPOCHS):
                agent.update(imagined_buffer)

        # === 阶段3：真实精调（原来的 PPO 更新）===
        agent.update(buffer)
        buffer.clear()

    # ---- 6. episode 结束 ----
    if done:
        episode_count += 1
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)

        # === NEW === 保存这个 episode 的 final_state
        if current_final_state is not None:
            real_trajectories.append({
                'final_state': current_final_state,
                'episode_reward': episode_reward,
            })
            # 只保留最近 10 个 episode 的 final_state
            if len(real_trajectories) > 10:
                real_trajectories.pop(0)

        state, _ = env.reset()
        state_vec = flatten_obs(state)
        episode_reward = 0
        current_final_state = None

    # ---- 7. 打印日志 ----
    if total_steps % LOG_INTERVAL == 0 and episode_count > 0:
        avg_r = np.mean(recent_rewards) if recent_rewards else 0
        elapsed = time.time() - start_time
        print(f"Step {total_steps:6d} | Episode {episode_count:4d} | "
              f"Avg Reward (10 ep): {avg_r:7.2f} | Time: {elapsed:.0f}s")

    # ---- 8. 保存模型 ----
    if total_steps % SAVE_INTERVAL == 0:
        agent.save(f"{LOG_DIR}/dyna_ppo_step_{total_steps}.pt")

# ============================================================
# 训练结束：保存 & 画图
# ============================================================

agent.save(f"{LOG_DIR}/dyna_ppo_final.pt")

# 画奖励曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards, alpha=0.4, label="Episode Reward")
if len(episode_rewards) > 10:
    smoothed = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
    plt.plot(range(9, len(episode_rewards)), smoothed, label="Smoothed (10 ep)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Dyna-PPO on Wireless Network Optimization")
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
plt.savefig(f"{LOG_DIR}/training_curve.png", dpi=150)
plt.show()

print(f"训练完成！共 {episode_count} 个 Episode, {total_steps} 步")
print(f"模型已保存至 {LOG_DIR}/dyna_ppo_final.pt")