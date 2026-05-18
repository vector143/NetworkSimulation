
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import time

import torch

from wireless_env import SimplifiedWirelessEnv, WirelessEnvConfig, flatten_obs, ActionNormalizer
from sac_agent import SACAgent, SACConfig, SACBuffer

# ============================================================
# 配置
# ============================================================

# 环境
ENV_CONFIG = WirelessEnvConfig(max_steps=200)



# 训练
TOTAL_STEPS = 100_000        # 总训练步数
SAVE_INTERVAL = 10_000       # 每多少步保存一次
LOG_INTERVAL = 1_000         # 每多少步打印一次
LOG_DIR = "logs_sac"
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 初始化
# ============================================================

env = SimplifiedWirelessEnv(ENV_CONFIG)
agent = SACAgent(state_dim=8,action_dim=1,config=SACConfig(),device="cpu")
total_steps = 0
episode_counts = 0
state,_ = env.reset()
state_env = flatten_obs(state)
episode_reward = 0

# 日志
episode_rewards = []       # 每个 episode 的总 reward
recent_rewards = deque(maxlen=10)  # 最近 10 个 episode 的平均
step_reward_log = []       # 所有步的 reward（画曲线用）

state, _ = env.reset()
state_vec = flatten_obs(state)


print(f"开始训练，总步数: {TOTAL_STEPS}")
start_time = time.time()

# ============================================================
# 训练主循环
# ============================================================

while total_steps < TOTAL_STEPS:
   if total_steps < agent.config.start_steps:
       action_value = np.random.uniform(-1,1)
   else:
       action_tensor,_ = agent.get_action(state_vec)
       action_value = action_tensor.item()

   action_dict = {
       "downtilt": np.array([[action_value]], dtype=np.float32),
       "drx_cycle": 2,
       "csi_rs_period": 3,
   }

   next_state,reward,terminated,truncated,_ = env.step(action_dict)
   next_state_vec = flatten_obs(next_state)
   done = terminated or truncated

   # 存入buffer
   agent.replay_buffer.add(state_vec, action_value, reward, next_state_vec, done)

   if len(agent.replay_buffer.states)>=agent.config.batch_size:
        agent.update(agent.replay_buffer)

   state_vec = next_state_vec
   episode_reward += reward
   total_steps += 1

   if done:
       episode_counts +=1
       episode_rewards.append(episode_reward)
       state, _ = env.reset()
       state_vec = flatten_obs(state)
       recent_rewards.append(episode_reward)
       episode_reward = 0

   if total_steps % LOG_INTERVAL == 0 and episode_counts > 0:
       avg_r = np.mean(recent_rewards) if recent_rewards else 0
       elapsed = time.time() - start_time
       print(f"Step {total_steps:6d} | Episode {episode_counts:4d} | "
             f"Avg Reward (10 ep): {avg_r:7.2f} | Time: {elapsed:.0f}s")


# ============================================================
# 训练结束：保存 & 画图
# ============================================================

agent.save(f"{LOG_DIR}/sac_final.pt")

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
plt.title("SAC on Wireless Network Optimization")
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
plt.savefig(f"{LOG_DIR}/training_curve_sac.png", dpi=150)
plt.show()

print(f"训练完成！共 {episode_counts} 个 Episode, {total_steps} 步")
print(f"模型已保存至 {LOG_DIR}/sac_final.pt")
print(f"训练曲线已保存至 {LOG_DIR}/training_curve_sac.png")