import re
import numpy as np

with open(r"C:\Users\74138\Desktop\training_log_per.txt",encoding='utf-8') as f:
    lines = f.readlines()

rewards = []
count = 0
for line in lines:
    match = re.search(r"Avg Reward \(10 ep\):\s+(-?[\d.]+)", line)
    if match:
        rewards.append(float(match.group(1)))
        print(f"第{count}轮: {rewards[-1]:.3f}")
    count += 1

# 按每10000步分段统计
chunk_size = 50  # 约10000步
n = len(rewards)
print(f"总记录数: {n}，总步数约: {n * 200}")
print()

for i in range(0, n, chunk_size):
    chunk = rewards[i:i+chunk_size]
    start_step = i * 200
    end_step = min((i + chunk_size) * 200, n * 200)
    print(f"Step {start_step:5d}-{end_step:5d}: "
          f"mean={np.mean(chunk):.1f}, "
          f"max={np.max(chunk):.1f}, "
          f"min={np.min(chunk):.1f}, "
          f"std={np.std(chunk):.1f}")

# 全局统计
print(f"\n=== 全局 ===")
print(f"前半段均值: {np.mean(rewards[:n//2]):.1f}")
print(f"后半段均值: {np.mean(rewards[n//2:]):.1f}")
print(f"整体均值: {np.mean(rewards):.1f}")
print(f"整体峰值: {np.max(rewards):.1f}")
print(f"最后10个均值: {np.mean(rewards[-10:]):.1f}")