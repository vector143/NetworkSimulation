"""环境因果链诊断：下倾角 → 吞吐量"""
import numpy as np
from wireless_env import SimplifiedWirelessEnv, WirelessEnvConfig

config = WirelessEnvConfig(shadowing_std=0.0, max_steps=200)
env = SimplifiedWirelessEnv(config)

for downtilt in [-10, -7, -4, -1, 0, 2, 3, 5, 6, 8, 10]:
    obs, _ = env.reset(seed=42)
    total_tput = 0
    for _ in range(50):  # 跑 50 步取平均，消除随机性
        action = {
            "downtilt": np.array([downtilt], dtype=np.float32),
            "tx_power_offset": np.array([0.0], dtype=np.float32),
            "p0_nominal_pusch": np.array([-111.0], dtype=np.float32),
            "drx_cycle": 1,
            "csi_rs_period": 1,
        }
        obs, _, _, _, _ = env.step(action)
        total_tput += obs["throughput_dl"].item()
    print(f"下倾角 {downtilt:4.0f}° → 平均吞吐量 {total_tput/50:.1f} Mbps")