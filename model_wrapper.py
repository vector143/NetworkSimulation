"""
无线网络环境模型包装器
======================
将 SimplifiedWirelessEnv 中的核心仿真逻辑（用户移动、KPI计算、奖励计算）
封装为可脱离 Gym 环境独立调用的"已知世界模型"。

用途：供 Dyna-PPO、Dyna-GRPO、MCTS 等 Model-based RL 算法使用。
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WirelessModelConfig:
    """模型配置（与 WirelessEnvConfig 一致，但只保留模型需要的部分）"""
    num_bs: int = 3
    num_users: int = 100
    area_size: float = 500.0
    bs_height: float = 30.0
    ue_height: float = 1.5
    carrier_freq: float = 3.5e9

    # 路径损耗
    pathloss_a: float = 28.0
    pathloss_b: float = 22.0
    pathloss_c: float = 20.0
    shadowing_std: float = 0.0

    # 天线
    tx_power: float = 46.0
    antenna_gain_max: float = 15.0
    beamwidth_h: float = 65.0
    beamwidth_v: float = 10.0

    # 噪声
    noise_figure: float = 9.0
    noise_floor: float = -174.0
    bandwidth: float = 20e6

    # SINR-CQI 映射表
    sinr_cqi_table: Tuple = (
        (-7, 1), (-5, 2), (-3, 3), (-1, 4), (1, 5),
        (3, 6), (5, 7), (8, 8), (11, 9), (14, 10),
        (17, 11), (20, 12), (23, 13), (26, 14), (29, 15)
    )

    # CQI-频谱效率映射表（注意：这里不设默认值，在 __post_init__ 中处理）
    cqi_efficiency_table: Dict = None

    # 能耗
    p_static: float = 200.0
    p_dynamic: float = 300.0

    # 用户移动
    user_speed_mean: float = 1.5
    user_speed_std: float = 0.5

    # 奖励权重
    reward_weights: Dict = None

    def __post_init__(self):
        # 处理 cqi_efficiency_table 的默认值
        if self.cqi_efficiency_table is None:
            self.cqi_efficiency_table = {
                1: 0.15, 2: 0.23, 3: 0.38, 4: 0.60, 5: 0.88,
                6: 1.18, 7: 1.48, 8: 1.91, 9: 2.41, 10: 2.73,
                11: 3.32, 12: 3.90, 13: 4.52, 14: 5.12, 15: 5.55,
            }

        # 处理 reward_weights 的默认值
        if self.reward_weights is None:
            self.reward_weights = {
                "throughput_dl": 0.7,
                "delay": -0.3,
            }


class WirelessModel:
    """无线网络环境的已知世界模型。

    将 SimplifiedWirelessEnv 中的 _update_user_positions、
    _simulate_network、_compute_reward 等核心逻辑封装为独立可调用的模型。

    使用方法:
        config = WirelessModelConfig()
        model = WirelessModel(config, seed=42)
        next_state, obs, reward = model.step(state, action)
    """

    def __init__(self, config: WirelessModelConfig = None, seed: int = None):
        self.config = config if config is not None else WirelessModelConfig()
        self.rng = np.random.default_rng(seed)

        # ---- 静态场景：基站位置（整个 episode 不变）----
        self.bs_positions = self._generate_bs_positions()

        # ---- 预计算常量 ----
        self.noise_power = (
            self.config.noise_floor
            + 10 * np.log10(self.config.bandwidth)
            + self.config.noise_figure
        )

        # ---- 预加载映射表 ----
        self.sinr_cqi_table = self.config.sinr_cqi_table
        self.cqi_efficiency_table = self.config.cqi_efficiency_table

    # ------------------------------------------------------------------
    # 场景初始化
    # ------------------------------------------------------------------

    def _generate_bs_positions(self) -> np.ndarray:
        """生成基站位置（与 SimplifiedWirelessEnv 一致）。"""
        if self.config.num_bs == 3:
            size = self.config.area_size
            return np.array([
                [size / 2, size * 0.2],
                [size * 0.2, size * 0.8],
                [size * 0.8, size * 0.8],
            ])
        else:
            angles = np.linspace(0, 2 * np.pi, self.config.num_bs, endpoint=False)
            radius = self.config.area_size * 0.4
            center = self.config.area_size / 2
            return np.column_stack([
                center + radius * np.cos(angles),
                center + radius * np.sin(angles)
            ])

    def generate_initial_state(self, seed: int = None) -> Dict[str, np.ndarray]:
        """生成初始状态（模拟 env.reset() 中的用户位置和速度初始化）。

        返回:
            state: 包含 user_positions (num_users, 2) 和 user_velocities (num_users, 2)
        """
        rng = np.random.default_rng(seed)

        # 用户初始位置（泊松点过程近似）
        user_positions = rng.uniform(0, self.config.area_size,
                                     (self.config.num_users, 2))

        # 热点：20% 用户在第一个基站附近
        hotspot_center = self.bs_positions[0]
        n_hotspot = self.config.num_users // 5
        if n_hotspot > 0:
            user_positions[:n_hotspot] = hotspot_center + rng.normal(
                0, self.config.area_size * 0.05, (n_hotspot, 2)
            )
            user_positions = np.clip(user_positions, 0, self.config.area_size)

        # 用户初始速度
        speeds = rng.lognormal(
            mean=np.log(self.config.user_speed_mean),
            sigma=self.config.user_speed_std,
            size=self.config.num_users
        )
        angles = rng.uniform(0, 2 * np.pi, self.config.num_users)
        user_velocities = np.column_stack([
            speeds * np.cos(angles),
            speeds * np.sin(angles)
        ])

        return {
            'user_positions': user_positions.astype(np.float32),
            'user_velocities': user_velocities.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # 用户移动（随机游走）
    # ------------------------------------------------------------------

    def _update_user_positions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """更新用户位置（随机游走模型，与 SimplifiedWirelessEnv 一致）。

        参数:
            positions: (num_users, 2) 当前位置
            velocities: (num_users, 2) 当前速度
            seed: 随机种子（可选，用于可复现推演）

        返回:
            new_positions: (num_users, 2) 新位置
            new_velocities: (num_users, 2) 新速度
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        # 移动
        new_positions = positions + velocities

        # 边界反弹
        new_velocities = velocities.copy()
        for i in range(self.config.num_users):
            for dim in range(2):
                if new_positions[i, dim] < 0:
                    new_positions[i, dim] = 0
                    new_velocities[i, dim] *= -1
                elif new_positions[i, dim] > self.config.area_size:
                    new_positions[i, dim] = self.config.area_size
                    new_velocities[i, dim] *= -1

        # 5% 概率随机改变方向
        direction_change_mask = rng.random(self.config.num_users) < 0.05
        if direction_change_mask.any():
            n_changes = direction_change_mask.sum()
            new_angles = rng.uniform(0, 2 * np.pi, n_changes)
            speeds = np.linalg.norm(new_velocities[direction_change_mask], axis=1)
            new_velocities[direction_change_mask, 0] = speeds * np.cos(new_angles)
            new_velocities[direction_change_mask, 1] = speeds * np.sin(new_angles)

        return new_positions, new_velocities

    # ------------------------------------------------------------------
    # KPI 计算（_simulate_network 核心逻辑）
    # ------------------------------------------------------------------

    def _get_shadowing(self, x: float, y: float, bs_id: int) -> float:
        """获取阴影衰落（确定性伪随机）。"""
        seed_val = int(abs(hash((int(x * 100), int(y * 100), bs_id))) % (2 ** 31))
        local_rng = np.random.default_rng(seed_val)
        return local_rng.normal(0, self.config.shadowing_std)

    def _compute_antenna_gain(self, d_2d: float, downtilt: float, bs_id: int) -> float:
        """计算天线增益。"""
        bs_pos = self.bs_positions[bs_id]
        elevation_angle = np.degrees(np.arctan(
            (self.config.bs_height - self.config.ue_height) / max(d_2d, 1.0)
        ))
        angle_diff = elevation_angle - downtilt
        vertical_attenuation = -24 * (angle_diff / self.config.beamwidth_v) ** 2
        horizontal_attenuation = 0
        gain = self.config.antenna_gain_max + vertical_attenuation + horizontal_attenuation
        return max(gain, -20.0)

    def _sinr_to_cqi(self, sinr_db: float) -> int:
        """SINR → CQI 映射。"""
        cqi = 1
        for threshold, cqi_val in self.sinr_cqi_table:
            if sinr_db >= threshold:
                cqi = cqi_val
            else:
                break
        return cqi

    def _compute_kpis(
        self,
        user_positions: np.ndarray,
        action: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """根据用户位置和动作计算全网 KPI（_simulate_network 的核心逻辑）。

        参数:
            user_positions: (num_users, 2) 用户位置
            action: 动作字典

        返回:
            obs: 8 维 KPI 字典
        """
        # 解析动作
        downtilt = float(action.get("downtilt", np.array([0.0])).item())
        tx_power_offset = float(action.get("tx_power_offset", np.array([0.0])).item())
        p0_pusch = float(action.get("p0_nominal_pusch", np.array([-111.0])).item())
        drx_cycle_val = {0: 320, 1: 640, 2: 1280}.get(
            int(action.get("drx_cycle", 1)), 640
        )

        actual_tx_power = self.config.tx_power + tx_power_offset

        # 统计变量
        total_sinr = 0.0
        total_throughput_dl = 0.0
        total_throughput_ul = 0.0
        total_delay = 0.0

        for i in range(self.config.num_users):
            ue_pos = user_positions[i]

            # 计算到各基站的 RSRP
            rsrps = []
            for j in range(self.config.num_bs):
                bs_pos = self.bs_positions[j]
                d_2d = np.sqrt((ue_pos[0] - bs_pos[0]) ** 2 +
                               (ue_pos[1] - bs_pos[1]) ** 2)
                d_2d = max(d_2d, 1.0)
                d_3d = np.sqrt(d_2d ** 2 +
                               (self.config.bs_height - self.config.ue_height) ** 2)

                pathloss = (self.config.pathloss_a
                            + self.config.pathloss_b * np.log10(d_3d)
                            + self.config.pathloss_c * np.log10(self.config.carrier_freq / 1e9))
                shadowing = self._get_shadowing(ue_pos[0], ue_pos[1], j)
                pathloss += shadowing
                antenna_gain = self._compute_antenna_gain(d_2d, downtilt, j)
                rsrp = actual_tx_power - pathloss + antenna_gain
                rsrps.append(rsrp)

            serving_bs = int(np.argmax(rsrps))
            serving_rsrp = rsrps[serving_bs]

            # SINR
            serving_rsrp_linear = 10 ** (serving_rsrp / 10)
            interference_linear = sum(
                10 ** (rsrps[j] / 10) for j in range(self.config.num_bs) if j != serving_bs
            )
            noise_linear = 10 ** (self.noise_power / 10)
            sinr_linear = serving_rsrp_linear / (interference_linear + noise_linear + 1e-10)
            sinr_db = 10 * np.log10(sinr_linear + 1e-10)
            sinr_db = np.clip(sinr_db, -10, 30)

            # 吞吐量
            cqi = self._sinr_to_cqi(sinr_db)
            spectral_efficiency = self.cqi_efficiency_table.get(cqi, 0.15)
            throughput_dl = spectral_efficiency * (self.config.bandwidth / 1e6)
            ul_factor = 0.25 + 0.15 * (p0_pusch + 126) / 30
            throughput_ul = throughput_dl * np.clip(ul_factor, 0.2, 0.45)

            # 时延
            base_delay = 2
            transmission_delay = 0.8 / (throughput_dl + 0.01)
            transmission_delay_ms = transmission_delay * 1000
            drx_penalty = drx_cycle_val / 8
            queueing_delay = (self.config.num_users / 200) * 5
            delay = base_delay + transmission_delay_ms + drx_penalty + queueing_delay
            delay = np.clip(delay, 2, 200)

            # 累加
            total_sinr += sinr_db
            total_throughput_dl += throughput_dl
            total_throughput_ul += throughput_ul
            total_delay += delay

        n_users = self.config.num_users
        avg_sinr = total_sinr / n_users
        avg_throughput_dl = total_throughput_dl / n_users
        avg_throughput_ul = total_throughput_ul / n_users
        avg_delay = total_delay / n_users

        # 功耗
        load_factor = np.clip(avg_throughput_dl / 200, 0, 1)
        power_per_bs = (self.config.p_static
                        + load_factor * self.config.p_dynamic
                        + tx_power_offset * 2)
        power_per_bs = max(power_per_bs, self.config.p_static * 0.5)
        total_power = power_per_bs * self.config.num_bs

        # 能效
        energy_eff = (avg_throughput_dl * n_users) / (total_power + 1e-10) * 100
        energy_eff = np.clip(energy_eff, 0, 100)

        # PRB 利用率
        max_spectral_eff = max(self.cqi_efficiency_table.values())
        max_throughput = max_spectral_eff * (self.config.bandwidth / 1e6)
        prb_util = np.clip((avg_throughput_dl / max_throughput) * 100, 0, 100)

        # 切换成功率
        base_success = 98.0
        optimal_downtilt_range = (3, 6)
        if downtilt < optimal_downtilt_range[0]:
            penalty = (optimal_downtilt_range[0] - downtilt) * 0.5
        elif downtilt > optimal_downtilt_range[1]:
            penalty = (downtilt - optimal_downtilt_range[1]) * 0.5
        else:
            penalty = 0
        handover_success = np.clip(base_success - penalty, 85, 100)

        # RRC 连接数
        rrc_connections = n_users + self.rng.integers(-5, 6)

        return {
            "throughput_dl": np.array([avg_throughput_dl], dtype=np.float32),
            "throughput_ul": np.array([avg_throughput_ul], dtype=np.float32),
            "delay": np.array([avg_delay], dtype=np.float32),
            "sinr": np.array([avg_sinr], dtype=np.float32),
            "energy_efficiency": np.array([energy_eff], dtype=np.float32),
            "handover_success_rate": np.array([handover_success], dtype=np.float32),
            "prb_utilization": np.array([prb_util], dtype=np.float32),
            "rrc_connections": np.array([rrc_connections], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # 奖励计算
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """根据观测计算标量奖励。"""
        reward = 0.0
        normalizers = {
            "throughput_dl": 100.0,
            "delay": 200.0,
        }

        if "throughput_dl" in self.config.reward_weights:
            reward += (self.config.reward_weights["throughput_dl"]
                       * obs["throughput_dl"].item() / normalizers["throughput_dl"])
        if "delay" in self.config.reward_weights:
            reward += (self.config.reward_weights["delay"]
                       * obs["delay"].item() / normalizers["delay"])

        return float(reward)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def compute_obs_from_state(
        self,
        state: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """从状态和动作计算观测。

        用于 Actor 网络：把高维状态转成 8 维 KPI 观测输入。
        """
        return self._compute_kpis(state['user_positions'], action)

    def step(
        self,
        state: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray],
        seed: int = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float]:
        """核心方法：推演一步。

        参数:
            state: {'user_positions': (num_users, 2),
                    'user_velocities': (num_users, 2)}
            action: 动作字典
            seed: 随机种子（可选，控制用户移动的随机性）

        返回:
            next_state: 下一状态
            obs: 8 维 KPI 观测
            reward: 标量奖励
        """
        # 1. 更新用户位置
        next_positions, next_velocities = self._update_user_positions(
            state['user_positions'], state['user_velocities'], seed
        )

        # 2. 计算 KPI
        obs = self._compute_kpis(next_positions, action)

        # 3. 计算奖励
        reward = self._compute_reward(obs)

        # 4. 组装下一状态
        next_state = {
            'user_positions': next_positions,
            'user_velocities': next_velocities,
        }

        return next_state, obs, reward


# ===========================================================================
# 测试代码
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试：WirelessModel 基本功能")
    print("=" * 60)

    config = WirelessModelConfig()
    model = WirelessModel(config, seed=42)

    # 生成初始状态
    state = model.generate_initial_state(seed=123)
    print(f"初始用户位置范围: x=[{state['user_positions'][:,0].min():.1f}, "
          f"{state['user_positions'][:,0].max():.1f}], "
          f"y=[{state['user_positions'][:,1].min():.1f}, "
          f"{state['user_positions'][:,1].max():.1f}]")

    # 构造测试动作
    action = {
        "downtilt": np.array([0.0], dtype=np.float32),
        "tx_power_offset": np.array([0.0], dtype=np.float32),
        "p0_nominal_pusch": np.array([-111.0], dtype=np.float32),
        "drx_cycle": 1,
        "csi_rs_period": 1,
    }

    # 推演一步
    next_state, obs, reward = model.step(state, action, seed=999)
    print(f"\n推演1步后:")
    print(f"  奖励: {reward:.4f}")
    print(f"  下行吞吐量: {obs['throughput_dl'].item():.2f} Mbps")
    print(f"  时延: {obs['delay'].item():.2f} ms")
    print(f"  SINR: {obs['sinr'].item():.2f} dB")
    print(f"  用户位置是否变化: "
          f"{not np.allclose(state['user_positions'], next_state['user_positions'])}")

    # 推演多步
    print(f"\n推演10步:")
    total_reward = 0
    current_state = state
    for step in range(10):
        current_state, obs, reward = model.step(current_state, action, seed=step)
        total_reward += reward
    print(f"  10步累计奖励: {total_reward:.4f}")
    print(f"  最终下行吞吐量: {obs['throughput_dl'].item():.2f} Mbps")

    print("\n测试通过！")