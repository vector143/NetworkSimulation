"""
简化版无线网络优化 Gym 环境
============================
基于 3GPP TR 38.901 标准统计模型，模拟 5G 蜂窝网络的 KPI 响应。

通信概念速查（写给非通信背景的 RL 开发者）：
-------------------------------------------
- SINR (Signal-to-Interference-plus-Noise Ratio)：信干噪比，单位 dB。
  你可以理解为"有用信号的强度 / (干扰信号强度 + 背景噪声)"。
  值越高，信号质量越好。通常 -5dB 以下很差，20dB 以上很好。

- 路径损耗 (Path Loss)：信号在空间中传播会自然衰减，距离越远衰减越大。
  这个衰减用 dB 来表示，是 SINR 计算的基础。

- 阴影衰落 (Shadow Fading)：大型障碍物（建筑物、山体）造成的信号缓慢起伏。
  用对数正态分布来模拟，在路径损耗上加一个随机偏移。

- 天线下倾角 (Downtilt)：天线向下倾斜的角度。角度越大，信号覆盖越"近"；
  角度越小，信号打得越"远"但近处可能变弱。这是我们要优化的核心参数。

- PRB (Physical Resource Block)：物理资源块，LTE/5G 调度的最小单位。
  PRB 利用率越高，说明网络越"忙"，但太高会导致用户排队、时延增加。

- CQI (Channel Quality Indicator)：信道质量指示，UE（终端设备）根据 SINR
  测量上报给基站的值，范围 0-15。基站根据 CQI 选择 MCS 编码方式。

- MCS (Modulation and Coding Scheme)：调制编码方案，决定了每个符号能携带
  多少比特。CQI 越高，能用更高阶的 MCS，吞吐量越大。

- 基站 / BS (Base Station)：发射信号的设备，比如 5G 基站（gNB）。
  我们在这个环境里优化的是基站的参数。

- 终端 / UE (User Equipment)：手机、CPE 等接收信号的设备。

- 上行 / 下行：下行是基站→手机，上行是手机→基站。通常下行吞吐量远大于上行。

- 切换 (Handover)：用户从一个基站的覆盖范围移动到另一个基站时，
  需要切换服务基站。切换成功率是衡量用户体验的重要指标。

环境特点：
---------
- 纯 Python 实现，不依赖 ns-3 等外部仿真器
- 基于 3GPP TR 38.901 UMa（城市宏站）路径损耗模型
- 使用数学函数模拟 KPI 响应，计算速度快，适合 RL 大规模采样
- 支持连续动作空间（天线下倾角、功率偏置等）
- 观测采用 Dict 空间，按名称访问，便于后续迁移到 ns3-gym

参考文献：
----------
- 3GPP TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz
- 3GPP TS 38.214: Physical layer procedures for data (CQI/MCS 映射表)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# ===========================================================================
# 配置类：所有可调参数集中管理，不硬编码在环境内部
# ===========================================================================

@dataclass
class WirelessEnvConfig:
    """无线网络仿真环境的配置参数。

    修改这里的数据来改变场景设定（基站数、用户数等），
    不需要改动环境代码。
    """
    # ---- 场景布局 ----
    num_bs: int = 3  # 基站数量
    num_users: int = 100  # 用户(UE)数量
    area_size: float = 500.0  # 仿真区域大小（米），正方形区域
    bs_height: float = 30.0  # 基站天线高度（米）
    ue_height: float = 1.5  # 用户设备高度（米）
    carrier_freq: float = 3.5e9  # 载波频率，5G中频段 3.5GHz

    # ---- 信道模型参数（来自 3GPP TR 38.901 UMa 场景）----
    # 路径损耗公式: PL = A + B*log10(d_3D) + C*log10(f_c)
    pathloss_a: float = 28.0  # 截距项
    pathloss_b: float = 22.0  # 距离系数
    pathloss_c: float = 20.0  # 频率系数
    shadowing_std: float = 0.0  # 阴影衰落标准差（dB），UMa LOS 场景 从4.0改为1.0，再改为0，先舍弃

    # ---- 天线参数 ----
    tx_power: float = 46.0  # 基站发射功率（dBm），46dBm ≈ 40W
    antenna_gain_max: float = 15.0  # 天线最大增益（dBi）
    # 天线水平/垂直波束宽度（3dB 带宽，度）
    beamwidth_h: float = 65.0  # 水平波束宽度
    beamwidth_v: float = 10.0  # 垂直波束宽度

    # ---- 噪声与干扰 ----
    noise_figure: float = 9.0  # 接收机噪声系数（dB）
    noise_floor: float = -174.0  # 热噪声基底（dBm/Hz），-174 dBm/Hz 是物理常数
    bandwidth: float = 20e6  # 系统带宽（Hz），20MHz 是典型 5G 载波带宽

    # ---- KPI 计算参数 ----
    # SINR 到 CQI 的映射（简化表，来自 TS 38.214 的近似）
    sinr_cqi_table: tuple = field(default_factory=lambda: (
        (-7, 1), (-5, 2), (-3, 3), (-1, 4), (1, 5),
        (3, 6), (5, 7), (8, 8), (11, 9), (14, 10),
        (17, 11), (20, 12), (23, 13), (26, 14), (29, 15)
    ))
    # CQI 到频谱效率的映射（bps/Hz，简化版）
    cqi_efficiency_table: dict = field(default_factory=lambda: {
        1: 0.15, 2: 0.23, 3: 0.38, 4: 0.60, 5: 0.88,
        6: 1.18, 7: 1.48, 8: 1.91, 9: 2.41, 10: 2.73,
        11: 3.32, 12: 3.90, 13: 4.52, 14: 5.12, 15: 5.55
    })

    # ---- 能耗模型参数 ----
    # 基站功耗模型: P_total = P_static + load_factor * P_dynamic
    p_static: float = 200.0  # 静态功耗（W），即使空载也要消耗
    p_dynamic: float = 300.0  # 满载时的额外动态功耗（W）

    # ---- 仿真控制 ----
    max_steps: int = 200  # 每个 Episode 的最大步数

    # ---- 用户移动模型 ----
    user_speed_mean: float = 1.5  # 用户平均移动速度（m/s），步行约 1.5m/s
    user_speed_std: float = 0.5  # 速度标准差

    # ---- LLM 引导相关 ----
    # 默认奖励权重（当不使用 LLM 时），范围 0-1
    default_reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "throughput_dl": 0.7,
        "delay": -0.3,
        # "throughput_ul": 0.15,
        # "delay": -0.15,
        # "energy_efficiency": 0.15,
        # "handover_success_rate": 0.20,  # 提高，鼓励优化覆盖
    })


# ===========================================================================
# 统一的观测空间和动作空间定义（全局常量，被所有环境实现共享）
# ===========================================================================

OBSERVATION_SPACE = spaces.Dict({
    # 下行平均吞吐量（Mbps），0-1000，正常范围 10-500
    "throughput_dl": spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32),
    # 上行平均吞吐量（Mbps），0-500，通常比下行小很多
    "throughput_ul": spaces.Box(low=0.0, high=500.0, shape=(1,), dtype=np.float32),
    # 平均包时延（ms），5-100
    "delay": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
    # 平均 SINR（dB），-10 到 30
    "sinr": spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32),
    # 能量效率（Mbps/W），0-100，表示每瓦功耗能传输多少数据
    "energy_efficiency": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
    # 切换成功率（%），80-100
    "handover_success_rate": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
    # PRB 利用率（%），0-100
    "prb_utilization": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
    # RRC 连接数（个），0-500
    "rrc_connections": spaces.Box(low=0.0, high=500.0, shape=(1,), dtype=np.float32),
})

ACTION_SPACE = spaces.Dict({
    "downtilt": spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
    "drx_cycle": spaces.Discrete(3),
})

# ACTION_SPACE = spaces.Dict({
#     # 天线下倾角（度），-10 到 10
#     # 正数=往下倾斜，缩小覆盖；负数=往上倾斜，扩大覆盖
#     "downtilt": spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
#     # 发射功率偏置（dB），-6 到 6
#     # 正值=增大功率（覆盖更好但费电、干扰大），负值=降低功率
#     "tx_power_offset": spaces.Box(low=-6.0, high=6.0, shape=(1,), dtype=np.float32),
#     # 上行功控基准 P0（dBm），-126 到 -96
#     # 值越高，手机发射功率越大，上行信号越好但手机更费电
#     "p0_nominal_pusch": spaces.Box(low=-126.0, high=-96.0, shape=(1,), dtype=np.float32),
#     # DRX 周期：0=320ms, 1=640ms, 2=1280ms
#     # 越大越省电，但用户响应越慢
#     "drx_cycle": spaces.Discrete(3),
#     # CSI-RS 周期：0=20ms, 1=40ms, 2=80ms, 3=160ms
#     # CSI-RS 用于测量信道质量，周期越小测量越准但开销越大
#     "csi_rs_period": spaces.Discrete(4),
# })

# DRX 周期映射表（动作值→实际ms值）
DRX_CYCLE_MAP = {0: 320, 1: 640, 2: 1280}
# CSI-RS 周期映射表
CSI_RS_PERIOD_MAP = {0: 20, 1: 40, 2: 80, 3: 160}


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """将字典观测展平为一维 numpy 数组。

    某些 RL 算法（如 DQN 的 MLP 输入）需要扁平向量而非字典。
    这个函数提供一个标准化的转换。

    参数:
        obs: 字典格式的观测
    返回:
        一维 float32 数组，形状 (8,)
    """
    ordered_keys = [
        "throughput_dl", "throughput_ul", "delay", "sinr",
        "energy_efficiency", "handover_success_rate",
        "prb_utilization", "rrc_connections"
    ]
    return np.concatenate([obs[k].flatten() for k in ordered_keys]).astype(np.float32)


def flatten_action(action: Dict[str, np.ndarray]) -> np.ndarray:
    """将字典动作展平为一维 numpy 数组。

    参数:
        action: 字典格式的动作
    返回:
        一维 float32 数组
    """
    ordered_keys = ["downtilt", "tx_power_offset", "p0_nominal_pusch"]
    continuous = np.concatenate([action[k].flatten() for k in ordered_keys])
    discrete = np.array([float(action["drx_cycle"]), float(action["csi_rs_period"])], dtype=np.float32)
    return np.concatenate([continuous, discrete])


# ===========================================================================
# 环境主类
# ===========================================================================

class SimplifiedWirelessEnv(gym.Env):
    """简化版无线网络优化环境。

    模拟 3 个基站、100 个用户的 5G 蜂窝网络。
    Agent 通过调整基站参数来优化网络 KPI。

    使用示例：
        >>> config = WirelessEnvConfig(num_bs=3, num_users=100)
        >>> env = SimplifiedWirelessEnv(config)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()  # 随机动作
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    # 元数据：兼容 Gymnasium 接口
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: WirelessEnvConfig = None,
                 reward_weights: Dict[str, float] = None):
        """
        参数:
            config: 仿真配置，为 None 时使用默认值
            reward_weights: 奖励权重字典。
                           为 None 时使用 config 里的默认值。
                           LLM 引导时可以动态修改这个字典。
        """
        super().__init__()

        # ---- 配置 ----
        self.config = config if config is not None else WirelessEnvConfig()
        self.reward_weights = (reward_weights if reward_weights is not None
                               else self.config.default_reward_weights.copy())

        # ---- 定义 Gym 空间（使用全局常量）----
        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE

        # ---- 内部状态（在 reset() 中初始化）----
        self.step_count: int = 0
        self.user_positions: np.ndarray = None  # 形状 (num_users, 2)
        self.user_velocities: np.ndarray = None  # 形状 (num_users, 2)
        self.bs_positions: np.ndarray = None  # 形状 (num_bs, 2)
        self.kpi_history: list = []  # 历史 KPI，用于延迟反馈
        self.rng: np.random.Generator = None  # 随机数生成器

    # ------------------------------------------------------------------
    # 场景初始化与重置
    # ------------------------------------------------------------------

    def reset(self, seed: int = None,
              options: dict = None) -> Tuple[Dict[str, np.ndarray], dict]:
        """重置环境到初始状态。

        这是 Gym 标准接口，在每个 Episode 开始时调用。
        1. 重新随机生成基站位置和用户位置
        2. 重置步数计数器
        3. 返回初始观测

        参数:
            seed: 随机种子（用于可复现性）
            options: 额外选项（Gym 标准参数，这里暂不使用）
        返回:
            observation: 字典格式的初始 KPI 观测
            info: 额外信息字典（这里为空）
        """
        # 调用父类 reset，处理种子
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # ---- 1. 生成基站位置 ----
        # 3 个基站呈等边三角形分布，覆盖 500m×500m 区域
        self.bs_positions = self._generate_bs_positions()

        # ---- 2. 生成用户初始位置 ----
        # 使用泊松点过程（在区域内随机均匀撒点，有聚集效应）
        self.user_positions = self._generate_user_positions()

        # ---- 3. 生成用户移动速度和方向 ----
        self.user_velocities = self._generate_user_velocities()

        # ---- 4. 重置内部状态 ----
        self.step_count = 0
        self.kpi_history = []

        # ---- 5. 计算初始 KPI ----
        # 使用默认动作（所有参数取中间值）计算初始观测
        default_action = {
            "downtilt": np.array([-8.0], dtype=np.float32),
            "tx_power_offset": np.array([-5.0], dtype=np.float32),
            "p0_nominal_pusch": np.array([-120], dtype=np.float32),  # -96 和 -126 中间
            "drx_cycle": 2,   # 原来是 1 (640ms)，改成 1280ms
            "csi_rs_period": 3,  # 原来是 1 (40ms)，改成 160ms
        }
        obs = self._simulate_network(default_action)

        # ---- 6. 初始化 KPI 历史（用于后续滑动窗口）----
        self.kpi_history = [obs.copy() for _ in range(5)]  # 保留 5 步历史

        return obs, {}

    def _generate_bs_positions(self) -> np.ndarray:
        """生成基站位置。

        返回:
            numpy 数组，形状 (num_bs, 2)，每行是 (x, y) 坐标
        """
        if self.config.num_bs == 3:
            # 3 基站等边三角形排列，覆盖正方形区域
            size = self.config.area_size
            return np.array([
                [size / 2, size * 0.2],  # 上方基站
                [size * 0.2, size * 0.8],  # 左下基站
                [size * 0.8, size * 0.8],  # 右下基站
            ])
        else:
            # 均匀分布在区域边界上
            angles = np.linspace(0, 2 * np.pi, self.config.num_bs, endpoint=False)
            radius = self.config.area_size * 0.4
            center = self.config.area_size / 2
            return np.column_stack([
                center + radius * np.cos(angles),
                center + radius * np.sin(angles)
            ])

    def _generate_user_positions(self) -> np.ndarray:
        """使用泊松点过程生成用户初始位置。

        泊松点过程会让用户有自然聚集的趋势（类似真实场景中
        某些区域人更多），而不是完全均匀分布。

        返回:
            numpy 数组，形状 (num_users, 2)，每行是 (x, y) 坐标
        """
        # 用均匀分布模拟，加上一些热点（高斯混合）
        users = self.rng.uniform(0, self.config.area_size,
                                 (self.config.num_users, 2))

        # 可以在这里添加热点（例如在某个基站附近聚集更多用户）
        # 例如：把 20% 的用户移动到某个热点区域
        hotspot_center = self.bs_positions[0]  # 以第一个基站为中心
        n_hotspot = self.config.num_users // 5
        if n_hotspot > 0:
            users[:n_hotspot] = hotspot_center + self.rng.normal(
                0, self.config.area_size * 0.05, (n_hotspot, 2)
            )
            # 确保热点用户不跑出边界
            users = np.clip(users, 0, self.config.area_size)

        return users

    def _generate_user_velocities(self) -> np.ndarray:
        """生成用户移动速度向量。

        每个用户有随机的速度和移动方向。
        速度服从对数正态分布（大部分用户慢速步行，少数快速移动）。

        返回:
            numpy 数组，形状 (num_users, 2)，每行是 (vx, vy)
        """
        # 速度大小：从对数正态分布采样（大部分人在 1-3 m/s）
        speeds = self.rng.lognormal(
            mean=np.log(self.config.user_speed_mean),
            sigma=self.config.user_speed_std,
            size=self.config.num_users
        )
        # 移动方向：均匀随机
        angles = self.rng.uniform(0, 2 * np.pi, self.config.num_users)
        # 分解为 x 和 y 分量
        vx = speeds * np.cos(angles)
        vy = speeds * np.sin(angles)
        return np.column_stack([vx, vy])

    # ------------------------------------------------------------------
    # 核心仿真：动作 → KPI
    # ------------------------------------------------------------------

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], float, bool, bool, dict
    ]:
        """执行一步仿真。

        这是 Gym 标准接口的核心。Agent 给出动作（要调整的参数），
        环境返回新的观测（KPI）和奖励。

        流程：
        1. 更新用户位置（模拟移动）
        2. 根据 action 给定的参数，重算全网 KPI
        3. 计算奖励
        4. 判断是否结束

        参数:
            action: 字典格式的动作，键名与 ACTION_SPACE 一致

        返回:
            observation: 字典格式的新 KPI 观测
            reward: 标量奖励值
            terminated: 是否达到终止条件
            truncated: 是否被截断（这里始终为 False）
            info: 额外信息（包含原始 KPI 字典）
        """
        self.step_count += 1

        # ---- 1. 更新用户位置（模拟移动）----
        self._update_user_positions()

        # ---- 2. 运行动作，计算 KPI ----
        obs = self._simulate_network(action)

        # ---- 3. 更新 KPI 历史（用于延迟反馈）----
        self.kpi_history.append(obs.copy())
        if len(self.kpi_history) > 5:
            self.kpi_history.pop(0)

        # ---- 4. 计算奖励 ----
        reward = self._compute_reward(obs)

        # ---- 5. 判断终止 ----
        terminated = (self.step_count >= self.config.max_steps)
        truncated = False

        return obs, reward, terminated, truncated, {"kpis": obs,
                                                    "step": self.step_count}

    def _update_user_positions(self):
        """更新用户位置（随机游走模型）。

        每个用户按自己的速度向量移动一步（1 秒时间步长）。
        碰到边界时反弹（简单的边界处理）。
        """
        # 移动
        self.user_positions = self.user_positions + self.user_velocities

        # 边界反弹：如果用户跑出仿真区域，反转速度分量
        for i in range(self.config.num_users):
            for dim in range(2):  # x 和 y 两个维度
                if self.user_positions[i, dim] < 0:
                    self.user_positions[i, dim] = 0
                    self.user_velocities[i, dim] *= -1  # 反向
                elif self.user_positions[i, dim] > self.config.area_size:
                    self.user_positions[i, dim] = self.config.area_size
                    self.user_velocities[i, dim] *= -1

        # 小概率随机改变方向（模拟用户行为的不确定性）
        direction_change_mask = self.rng.random(self.config.num_users) < 0.05
        if direction_change_mask.any():
            new_angles = self.rng.uniform(0, 2 * np.pi, direction_change_mask.sum())
            speeds = np.linalg.norm(self.user_velocities[direction_change_mask], axis=1)
            self.user_velocities[direction_change_mask, 0] = speeds * np.cos(new_angles)
            self.user_velocities[direction_change_mask, 1] = speeds * np.sin(new_angles)

    def _simulate_network(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """核心仿真函数：根据动作参数计算全网 KPI。

        这是整个环境最核心的部分。采用 3GPP TR 38.901 的 UMa 场景
        路径损耗模型，加入阴影衰落和天线增益，计算每个用户的 SINR，
        然后映射到吞吐量等 KPI。

        步骤：
        1. 解析动作参数（包括连续的物理量和离散的配置选项）
        2. 对每个用户，计算到各基站的距离和路径损耗
        3. 计算信号强度（考虑天线增益和功率偏置）
        4. 计算 SINR（有用信号 / 干扰+噪声）
        5. SINR→CQI→频谱效率→吞吐量
        6. 汇总统计全网 KPI

        参数:
            action: 字典，包含 downtilt, tx_power_offset, p0_nominal_pusch,
                    drx_cycle, csi_rs_period
        返回:
            kpis: 字典，键名与 OBSERVATION_SPACE 一致，值都是形状为(1,)的 numpy 数组
        """
        # ---- 解析动作 ----
        downtilt = float(action.get("downtilt", np.array([0.0])).item())
        tx_power_offset = float(action.get("tx_power_offset", np.array([0.0])).item())
        p0_pusch = float(action.get("p0_nominal_pusch", np.array([-111.0])).item())
        drx_cycle_val = DRX_CYCLE_MAP.get(int(action.get("drx_cycle", 1)), 640)
        csi_rs_period_val = CSI_RS_PERIOD_MAP.get(int(action.get("csi_rs_period", 1)), 40)

        # ---- 预计算常量 ----
        # 实际发射功率（dBm）= 配置值 + 偏置
        actual_tx_power = self.config.tx_power + tx_power_offset
        # 噪声功率（dBm）= 热噪声基底 + 10*log10(带宽) + 噪声系数
        # 热噪声基底是物理常数：-174 dBm/Hz，表示 1Hz 带宽内的噪声功率
        # 乘以带宽（20MHz）后，噪声功率上升约 73dB
        noise_power = (self.config.noise_floor
                       + 10 * np.log10(self.config.bandwidth)
                       + self.config.noise_figure)

        # ---- 初始化统计变量 ----
        total_sinr = 0.0  # 所有用户 SINR 之和（用于求平均）
        total_throughput_dl = 0.0  # 所有用户下行吞吐量之和
        total_throughput_ul = 0.0  # 所有用户上行吞吐量之和
        total_delay = 0.0  # 所有用户时延之和
        handover_count = 0  # 切换次数（用户切换了服务基站）
        total_power_consumption = 0.0  # 各基站功耗之和

        # ---- 对每个用户计算 ----
        for i in range(self.config.num_users):
            ue_pos = self.user_positions[i]  # 当前用户坐标 (x, y)

            # ==== 步骤 1：计算用户到每个基站的距离和路径损耗 ====
            rsrps = []  # RSRP: Reference Signal Received Power，参考信号接收功率（dBm）
            # RSRP 就是用户测量到的基站信号强度，是选择服务基站的关键指标

            for j in range(self.config.num_bs):
                bs_pos = self.bs_positions[j]

                # 2D 距离（水平面上的距离）
                d_2d = np.sqrt((ue_pos[0] - bs_pos[0]) ** 2 +
                               (ue_pos[1] - bs_pos[1]) ** 2)
                # 避免距离为 0（用户不可能和基站在同一位置）
                d_2d = max(d_2d, 1.0)

                # 3D 距离（考虑基站和用户的高度差）
                d_3d = np.sqrt(d_2d ** 2 +
                               (self.config.bs_height - self.config.ue_height) ** 2)

                # ---- 路径损耗计算（3GPP TR 38.901 UMa LOS）----
                # 公式: PL = 28.0 + 22*log10(d_3D) + 20*log10(f_c)
                # 这是经过大量实测数据拟合出来的统计模型
                # 距离越远、频率越高，路径损耗越大
                pathloss = (self.config.pathloss_a
                            + self.config.pathloss_b * np.log10(d_3d)
                            + self.config.pathloss_c * np.log10(self.config.carrier_freq / 1e9))

                # ---- 阴影衰落 ----
                # 对数正态分布：在 dB 域上加一个随机偏移
                # 使用基于位置的伪随机（同一位置总是得到同样的衰落值）
                shadowing = self._get_shadowing(ue_pos[0], ue_pos[1], j)
                pathloss += shadowing

                # ---- 天线增益 ----
                # 天线不是全向的，增益取决于用户偏离天线主瓣的角度
                # 主瓣方向由下倾角决定
                antenna_gain = self._compute_antenna_gain(d_2d, downtilt, j)

                # ---- 计算 RSRP ----
                # RSRP = 发射功率 - 路径损耗 + 天线增益
                rsrp = actual_tx_power - pathloss + antenna_gain
                rsrps.append(rsrp)

            # ==== 步骤 2：确定服务基站 ====
            # 用户连接到 RSRP 最强的基站
            serving_bs = int(np.argmax(rsrps))
            serving_rsrp = rsrps[serving_bs]

            # ==== 步骤 3：计算 SINR ====
            # SINR = 有用信号功率 / (干扰功率之和 + 噪声功率)
            # 这里简化：有用信号=服务基站 RSRP，干扰=其他基站 RSRP 之和
            # 注意：RSRP 是 dBm，需要转成线性值（毫瓦）才能加减，然后再转回 dB

            # 转成线性值（mW）
            serving_rsrp_linear = 10 ** (serving_rsrp / 10)  # 有用信号
            interference_linear = 0.0
            for j in range(self.config.num_bs):
                if j != serving_bs:
                    interference_linear += 10 ** (rsrps[j] / 10)  # 干扰信号
            noise_linear = 10 ** (noise_power / 10)

            # SINR（线性值）
            sinr_linear = serving_rsrp_linear / (interference_linear + noise_linear + 1e-10)
            # 转回 dB
            sinr_db = 10 * np.log10(sinr_linear + 1e-10)
            # 裁剪到合理范围 [-10, 30] dB
            sinr_db = np.clip(sinr_db, -10, 30)

            # ==== 步骤 4：SINR → CQI → 吞吐量 ====
            cqi = self._sinr_to_cqi(sinr_db)
            spectral_efficiency = self.config.cqi_efficiency_table.get(cqi, 0.15)
            # 吞吐量 = 频谱效率 × 带宽
            throughput_dl = spectral_efficiency * (self.config.bandwidth / 1e6)  # bps→Mbps
            # 上行吞吐量通常是下行的 20%-40%
            # p0_pusch 控制上行功率：p0 越高，UE 发射功率越大，上行越好
            ul_factor = 0.25 + 0.15 * (p0_pusch + 126) / 30  # 映射 [-126,-96] 到 [0.25, 0.4]
            throughput_ul = throughput_dl * np.clip(ul_factor, 0.2, 0.45)

            # ==== 步骤 5：时延估计 ====
            # 时延和多个因素有关：
            # - 吞吐量越高，数据传得快，时延低
            # - DRX 周期越大，用户可能"睡"更久，响应慢
            # - 用户数越多，调度排队时间长
            # 基础传播时延（5G NR 典型 1-5ms，教学楼/体育馆场景更小）
            base_delay = 2
            # 传输时延：假设平均数据包 100KB = 0.8 Mbit
            transmission_delay = 0.8 / (throughput_dl + 0.01)  # 单位：秒
            transmission_delay_ms = transmission_delay * 1000  # 转为毫秒
            # DRX 额外等待（平均半个周期）
            drx_penalty = drx_cycle_val / 8  # 从 /2 改为 /4,  从 /4 降到 /8，弱化 DRX 对时延的极端影响
            # 用户数对排队时延的影响（限制在一个合理范围）
            queueing_delay = (self.config.num_users / 200) * 5  # 100用户约 2.5ms

            delay = base_delay + transmission_delay_ms + drx_penalty + queueing_delay
            delay = np.clip(delay, 2, 200)  # 下限 2ms（物理极限），上限 200ms

            # ---- 累加统计 ----
            total_sinr += sinr_db
            total_throughput_dl += throughput_dl
            total_throughput_ul += throughput_ul
            total_delay += delay

        # ==== 步骤 6：汇总全网 KPI ====
        n_users = self.config.num_users
        avg_sinr = total_sinr / n_users
        avg_throughput_dl = total_throughput_dl / n_users
        avg_throughput_ul = total_throughput_ul / n_users
        avg_delay = total_delay / n_users

        # ---- 功耗计算 ----
        # 基站功耗 = 静态功耗 + 负载相关动态功耗
        # 负载用下行吞吐量来近似
        load_factor = np.clip(avg_throughput_dl / 200, 0, 1)  # 假设200Mbps为满载
        # 功率偏置也会影响功耗：增大发射功率，功耗相应增加
        power_per_bs = (self.config.p_static
                        + load_factor * self.config.p_dynamic
                        + tx_power_offset * 2)  # 功率增加会额外耗电
        power_per_bs = max(power_per_bs, self.config.p_static * 0.5)  # 不低于静态功耗一半
        total_power = power_per_bs * self.config.num_bs

        # ---- 能效 ----
        # 能效 = 总吞吐量 / 总功耗，单位 Mbps/W
        energy_eff = (avg_throughput_dl * n_users) / (total_power + 1e-10) * 100
        energy_eff = np.clip(energy_eff, 0, 100)

        # ---- PRB 利用率 ----
        # PRB 利用率 ≈ 当前总吞吐量 / 理论最大吞吐量
        # 理论最大：最高频谱效率 × 带宽 × PRB 总数
        max_spectral_eff = max(self.config.cqi_efficiency_table.values())
        max_throughput = max_spectral_eff * (self.config.bandwidth / 1e6)
        prb_util = np.clip((avg_throughput_dl / max_throughput) * 100, 0, 100)

        # ---- 切换成功率 ----
        # 当用户信号质量差、同时另一个基站信号好时，发生切换
        # 切换成功率受下倾角影响：下倾角太大覆盖突变，切换容易失败
        # 简化：基础成功率 98%，下倾角偏离最优时降低
        base_success = 98.0
        # 下倾角偏移惩罚：最优下倾角假设在 3-6 度（城区宏站典型值）
        optimal_downtilt_range = (3, 6)
        if downtilt < optimal_downtilt_range[0]:
            penalty = (optimal_downtilt_range[0] - downtilt) * 0.5
        elif downtilt > optimal_downtilt_range[1]:
            penalty = (downtilt - optimal_downtilt_range[1]) * 0.5
        else:
            penalty = 0
        handover_success = np.clip(base_success - penalty, 85, 100)

        # ---- RRC 连接数 ----
        # 简化为用户数 + 小幅随机波动
        rrc_connections = n_users + self.rng.integers(-5, 6)

        # ---- 组装观测字典 ----
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

    def _get_shadowing(self, x: float, y: float, bs_id: int) -> float:
        """获取指定位置的阴影衰落值。

        阴影衰落模拟建筑物、地形等大型障碍物对信号的影响。
        使用基于位置的确定性伪随机，保证同一位置始终得到同样的值
        （这对 RL 的可复现性很重要——如果阴影每次随机变，
        Agent 观察到的状态变化可能来自阴影波动而非自身动作）。

        参数:
            x, y: 用户位置
            bs_id: 基站编号（用于生成与基站相关的阴影）
        返回:
            阴影衰落值（dB），服从 N(0, sigma^2)
        """
        # 用坐标+基站ID作为种子，保证同一位置同一基站始终得到相同的阴影
        seed_val = int(abs(hash((int(x * 100), int(y * 100), bs_id))) % (2 ** 31))
        local_rng = np.random.default_rng(seed_val)
        return local_rng.normal(0, self.config.shadowing_std)

    def _compute_antenna_gain(self, d_2d: float, downtilt: float, bs_id: int) -> float:
        """计算天线在用户方向上的增益。

        天线通常不是全向的——它在主瓣方向增益最大，
        越偏离主瓣方向增益越小。下倾角决定了主瓣的"俯仰"方向。

        简化模型：
        - 水平方向：用户偏离主瓣水平角度的衰减
        - 垂直方向：下倾角决定了主瓣在垂直面上的指向

        参数:
            d_2d: 用户到基站的水平距离
            downtilt: 天线下倾角（度）
            bs_id: 基站编号
        返回:
            天线增益（dBi）
        """
        # ---- 垂直衰减 ----
        # 主瓣在垂直面上的指向角度 = 下倾角
        # 当用户所在角度 = 主瓣指向时，增益最大
        # 用户所在角度 = arctan(基站高度 / 水平距离) —— 这是基站看用户的仰角
        bs_pos = self.bs_positions[bs_id]

        # 计算基站到用户的仰角（度）
        # 仰角 = arctan(高度差 / 水平距离)，再换算成度
        elevation_angle = np.degrees(np.arctan(
            (self.config.bs_height - self.config.ue_height) / max(d_2d, 1.0)
        ))

        # 主瓣指向角度 = 下倾角（向下倾斜，所以是负的仰角方向）
        # 用户仰角与主瓣指向的偏差
        angle_diff = elevation_angle - downtilt

        # 垂直波束衰减：偏差越大，增益越低
        # 使用高斯型衰减（接近真实天线方向图）
        # 垂直 3dB 波束宽度约 10°
        vertical_attenuation = -24 * (angle_diff / self.config.beamwidth_v) ** 2  #从12改为24

        # ---- 水平衰减 ----
        # 简化：假设用户均匀分布在水平面，取平均水平增益
        # 完整实现需要计算用户方位角与天线朝向的偏差
        horizontal_attenuation = 0  # 简化：不考虑水平方向偏差

        # ---- 总增益 ----
        gain = self.config.antenna_gain_max + vertical_attenuation + horizontal_attenuation
        # 增益不能低于 -20 dBi（实际天线有最低增益限制）
        gain = max(gain, -20.0)

        return gain

    def _sinr_to_cqi(self, sinr_db: float) -> int:
        """将 SINR（dB）映射为 CQI 值（0-15）。

        CQI (Channel Quality Indicator) 是 UE 根据测量的 SINR
        向基站报告的信道质量。基站根据 CQI 选择 MCS。

        映射关系来自 TS 38.214 的简化版本。

        参数:
            sinr_db: SINR 值，单位 dB
        返回:
            CQI 值，范围 1-15
        """
        # 查表：找到 SINR 超过的最后一个阈值
        cqi = 1  # 最差质量，对应最低调制方式
        for threshold, cqi_val in self.config.sinr_cqi_table:
            if sinr_db >= threshold:
                cqi = cqi_val
            else:
                break
        return cqi

    # ------------------------------------------------------------------
    # 奖励计算
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """根据当前观测和奖励权重计算标量奖励。

        奖励 = sum(weight_i * normalized_kpi_i)

        每个 KPI 先做归一化，再乘以权重后求和。
        LLM 引导时可以动态修改 self.reward_weights。

        参数:
            obs: 当前 KPI 观测字典
        返回:
            标量奖励值
        """
        reward = 0.0

        # 归一化参考值（用于把不同量纲的 KPI 映射到相似范围）
        # 这些值是基于典型 5G 网络的大致范围
        normalizers = {
            "throughput_dl": 100.0,  # 原来是 200，现在环境吞吐量 ~40，降到 100 让它贡献更大
            "throughput_ul": 50.0,
            "delay": 200.0,  # 原来是 50，现在真实时延 100-180，提到 100 降低惩罚力度,提到200继续降低力度
            "energy_efficiency": 50.0,
            "handover_success_rate": 100.0,
        }

        if "throughput_dl" in self.reward_weights:
            reward += (self.reward_weights["throughput_dl"]
                       * obs["throughput_dl"].item() / normalizers["throughput_dl"])

        if "throughput_ul" in self.reward_weights:
            reward += (self.reward_weights["throughput_ul"]
                       * obs["throughput_ul"].item() / normalizers["throughput_ul"])

        if "delay" in self.reward_weights:
            # 时延越小越好，所以取负号
            reward += (self.reward_weights["delay"]
                       * obs["delay"].item() / normalizers["delay"])

        if "energy_efficiency" in self.reward_weights:
            reward += (self.reward_weights["energy_efficiency"]
                       * obs["energy_efficiency"].item() / normalizers["energy_efficiency"])

        if "handover_success_rate" in self.reward_weights:
            reward += (self.reward_weights["handover_success_rate"]
                       * obs["handover_success_rate"].item() / normalizers["handover_success_rate"])

        return float(reward)

    # ------------------------------------------------------------------
    # 其他辅助方法
    # ------------------------------------------------------------------

    def set_reward_weights(self, weights: Dict[str, float]):
        """动态设置奖励权重（LLM 引导时使用）。

        参数:
            weights: 新的权重字典
        """
        self.reward_weights = weights.copy()

    def get_scenario_summary(self) -> str:
        """获取当前场景的文字摘要（供 LLM 分析使用）。

        返回:
            描述当前网络状态的简短文本
        """
        if not self.kpi_history:
            return "无历史数据"

        latest = self.kpi_history[-1]
        return (
            f"步数: {self.step_count}, "
            f"下行吞吐量: {latest['throughput_dl'].item():.1f} Mbps, "
            f"上行吞吐量: {latest['throughput_ul'].item():.1f} Mbps, "
            f"时延: {latest['delay'].item():.1f} ms, "
            f"SINR: {latest['sinr'].item():.1f} dB, "
            f"能效: {latest['energy_efficiency'].item():.1f} Mbps/W, "
            f"切换成功率: {latest['handover_success_rate'].item():.1f}%"
        )


# 新文件：wrappers.py，或加在 wireless_env.py 末尾

class ActionNormalizer(gym.ActionWrapper):
    """将 Agent 输出的 [-1,1] 连续动作映射回真实物理范围。

    只处理连续动作维度（downtilt, tx_power_offset, p0_nominal_pusch），
    离散动作（drx_cycle, csi_rs_period）原样透传。
    """

    def __init__(self, env):
        super().__init__(env)
        # 原始连续动作空间（从 ACTION_SPACE 中提取连续部分）
        self.cont_low = np.array([-10.0, -6.0, -126.0], dtype=np.float32)
        self.cont_high = np.array([10.0, 6.0, -96.0], dtype=np.float32)
        # 替换连续部分为 [-1,1]
        new_cont_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Dict({
            "continuous": new_cont_space,
            "drx_cycle": spaces.Discrete(4),
            "csi_rs_period": spaces.Discrete(4),
        })

    def action(self, action):
        """[-1,1] → 物理范围映射（Gym 自动调用）"""
        cont = np.asarray(action["continuous"], dtype=np.float32).flatten()
        # 手动线性映射：[-1,1] → [low, high]
        real_cont = self.cont_low + (cont + 1.0) / 2.0 * (self.cont_high - self.cont_low)
        return {
            "downtilt": np.array([real_cont[0]], dtype=np.float32),
            "tx_power_offset": np.array([real_cont[1]], dtype=np.float32),
            "p0_nominal_pusch": np.array([real_cont[2]], dtype=np.float32),
            "drx_cycle": int(action["drx_cycle"]),
            "csi_rs_period": int(action["csi_rs_period"]),
        }

# ===========================================================================
# 测试代码（可以用 `python wireless_env.py` 直接运行）
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试：SimplifiedWirelessEnv 基本功能")
    print("=" * 60)

    # 1. 创建环境
    config = WirelessEnvConfig(num_bs=3, num_users=100, max_steps=10)
    env = SimplifiedWirelessEnv(config)

    # 2. 重置
    obs, info = env.reset(seed=42)
    print(f"\n初始观测（键值）:")
    for key, value in obs.items():
        print(f"  {key:25s}: {value.item():.2f}")

    # 3. 跑几步随机动作
    print(f"\n随机动作测试:")
    total_reward = 0
    for step_idx in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  步 {step_idx + 1}: reward={reward:.3f}, sinr={obs['sinr'].item():.1f} dB, "
              f"吞吐量={obs['throughput_dl'].item():.1f} Mbps, 时延={obs['delay'].item():.1f} ms")
        if terminated:
            break

    print(f"\n总奖励: {total_reward:.3f}")
    print("测试通过！")

    # === 新增：恒定默认动作测试（坏参数版）===
    print("\n" + "=" * 60)
    print("基线测试：恒定坏默认动作（200步总Reward）")
    print("=" * 60)
    env_test = SimplifiedWirelessEnv(config)
    obs, _ = env_test.reset()
    total_r = 0
    bad_action = {
        "downtilt": np.array([-8.0], dtype=np.float32),
        "tx_power_offset": np.array([-5.0], dtype=np.float32),
        "p0_nominal_pusch": np.array([-120.0], dtype=np.float32),
        "drx_cycle": 2,
        "csi_rs_period": 3,
    }
    for _ in range(200):
        obs, reward, _, _, _ = env_test.step(bad_action)
        total_r += reward
    print(f"恒定坏默认动作 200步总Reward: {total_r:.1f}")