import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from wireless_env import SimplifiedWirelessEnv

class EvaluateActionWrapper(gym.Wrapper):
    """为环境添加评估动作的能力，不推进环境状态。

    用于 GRPO 的组内多动作采样——对同一个状态尝试 G 个动作，
    用环境真实奖励做横向对比。

    原理：在执行每个评估动作前，保存环境内部状态（用户位置、
    速度、步数、KPI历史），执行后恢复。环境对外完全无感知。
    """

    def __init__(self, env):
        super().__init__(env)

    def evaluate_action(self, state_vec: np.ndarray,
                        action_dict: Dict[str, np.ndarray]) -> float:
        """评估一个动作，返回奖励，不推进环境状态。

        参数:
            state_vec: 展平后的状态向量，形状 (8,)
            action_dict: 动作字典
        返回:
            reward: 标量奖励值
        """
        env_inner = self.env
        # 如果外面还包了其他 wrapper，逐层解开找到 SimplifiedWirelessEnv
        while hasattr(env_inner, 'env') and not isinstance(env_inner, SimplifiedWirelessEnv):
            env_inner = env_inner.env

        # 保存当前环境状态
        saved_positions = env_inner.user_positions.copy()
        saved_velocities = env_inner.user_velocities.copy()
        saved_step_count = env_inner.step_count
        saved_kpi_history = [h.copy() for h in env_inner.kpi_history]

        # 临时计算 KPI 和奖励
        obs = env_inner._simulate_network(action_dict)
        reward = env_inner._compute_reward(obs)

        # 恢复环境状态
        env_inner.user_positions = saved_positions
        env_inner.user_velocities = saved_velocities
        env_inner.step_count = saved_step_count
        env_inner.kpi_history = saved_kpi_history

        return reward