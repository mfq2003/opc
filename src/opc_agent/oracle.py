"""本模块提供 PPO 精算阶段的受限动作环境和训练前置条件检查。

输入为点级状态向量、可调用的精算评价函数及 ±40nm 离散动作；输出为 Gymnasium 环境奖励和 PPO 训练器。
关键依赖为 Gymnasium、Stable-Baselines3 与 OpenILT 适配层；真实计算仅在数据与 GPU 后端均就绪时启动。
"""
from __future__ import annotations

from typing import Callable, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .metrics import DISPLACEMENT_CLASSES_NM


class ClipPPOEnv(gym.Env):
    """将一个 clip 的状态和有限法线动作包装成可由 PPO 优化的环境。"""

    metadata = {"render_modes": []}

    def __init__(self, initial_state: np.ndarray, evaluator: Callable[[int], Tuple[float, float, float]], max_steps: int = 1):
        super().__init__()
        self.initial_state = np.asarray(initial_state, dtype=np.float32)
        if self.initial_state.ndim != 1:
            raise ValueError("PPO 状态必须是一维向量")
        self.evaluator = evaluator
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.initial_state.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(DISPLACEMENT_CLASSES_NM))
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return self.initial_state.copy(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("动作不在九分类位移集合中")
        l2, epe, pvb = self.evaluator(DISPLACEMENT_CLASSES_NM[action])
        self._steps += 1
        reward = -(l2 + 100.0 * epe + pvb)
        terminated = self._steps >= self.max_steps
        return self.initial_state.copy(), float(reward), terminated, False, {"displacement_nm": DISPLACEMENT_CLASSES_NM[action]}

